#!/usr/bin/env bash
# Gate 0 job queue (PROJECT_V2.md Section 7) for ONE GPU, from the repository root, after
# scripts/vm_setup.sh and `hf auth login`. Every step is resumable: rerunning skips finished outputs.
#
#   bash scripts/run_gate0.sh          # real run (VM)
#   bash scripts/run_gate0.sh --tiny   # tiny pure-PyTorch rehearsal (CPU, minutes; numbers meaningless)
#
# Lanes (run at the same time; memory caps per process add up to under the 96 GB card):
#   downloads  Qwen3-1.7B + PLTs, Gemma-2-2B + PLTs, Gemma CLT-426k, in parallel (network and disk only)
#   lane Q     gen_traces -> gate0_union GPU phase (cap 50 GB) -> gate0_columns (cap 40 GB)
#   lane U     gate0_union CPU phase (fidelity searches on saved graphs, while lane Q keeps the GPU busy)
#   lane G     gate0_short PLT (cap 40 GB) -> [after the union GPU phase] gate0_short CLT (cap 50 GB)
#   report     gate0_report once every lane has finished
# A K0.7 failure (exit code 3) stops Gate 0: the other lanes are stopped and no verdict is issued.
set -uo pipefail
cd "$(dirname "$0")/.."
TINY=""
[[ "${1:-}" == "--tiny" ]] && TINY="--tiny"
PY="${PY:-python}"
mkdir -p logs
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[$(stamp)] $*" | tee -a logs/run_gate0.log; }

if [[ -z "$TINY" ]]; then
  : "${HF_HOME:?set HF_HOME to a directory on the big disk first}"
  nvidia-smi | tee -a logs/run_gate0.log
  # never start on a GPU somebody else is using heavily
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if (( used > 4000 )); then log "GPU already has ${used} MiB in use: check nvidia-smi before starting"; exit 1; fi
  $PY scripts/download_models.py --set qwen3_1p7b_plt      > logs/dl_qwen.log 2>&1 & DL_Q=$!
  $PY scripts/download_models.py --set gemma2_2b_plt       > logs/dl_gemma_plt.log 2>&1 & DL_GP=$!
  $PY scripts/download_models.py --set gemma2_2b_clt426k   > logs/dl_gemma_clt.log 2>&1 & DL_GC=$!
else
  DL_Q=""; DL_GP=""; DL_GC=""
fi
waitfor() { [[ -n "$1" ]] && { wait "$1" || { log "download $2 failed (see logs)"; return 1; }; }; return 0; }

# ---- lane Q: traces, union growth (GPU), sampled columns -----------------------------------------
lane_q() {
  waitfor "$DL_Q" qwen || return 1
  if [[ -z "$TINY" ]]; then
    if [[ ! -f results/gate0/traces.json ]]; then
      $PY scripts/gen_traces.py || return 1
      git add results/gate0/traces.json configs/models/qwen3_1p7b_plt.yaml && \
        git commit -m "Gate 0: commit generated traces before analysis" || log "commit of traces failed: commit by hand"
    fi
  else
    $PY scripts/gen_traces.py --tiny --force || return 1
  fi
  CCT_MEMORY_CAP_GB=50 $PY scripts/gate0_union.py --phase gpu $TINY || return 1
  touch logs/.gate0_union_gpu_done
  CCT_MEMORY_CAP_GB=40 $PY scripts/gate0_columns.py $TINY
}

# ---- lane U: union CPU phase -----------------------------------------------------------------------
lane_u() {
  # waits for the traces file, then polls for graphs until the GPU phase writes its DONE marker
  until [[ -f results/gate0/traces.json || -n "$TINY" ]]; do sleep 30; done
  [[ -n "$TINY" ]] && until [[ -f results/tiny/gate0/traces.json ]]; do sleep 2; done
  $PY scripts/gate0_union.py --phase cpu $TINY
}

# ---- lane G: Gemma short prompts ---------------------------------------------------------------
lane_g() {
  waitfor "$DL_GP" gemma_plt || return 1
  CCT_MEMORY_CAP_GB=40 $PY scripts/gate0_short.py --transcoders plt $TINY || return $?
  waitfor "$DL_GC" gemma_clt || return 1
  # the CLT needs more memory: start it once the union GPU phase has released its share
  until [[ -f logs/.gate0_union_gpu_done ]]; do sleep 30; done
  CCT_MEMORY_CAP_GB=50 $PY scripts/gate0_short.py --transcoders clt $TINY
}

RUN=real; [[ -n "$TINY" ]] && RUN=tiny
rm -f logs/.gate0_union_gpu_done artifacts/$RUN/*/gate0_union/DONE
log "Gate 0 start ${TINY:-real}"
lane_q > logs/lane_q.log 2>&1 & LQ=$!
lane_u > logs/lane_u.log 2>&1 & LU=$!
lane_g > logs/lane_g.log 2>&1 & LG=$!

status=0
for pid_name in "$LQ:Q" "$LG:G" "$LU:U"; do
  pid=${pid_name%%:*}; name=${pid_name##*:}
  wait "$pid"; rc=$?
  log "lane $name finished with exit code $rc"
  if (( rc == 3 )); then
    log "K0.7 failed in lane $name: stopping the other lanes, no verdict (PREREG_GATE0)"
    kill "$LQ" "$LG" "$LU" 2>/dev/null
    status=3
    break
  fi
  (( rc != 0 )) && status=1
done
touch logs/.gate0_union_gpu_done  # release lane G if lane Q failed before writing it
wait 2>/dev/null
$PY scripts/gate0_report.py $TINY
log "Gate 0 done (status $status). Report: results/${TINY:+tiny/}gate0/GATE0_REPORT.md"
[[ -z "$TINY" ]] && nvidia-smi | tee -a logs/run_gate0.log
exit $status
