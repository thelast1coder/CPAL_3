#!/usr/bin/env bash
# Gate 0 premise tests (GATE0_PREMISES.md) on ONE GPU, from the repository root, after
# scripts/vm_setup.sh and `hf auth login`. Resumable: rerunning skips finished outputs.
#
#   bash scripts/run_premises.sh          # real run (VM, inside tmux)
#   bash scripts/run_premises.sh --tiny   # CPU rehearsal with the tiny model (numbers meaningless)
#
# Order and lanes (stop rules from PREREG_GATE0 are honoured: a stage that exits with code 2 stops
# every stage that depends on it, and the report is written):
#   0. downloads (Qwen3-1.7B, Gemma PLT, Gemma CLT) in parallel; traces generated and committed
#   1. Stage A (PR1 to PR3)                                    gate for everything else
#   2. lane B: Stage B (51 reference graphs, PR4)  ||  lane C: Stage C (full short matrices, PR5, PR6)
#   3. lane C continues with Stage D (needs epsilon* from C) while lane B is still building graphs
#   4. Stage F (needs B) || Stage E (needs B and D)
#   5. SWITCH-MODEL path only if PR4 fails on Qwen3-1.7B: Stage B on Qwen3-4B (0.5 h cap)
#   6. report: results/gate0/PREMISE_REPORT.md
set -uo pipefail
cd "$(dirname "$0")/.."
TINY=""
[[ "${1:-}" == "--tiny" ]] && TINY="--tiny"
PY="${PY:-python}"
RES=results/gate0; [[ -n "$TINY" ]] && RES=results/tiny/gate0
mkdir -p logs "$RES"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[$(stamp)] $*" | tee -a logs/run_premises.log; }
run() {  # run <name> <memory cap GB> <command...>; returns the command's exit code
  local name=$1 cap=$2; shift 2
  log "start $name"
  local t0=$SECONDS
  CCT_MEMORY_CAP_GB=$cap "$@" > "logs/premise_${name}.log" 2>&1
  local rc=$?
  log "end $name (exit $rc, $((SECONDS - t0)) s)"
  return $rc
}

if [[ -z "$TINY" ]]; then
  : "${HF_HOME:?set HF_HOME to a directory on the big disk first}"
  # pruning, influence and graph scores on the GPU (CPU fallback on out-of-memory): Stage C alone
  # runs over 200 prunes of graphs with thousands of nodes
  export CCT_PRUNE_DEVICE="${CCT_PRUNE_DEVICE:-cuda}"
  nvidia-smi | tee -a logs/run_premises.log
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if (( used > 4000 )); then log "GPU already has ${used} MiB in use: check nvidia-smi first"; exit 1; fi
  $PY scripts/download_models.py --set qwen3_1p7b_plt    > logs/dl_qwen.log 2>&1 & DL_Q=$!
  $PY scripts/download_models.py --set gemma2_2b_plt     > logs/dl_gemma_plt.log 2>&1 & DL_GP=$!
  $PY scripts/download_models.py --set gemma2_2b_clt426k > logs/dl_gemma_clt.log 2>&1 & DL_GC=$!
  wait $DL_Q || { log "Qwen download failed"; exit 1; }
  if [[ ! -f results/gate0/traces.json ]]; then
    run traces 60 $PY scripts/gen_traces.py || { log "trace generation failed"; exit 1; }
    git add results/gate0/traces.json && git commit -q -m "Gate 0: generated traces, committed before analysis" \
      || log "commit of traces failed: commit by hand before analysis"
  fi
  wait $DL_GP || { log "Gemma PLT download failed"; exit 1; }
  wait $DL_GC || { log "Gemma CLT download failed"; exit 1; }
else
  $PY scripts/gen_traces.py --tiny --force > logs/premise_traces.log 2>&1 || { log "tiny traces failed"; exit 1; }
fi

report() { $PY scripts/premise_report.py $TINY > logs/premise_report.log 2>&1; log "report: $RES/PREMISE_REPORT.md"; }

# ---- 1. Stage A: foundations -------------------------------------------------------------------
run A 85 $PY scripts/premise_A.py $TINY; rc=$?
if (( rc != 0 )); then log "Stage A stopped (exit $rc): no further stages"; report; exit $rc; fi

# ---- 2/3. lane B (graphs) || lane C -> D ---------------------------------------------------------
( run B 45 $PY scripts/premise_B.py $TINY ) & LB=$!
( run C 40 $PY scripts/premise_C.py $TINY && run D 40 $PY scripts/premise_D.py $TINY ) & LCD=$!
wait $LB; rcB=$?
wait $LCD; rcCD=$?
log "lane B exit $rcB; lane C/D exit $rcCD"

# ---- 4. Stage F (needs B) || Stage E (needs B and D) ----------------------------------------------
if (( rcB == 0 )); then
  ( run F 45 $PY scripts/premise_F.py $TINY ) & LF=$!
  if (( rcCD == 0 )); then ( run E 40 $PY scripts/premise_E.py $TINY ) & LE=$!; else LE=""; fi
  wait $LF; [[ -n "$LE" ]] && wait $LE
fi

# ---- 5. SWITCH-MODEL path (PR4 fails on Qwen3-1.7B) ------------------------------------------------
pr4=$($PY -c "import json,sys;d=json.load(open('$RES/stage_B.json'));print(d['premises']['PR4']['status'])" 2>/dev/null || echo missing)
if [[ "$pr4" == "fail" || "$pr4" == "kill" ]]; then
  log "PR4 is $pr4 on Qwen3-1.7B: SWITCH-MODEL check on Qwen3-4B (0.5 h cap)"
  if [[ -z "$TINY" ]]; then
    $PY scripts/download_models.py --set qwen3_4b_plt > logs/dl_qwen4b.log 2>&1 || log "Qwen3-4B download failed"
  fi
  timeout 2400 $PY scripts/premise_B.py --model-set qwen3_4b_plt $TINY > logs/premise_B_4b.log 2>&1
  rc4=$?
  case $rc4 in
    0) log "Stage B on Qwen3-4B finished: PR4 result in $RES/stage_B_qwen3_4b.json" ;;
    2) log "Stage B on Qwen3-4B finished with its stop rule (PR4 kill on both models)" ;;
    124) log "Stage B on Qwen3-4B hit its 40-minute cap (0.5 h budget plus loading)" ;;
    *) log "Stage B on Qwen3-4B failed (exit $rc4), see logs/premise_B_4b.log" ;;
  esac
fi

# ---- 6. report ----------------------------------------------------------------------------------
report
[[ -z "$TINY" ]] && nvidia-smi | tee -a logs/run_premises.log
log "Gate 0 premise tests done. Stop: waiting for Dawar's decision."
