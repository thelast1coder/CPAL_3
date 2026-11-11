# CCT_CPAL: Compressed Circuit Tracing

Attribution graphs from a few mixed backward passes. Specification: `PROJECT.md` (what and why),
`EXPERIMENTS.md` (phases and gates), `PHASE1_KILL_TEST.md` and `PREREG_KILL_TEST.md` (the go/no-go
test). Status: `results/STATUS.md`. Implementation decisions and open questions: `docs/decisions.md`.

## Layout
- `src/cct/operators/`: the operator interface (`base.py`), a dense operator for simulation, the tiny
  pure-PyTorch backend (CPU tests) and the circuit-tracer backend (VM).
- `src/cct/sensing`, `recovery`, `hubs.py`, `certify`, `graph`, `baselines`, `interventions`,
  `cost.py`, `pipeline.py` (`run_cct`), `killtest.py` (kill-test statistics and decision rules).
- `scripts/`: `sanity_k0.py`, `run_exact_graphs.py`, `analyse_structure.py` (K1),
  `simulate_sensing.py` (K2), `run_real_cct.py` (K3), `make_kill_report.py`, `run_kill_test.sh`,
  `vm_setup.sh`, `download_models.py`.
- `configs/`: models, prompts (frozen at the PREREG commit), the kill-test experiment and the prereg YAML.

## Laptop (CPU)
```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # Windows path
.venv/Scripts/python -m pip install numpy scipy pyyaml pytest
.venv/Scripts/python -m pip install -e . --no-deps
.venv/Scripts/python -m pytest -q                      # 31 tests, including the tiny end-to-end run
bash scripts/run_kill_test.sh --tiny                   # writes results/tiny/KILL_REPORT.md (TINY)
```

## VM (H100, shared): first commands for P1
```bash
git clone <private repo URL> CCT_CPAL && cd CCT_CPAL
export HF_HOME=/path/on/vm/big_disk/hf          # all downloads go here
df -h "$HF_HOME"                                 # about 150 GB free needed
nvidia-smi                                       # see what Mirror_CPAL, MoE_CPAL and the CVPR job use
bash scripts/vm_setup.sh                         # venv, CUDA torch, requirements, versions, pytest -q
source .venv/bin/activate
hf auth login                                    # fresh token (verify command); Gemma licence accepted
python scripts/download_models.py --set gemma2_2b_plt --dry-run      # sizes and revisions; settles Q2
python scripts/download_models.py --set gemma2_2b_plt
python scripts/download_models.py --set gemma2_2b_clt426k --dry-run
python scripts/download_models.py --set gemma2_2b_clt426k
# D10: set dtype (float32 or bfloat16) and memory_cap_gb in configs/models/*.yaml, record in STATUS.md
python scripts/sanity_k0.py --transcoders both   # K0: stop here and report if anything fails
```
After K0 passes and is reported:
```bash
bash scripts/run_kill_test.sh --transcoders both   # exact graphs, K1, K2, K3, report (resumable)
```
Each step can also be run on its own (same flags). Every script supports `--config`, `--seed`,
`--tiny`, `--force` (otherwise finished outputs are skipped) and `--transcoders plt|clt|both`.
Logs go to `logs/`, large artefacts to `artifacts/` (both gitignored).

## Rules that the code enforces
- GPU memory is capped per process (`cct.utils.gpu.cap_memory`) from the model config before loading.
- Every result file carries a run record: config, seed, git hash and dirty flag, package versions,
  GPU name, wall time.
- Thresholds are read only from `configs/prereg_kill_test.yaml`, which a test keeps identical to the
  block in `PREREG_KILL_TEST.md`.
