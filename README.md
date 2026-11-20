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
.venv/Scripts/python -m pytest -q                      # all CPU tests, including the tiny end-to-end runs
bash scripts/run_kill_test.sh --tiny                   # writes results/tiny/KILL_REPORT.md (TINY)
```

## VM (RTX PRO 6000 96 GB, or H100): Gate 0 premise tests, the first GPU job (GATE0_PREMISES.md)
Rent with a CUDA 12.8 (or newer) PyTorch image, at least 150 GB disk and 64 GB or more of system RAM.
```bash
git clone https://github.com/thelast1coder/CPAL_3.git CCT_CPAL && cd CCT_CPAL
export HF_HOME=/path/on/big_disk/hf        # all downloads go here
bash scripts/vm_setup.sh                    # venv, torch cu128, pinned requirements, GPU check, pytest
source .venv/bin/activate
hf auth login                               # fresh read-only token; accept the Gemma-2 licence first
for s in qwen3_1p7b_plt gemma2_2b_plt gemma2_2b_clt426k; do python scripts/download_models.py --set $s --dry-run; done
tmux new -s gate0                           # survives a dropped SSH connection
bash scripts/run_premises.sh                # downloads, Stages A to F (parallel lanes), report; resumable
```
Then read `results/gate0/PREMISE_REPORT.md` and stop: Dawar decides. Rent a host with PCIe 5 and at
least 128 GB of system memory (circuit-tracer moves the whole edge matrix to the GPU at every
influence update). Rehearsal on the laptop (numbers meaningless):
`PY=.venv/Scripts/python bash scripts/run_premises.sh --tiny`.

The v1 kill test (`scripts/run_kill_test.sh`, K0 to K3) runs after Gate 0 on the GO-BIG path.
Every script supports `--config`, `--seed`, `--tiny`, `--force` and `--transcoders plt|clt|both`.
Logs go to `logs/`, large artefacts to `artifacts/` (both gitignored).

## Rules that the code enforces
- GPU memory is capped per process (`cct.utils.gpu.cap_memory`) from the model config before loading.
- Every result file carries a run record: config, seed, git hash and dirty flag, package versions,
  GPU name, wall time.
- Thresholds are read only from `configs/prereg_kill_test.yaml`, which a test keeps identical to the
  block in `PREREG_KILL_TEST.md`.
