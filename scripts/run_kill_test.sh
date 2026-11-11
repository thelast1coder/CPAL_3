#!/usr/bin/env bash
# Kill test, end to end: K0 -> exact graphs -> K1 -> K2 -> K3 -> report.
# Usage: bash scripts/run_kill_test.sh [--tiny] [--transcoders plt|clt|both] [--force]
# On the VM: stops after K0 if any check fails (the script exits non-zero).
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x .venv/bin/python ]; then PY=.venv/bin/python
  elif [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
  else PY=python; fi
fi

ARGS=("$@")
echo "== K0 sanity"
"$PY" scripts/sanity_k0.py "${ARGS[@]}"
echo "== exact graphs (regimes D and F)"
"$PY" scripts/run_exact_graphs.py "${ARGS[@]}"
echo "== K1 structure"
"$PY" scripts/analyse_structure.py "${ARGS[@]}"
echo "== K2 simulation"
"$PY" scripts/simulate_sensing.py "${ARGS[@]}"
echo "== K3 real mixed passes"
"$PY" scripts/run_real_cct.py "${ARGS[@]}"
echo "== report"
"$PY" scripts/make_kill_report.py "${ARGS[@]}"
