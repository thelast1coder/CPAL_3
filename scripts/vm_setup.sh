#!/usr/bin/env bash
# VM bootstrap (run once on the GPU VM, from the repository root).
# Requires: HF_HOME set to a directory on the big disk. No tokens are read or written by this script;
# log in separately with `hf auth login` (huggingface_hub >= 0.34; older: `huggingface-cli login`) (verify).
set -euo pipefail
cd "$(dirname "$0")/.."

: "${HF_HOME:?set HF_HOME to a directory on the big disk first}"
echo "HF_HOME=$HF_HOME"
df -h "$HF_HOME"
nvidia-smi || { echo "no GPU visible"; exit 1; }

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CUDA build of torch first. Blackwell GPUs (RTX PRO 6000, RTX 5090, B200; compute capability 12.x or
# 10.x) need a CUDA 12.8 build (torch >= 2.7); Hopper/Ada also run it with a recent driver. Override
# with TORCH_CUDA=cu124 if the driver is older (nvidia-smi shows the maximum CUDA version).
TORCH_CUDA="${TORCH_CUDA:-cu128}"
python -m pip install "torch>=2.7" --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
mkdir -p results
python -m pip freeze > results/vm_pip_freeze.txt
python - <<'PY'
import torch, circuit_tracer, transformer_lens
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0),
      "capability", torch.cuda.get_device_capability(0))
x = torch.randn(256, 256, device="cuda"); assert torch.isfinite(x @ x).all()  # kernels run on this GPU
print("circuit_tracer", getattr(circuit_tracer, "__version__", "?"), "transformer_lens", transformer_lens.__version__)
PY
python -m pytest -q -m "not slow"   # fast tests only: the slow tiny runs were checked on the laptop
echo "Setup done. Next: hf auth login (fresh token), then python scripts/download_models.py --set gemma2_2b_plt"
