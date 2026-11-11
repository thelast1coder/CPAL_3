"""Run records: every result file stores config, seed, git hash (and dirty flag), package versions,
GPU name and wall time (CLAUDE.md rules)."""
from __future__ import annotations

import datetime as _dt
import importlib
import logging as _logging
import os
import platform
import subprocess
import sys

import torch

PACKAGES = ["torch", "numpy", "scipy", "circuit_tracer", "transformer_lens", "nnsight", "transformers",
            "huggingface_hub", "yaml"]


def git_info(root: str | None = None) -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, cwd=root,
                                  timeout=10).stdout.strip()
        except Exception:
            return ""
    return {"hash": run("rev-parse", "HEAD") or "unknown", "dirty": bool(run("status", "--porcelain"))}


def package_versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for p in PACKAGES:
        try:
            out[p] = getattr(importlib.import_module(p), "__version__", "installed")
        except Exception:
            out[p] = None
    return out


def gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return f"cpu ({platform.machine()})"


def run_record(config: dict, seed: int | None, wall_seconds: float | None = None, **extra) -> dict:
    return {
        "config": config, "seed": seed, "git": git_info(), "packages": package_versions(),
        "gpu": gpu_name(), "wall_seconds": wall_seconds,
        "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), **extra,
    }


def get_logger(name: str, log_dir: str = "logs") -> _logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    lg = _logging.getLogger(name)
    if not lg.handlers:
        lg.setLevel(_logging.INFO)
        fmt = _logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        fh = _logging.FileHandler(os.path.join(log_dir, f"{name}.log"), encoding="utf-8")
        fh.setFormatter(fmt)
        sh = _logging.StreamHandler()
        sh.setFormatter(fmt)
        lg.addHandler(fh)
        lg.addHandler(sh)
    return lg
