"""Atomic writes, resumable JSONL, and matrix storage.

Matrices are stored as fp16 with a per-row fp32 scale (row / max|row|), plus fp32 checksums of the
original. Plain fp16 would flush small edge weights to zero, which would make matrices look sparser
than they are (a bias in favour of H-K1); the per-row scale keeps relative precision of about 1e-3
for every entry within about 1e-7 of its row maximum (docs/decisions.md).
"""
from __future__ import annotations

import json
import os
import tempfile

import torch
from torch import Tensor


def atomic_write_text(path: str, text: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _default(o):
    if isinstance(o, Tensor):
        return o.tolist()
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)


def atomic_write_json(path: str, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, default=_default))


def read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=_default) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:  # torn last line after a crash
                    pass
    return out


def pack_matrix(A: Tensor, fp32: bool = False) -> dict:
    A32 = A.detach().float().cpu()
    d = {"shape": list(A32.shape), "checksum_sum": float(A32.double().sum()),
         "checksum_abs": float(A32.double().abs().sum())}
    if fp32:
        d["fp32"] = A32
        return d
    scale = A32.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    d["fp16"] = (A32 / scale).half()
    d["row_scale"] = scale.squeeze(1)
    return d


def unpack_matrix(d: dict, check: bool = True, rtol: float = 1e-2) -> Tensor:
    if "fp32" in d:
        A = d["fp32"].float()
    else:
        A = d["fp16"].float() * d["row_scale"][:, None]
    if check:
        s = float(A.double().abs().sum())
        if abs(s - d["checksum_abs"]) > rtol * max(1e-30, abs(d["checksum_abs"])):
            raise ValueError(f"matrix checksum mismatch: {s} vs {d['checksum_abs']}")
    return A


def save_pt(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_pt(path: str):
    return torch.load(path, map_location="cpu", weights_only=False)
