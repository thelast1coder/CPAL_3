"""Helpers shared by scripts/: arguments, paths, prompt sets and operators for tiny and real runs."""
from __future__ import annotations

import argparse
import os

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SETS_REAL = {"plt": "gemma2_2b_plt", "clt": "gemma2_2b_clt426k"}
SETS_TINY = {"plt": "tiny_plt", "clt": "tiny_clt"}


def parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "experiments", "kill_test.yaml"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tiny", action="store_true", help="tiny pure-PyTorch backend; numbers are meaningless")
    p.add_argument("--force", action="store_true", help="recompute finished outputs")
    p.add_argument("--transcoders", default="both", choices=["plt", "clt", "both"])
    return p


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def prereg() -> dict:
    return load_yaml(os.path.join(ROOT, "configs", "prereg_kill_test.yaml"))


def sets(args) -> list[str]:
    s = ["plt", "clt"] if args.transcoders == "both" else [args.transcoders]
    return [(SETS_TINY if args.tiny else SETS_REAL)[x] for x in s]


def run_name(args) -> str:
    return "tiny" if args.tiny else "real"


def artifacts_dir(args, set_name: str, *parts: str) -> str:
    d = os.path.join(ROOT, "artifacts", run_name(args), set_name, *parts)
    os.makedirs(d, exist_ok=True)
    return d


def results_dir(args) -> str:
    d = os.path.join(ROOT, "results", "tiny") if args.tiny else os.path.join(ROOT, "results")
    os.makedirs(d, exist_ok=True)
    return d


def model_config(set_name: str) -> dict:
    """Model config; CCT_MEMORY_CAP_GB in the environment overrides memory_cap_gb (used by the job queue
    to split one GPU between lanes)."""
    mc = load_yaml(os.path.join(ROOT, "configs", "models", f"{set_name}.yaml"))
    if os.environ.get("CCT_MEMORY_CAP_GB"):
        mc["memory_cap_gb"] = float(os.environ["CCT_MEMORY_CAP_GB"])
    return mc


# -- tiny ------------------------------------------------------------------------------------------
_TINY_CACHE: dict = {}


def tiny_model(cfg: dict, set_name: str):
    from .operators.tiny import TinyConfig, TinyModel

    if set_name not in _TINY_CACHE:
        tc = "plt" if set_name.endswith("plt") else "clt"
        _TINY_CACHE[set_name] = TinyModel(TinyConfig(transcoder=tc, seed=0, **cfg["tiny"]["model"]))
    return _TINY_CACHE[set_name]


def tiny_prompts(cfg: dict) -> dict[str, list[dict]]:
    """Random-token prompts with category labels (4 categories), deterministic."""
    from .operators.tiny import random_prompts

    t = cfg["tiny"]
    V = t["model"]["vocab"]
    out = {}
    for name, n, lens, seed in [("test", t["n_test"], t["prompt_lengths"], 1),
                                ("validation", t["n_validation"], t["prompt_lengths"], 2),
                                ("long", t["n_long"], t["long_lengths"], 3)]:
        toks = random_prompts(n, V, tuple(lens), seed)
        out[name] = [{"id": f"tiny_{name}_{i:02d}", "category": f"cat{i % 4}", "tokens": x.tolist()}
                     for i, x in enumerate(toks)]
    return out


def real_prompts(cfg: dict) -> dict[str, list[dict]]:
    from .killtest import load_prompts

    return {k: load_prompts(os.path.join(ROOT, cfg["prompts"][k])) for k in ("test", "validation", "long")}


def prompts(args, cfg: dict) -> dict[str, list[dict]]:
    return tiny_prompts(cfg) if args.tiny else real_prompts(cfg)


def timing(args, cfg: dict, set_name: str) -> tuple[float, float]:
    """(row_seconds, kappa) measured at K0.5."""
    from .utils.io import read_json

    path = os.path.join(results_dir(args), "k0.json")
    if os.path.exists(path):
        k0 = read_json(path)
        t = k0.get("sets", {}).get(set_name, {}).get("timing", {})
        if t.get("row_seconds") and t.get("kappa"):
            return float(t["row_seconds"]), float(t["kappa"])
    if not args.tiny:
        mc = model_config(set_name)
        if mc["timing"]["row_seconds"] and mc["timing"]["kappa"]:
            return float(mc["timing"]["row_seconds"]), float(mc["timing"]["kappa"])
        raise RuntimeError(f"no K0.5 timing for {set_name}: run scripts/sanity_k0.py first")
    raise RuntimeError("no tiny timing: run scripts/sanity_k0.py --tiny first")


def torch_dtype(name: str | None) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float64": torch.float64,
            None: torch.float32}[name]
