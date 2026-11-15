"""Download a model and its transcoders into $HF_HOME (VM only) and record revisions and sizes in the
model config. No token appears here: authenticate beforehand with `hf auth login` (verify the command
for the installed huggingface_hub).

Usage: python scripts/download_models.py --set gemma2_2b_plt [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sys

from cct import scriptutil as su


MODEL_PATTERNS = ["*.json", "*.safetensors", "tokenizer*", "*.model"]  # weights, config, tokenizer only
TRANSCODER_IGNORE = ["features/*"]  # feature-visualisation data (several GB), not needed for attribution


def repo_size_gb(api, repo_id: str, revision: str | None, patterns=None, ignore=None) -> float:
    import fnmatch

    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    tot = 0
    for s in info.siblings:
        if ignore and any(fnmatch.fnmatch(s.rfilename, p) for p in ignore):
            continue
        if patterns is None or any(fnmatch.fnmatch(s.rfilename, p) for p in patterns):
            tot += s.size or 0
    return tot / 1024 ** 3


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--set", required=True)
    p.add_argument("--dry-run", action="store_true", help="only report sizes and revisions")
    p.add_argument("--no-warm", action="store_true", help="do not load the transcoders once after download")
    args = p.parse_args()
    if not os.environ.get("HF_HOME"):
        sys.exit("HF_HOME is not set: all downloads must go under $HF_HOME on the big disk")
    from huggingface_hub import HfApi, snapshot_download

    mc = su.model_config(args.set)
    api = HfApi()
    m_info = api.model_info(mc["model_name"])
    t_repo = mc["transcoder_set"]
    t_info = api.model_info(t_repo)
    sizes = {"model_gb": repo_size_gb(api, mc["model_name"], m_info.sha, MODEL_PATTERNS),
             "transcoders_gb": repo_size_gb(api, t_repo, t_info.sha, ignore=TRANSCODER_IGNORE)}
    print(f"{mc['model_name']}@{m_info.sha}: {sizes['model_gb']:.2f} GB; {t_repo}@{t_info.sha}: {sizes['transcoders_gb']:.2f} GB")
    st = os.statvfs(os.environ["HF_HOME"])
    free_gb = st.f_bavail * st.f_frsize / 1024 ** 3
    print(f"free under HF_HOME: {free_gb:.1f} GB")
    if args.dry_run:
        return
    if sizes["model_gb"] + sizes["transcoders_gb"] > 0.8 * free_gb:
        sys.exit("not enough free disk under HF_HOME (keep 20% headroom)")
    snapshot_download(mc["model_name"], revision=m_info.sha, allow_patterns=MODEL_PATTERNS)
    snapshot_download(t_repo, revision=t_info.sha, ignore_patterns=TRANSCODER_IGNORE)
    # Transcoder sets whose config points at other repositories (hf://...) are fetched by circuit-tracer
    # at first load, also under $HF_HOME; load_once warms that cache and records the time.
    if mc.get("warm_load") and not args.no_warm:
        import time

        t0 = time.perf_counter()
        from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

        import torch

        load_transcoder_from_hub(t_repo, device=torch.device("cpu"))
        print(f"transcoders loaded once on CPU in {time.perf_counter() - t0:.0f} s (cache warmed)")
    mc["revisions"] = {"model": m_info.sha, "transcoders": t_info.sha}
    mc["sizes_gb"] = sizes
    su.save_yaml(os.path.join(su.ROOT, "configs", "models", f"{args.set}.yaml"), mc)
    print("recorded revisions in", f"configs/models/{args.set}.yaml")


if __name__ == "__main__":
    main()
