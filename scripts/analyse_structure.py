"""K1: structure of the regime-D matrices (PHASE1 Section 6). Offline.
Writes results/<run>/k1.json and, if matplotlib is installed, the K1 plots."""
from __future__ import annotations

import glob
import os

import torch

from cct import scriptutil as su
from cct.killtest import k1_decision, record_matrix, s95, structure_stats
from cct.operators.base import KIND_LOGIT, NodeMeta
from cct.utils.io import atomic_write_json, load_pt
from cct.utils.logging import get_logger, run_record


def plots(set_name: str, curves: list, s95s: list, plot_dir: str, log) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.info("matplotlib not installed: K1 plots skipped")
        return []
    os.makedirs(plot_dir, exist_ok=True)
    out = []
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for c in curves[:8]:
        ax.plot(torch.arange(1, c.numel() + 1) / c.numel(), c, lw=1)
    ax.set_xlabel("fraction of targets (largest first)")
    ax.set_ylabel("cumulative out-edge energy")
    ax.set_title(f"Column energy curves, {set_name}")
    p = os.path.join(plot_dir, f"k1_column_energy_curves_{set_name}.png")
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(torch.cat(s95s).float().numpy(), bins=50)
    ax.set_xlabel("column s95 (targets)")
    ax.set_ylabel("columns")
    ax.set_title(f"Column s95, {set_name}")
    p = os.path.join(plot_dir, f"k1_s95_hist_{set_name}.png")
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)
    return out


def main() -> None:
    args = su.parser("K1 structure analysis").parse_args()
    cfg = su.load_yaml(args.config)
    pre = su.prereg()
    log = get_logger("analyse_structure", os.path.join(su.ROOT, "logs"))
    prompts = su.prompts(args, cfg)
    test_ids = {p["id"] for p in prompts["test"]}
    out = {"sets": {}}
    plot_dir = os.path.join(su.ROOT, "plots", "tiny") if args.tiny else os.path.join(su.ROOT, "plots")
    for set_name in su.sets(args):
        files = sorted(glob.glob(os.path.join(su.artifacts_dir(args, set_name, "exact_D"), "*.pt")))
        per_prompt, curves, s95s = {}, [], []
        for f in files:
            rec = load_pt(f)
            if rec["prompt_id"] not in test_ids:
                continue
            A = record_matrix(rec)
            tm, sm = NodeMeta.from_dict(rec["target_meta"]), NodeMeta.from_dict(rec["source_meta"])
            st = structure_stats(A, tm, sm)
            per_prompt[rec["prompt_id"]] = st
            Af = A[tm.kind != KIND_LOGIT]
            e = Af.pow(2)
            tot = e.sum(0)
            top = torch.argsort(tot, descending=True)[:8]
            for c in top:
                v = torch.sort(e[:, c], descending=True).values
                curves.append(torch.cumsum(v, 0) / v.sum())
            s95s.append(s95(Af)[tot > 0])
        fracs = [v["col_s95_ewm_frac"] for v in per_prompt.values()]
        dec = k1_decision(fracs, pre)
        out["sets"][set_name] = {"per_prompt": per_prompt, "decision": dec,
                                 "median_col_s95_frac": float(torch.tensor(fracs).median()) if fracs else None,
                                 "median_row_s95_frac": float(torch.tensor([v["row_s95_ewm_frac"] for v in per_prompt.values()]).median()) if fracs else None,
                                 "median_hub_energy_share": float(torch.tensor([v["hub_energy_share"] for v in per_prompt.values()]).median()) if fracs else None,
                                 "plots": plots(set_name, curves, s95s, plot_dir, log) if s95s else []}
        log.info(f"K1 {set_name}: {dec}")
    out["run_record"] = run_record({"k1": pre["k1"]}, args.seed)
    atomic_write_json(os.path.join(su.results_dir(args), "k1.json"), out)


if __name__ == "__main__":
    main()
