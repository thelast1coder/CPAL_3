"""Gate 0, G0-B step 1: generate the whole-trace prompts ONCE on the VM (PROJECT_V2.md Section 7),
then commit the traces file. Settings: configs/experiments/gate0.yaml, section g0b.

Model: Hugging Face transformers AutoModelForCausalLM + AutoTokenizer (the model of the g0b model
set), float32, chat template with enable_thinking=True, greedy decoding (do_sample=False),
max_new_tokens 512, fixed seed. Downloads go under $HF_HOME.

Selection rule (fixed before running; do not change after seeing any trace):
1. Go through the problems of g0b.problems_file in file order and generate each one.
2. A candidate is eligible if generation ended with an EOS token (the model finished, so the answer
   after </think> is complete) and its number of generated tokens (EOS included) lies in
   g0b.generated_tokens_range (inclusive). Keep eligible candidates in file order; stop generating
   once g0b.n_traces are kept.
3. If all problems are generated and fewer than n_traces are kept, the kept traces stay, and the
   remaining places are filled, in file order, from the already generated candidates that ended with
   EOS and whose generated length lies in g0b.fallback_generated_tokens_range (no regeneration).
4. If still fewer than n_traces, the script writes what it has and says so (Dawar decides).
Correctness of the answer is recorded (answer_found) but is not a selection criterion.

Per trace: id, question, expected answer, full token ids (prompt + generated), n_prompt_tokens,
n_generated, decoded text, answer_found (expected answer string appears after the </think> token),
answer_position (token index of the first token of the LAST occurrence of the answer after </think>,
found by decoding token spans; null if absent), plus generation settings, model revision and package
versions. Also a summary row for every generated candidate.

The file is written atomically and never overwritten without --force.

--tiny: tiny pure-PyTorch model (configs/experiments/kill_test.yaml tiny section, set tiny_plt),
random-token prompts, greedy generation from clean_forward logits for 20 tokens; every candidate is
kept; the "answer position" is the last generated token. Writes results/tiny/gate0/traces.json.
"""
from __future__ import annotations

import os
import sys
import time

import torch

from cct import scriptutil as su
from cct.gate0_union import find_answer_position
from cct.killtest import load_prompts
from cct.utils.io import atomic_write_json
from cct.utils.logging import get_logger, package_versions, run_record

TINY_NEW_TOKENS = 20


def traces_path(args, cfg: dict) -> str:
    if args.tiny:
        return os.path.join(su.ROOT, "results", "tiny", "gate0", "traces.json")
    return os.path.join(su.ROOT, cfg["g0b"]["traces_file"])


def in_range(n: int, r: list[int]) -> bool:
    return r[0] <= n <= r[1]


def select(cands: list[dict], n: int, primary: list[int], fallback: list[int]) -> tuple[list[dict], str]:
    """Selection rule steps 2 to 3 over generated candidates in file order."""
    kept = [c for c in cands if c["ended_with_eos"] and in_range(c["n_generated"], primary)][:n]
    rule = "primary"
    if len(kept) < n:
        rule = "fallback"
        ids = {c["id"] for c in kept}
        for c in cands:
            if len(kept) >= n:
                break
            if c["id"] not in ids and c["ended_with_eos"] and in_range(c["n_generated"], fallback):
                kept.append(c)
        order = {c["id"]: i for i, c in enumerate(cands)}
        kept.sort(key=lambda c: order[c["id"]])
    return kept, rule


# -------------------------------------------------------------------------------------------------
def gen_tiny(args, cfg: dict) -> dict:
    from cct.operators.tiny import random_prompts

    kt = su.load_yaml(os.path.join(su.ROOT, "configs", "experiments", "kill_test.yaml"))
    model = su.tiny_model(kt, "tiny_plt")
    t = kt["tiny"]
    n = cfg["g0b"]["n_traces"]
    prompts = random_prompts(n, t["model"]["vocab"], tuple(t["prompt_lengths"]), seed=cfg["seed"] + 100)
    traces = []
    for i, p in enumerate(prompts):
        ids = p.tolist()
        for _ in range(TINY_NEW_TOKENS):
            logits = model.clean_forward(torch.tensor(ids))["logits"][0, -1]
            ids.append(int(torch.argmax(logits)))
        L = len(ids)
        traces.append({"id": f"tiny_trace_{i:02d}", "category": "tiny", "question": None, "answer": None,
                       "token_ids": ids, "n_prompt_tokens": len(p), "n_generated": TINY_NEW_TOKENS,
                       "n_total_tokens": L, "text": None, "ended_with_eos": True, "answer_found": False,
                       "answer_position": L - 1, "end_position": L - 1, "think_end_index": None})
    return {"traces": traces, "candidates": [{k: tr[k] for k in ("id", "n_generated", "ended_with_eos",
                                                                  "answer_found")} for tr in traces],
            "selection": "tiny: all candidates kept", "generation": {"greedy": True,
                                                                     "max_new_tokens": TINY_NEW_TOKENS}}


# -------------------------------------------------------------------------------------------------
def gen_real(args, cfg: dict, log) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # VM only

    from cct.utils.gpu import cap_memory, nvidia_smi

    g = cfg["g0b"]
    gen = g["generation"]
    mc = su.model_config(g["model_set"])
    rev = (mc.get("revisions") or {}).get("model")
    log.info(nvidia_smi())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        cap_memory(mc["memory_cap_gb"])
    torch.manual_seed(cfg["seed"])
    tok = AutoTokenizer.from_pretrained(mc["model_name"], revision=rev)
    model = AutoModelForCausalLM.from_pretrained(mc["model_name"], revision=rev,
                                                 torch_dtype=su.torch_dtype(gen["dtype"])).to(device)
    model.eval()
    think_end = tok.convert_tokens_to_ids("</think>")
    if think_end is None or think_end == tok.unk_token_id:
        raise RuntimeError("tokenizer has no </think> token")
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    eos_ids.discard(None)
    problems = load_prompts(os.path.join(su.ROOT, g["problems_file"]))
    decode = lambda ids: tok.decode(ids, skip_special_tokens=False)  # noqa: E731

    cands: list[dict] = []
    full: dict[str, dict] = {}
    n_primary = 0
    for pr in problems:
        if n_primary >= g["n_traces"]:
            break
        text = tok.apply_chat_template([{"role": "user", "content": pr["question"]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=gen["enable_thinking"])
        enc = tok(text, return_tensors="pt", add_special_tokens=False)
        ids = enc.input_ids.to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(ids, attention_mask=enc.attention_mask.to(device), do_sample=gen["do_sample"],
                                 max_new_tokens=gen["max_new_tokens"], temperature=None, top_p=None, top_k=None,
                                 pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else min(eos_ids))
        secs = time.perf_counter() - t0
        all_ids = out[0].tolist()
        n_prompt = ids.shape[1]
        gen_ids = all_ids[n_prompt:]
        ended = bool(gen_ids) and gen_ids[-1] in eos_ids
        te = [i for i, x in enumerate(all_ids) if x == think_end and i >= n_prompt]
        te_idx = te[-1] if te else None
        ans_pos = find_answer_position(all_ids, te_idx + 1, pr["answer"], decode) if te_idx is not None else None
        rec = {"id": pr["id"], "category": pr["category"], "question": pr["question"], "answer": pr["answer"],
               "token_ids": all_ids, "n_prompt_tokens": n_prompt, "n_generated": len(gen_ids),
               "n_total_tokens": len(all_ids), "text": decode(all_ids), "ended_with_eos": ended,
               "think_end_index": te_idx, "answer_found": ans_pos is not None, "answer_position": ans_pos,
               "end_position": ans_pos if ans_pos is not None else len(all_ids) - 1,
               "first_token_is_special": all_ids[0] in tok.all_special_ids,
               "generation_seconds": secs}
        full[pr["id"]] = rec
        cands.append({k: rec[k] for k in ("id", "n_generated", "ended_with_eos", "answer_found",
                                          "answer_position", "think_end_index")})
        if ended and in_range(len(gen_ids), g["generated_tokens_range"]):
            n_primary += 1
        log.info(f"{pr['id']}: generated {len(gen_ids)} eos={ended} answer_found={ans_pos is not None} "
                 f"{secs:.1f}s")
    kept, rule = select(cands, g["n_traces"], g["generated_tokens_range"], g["fallback_generated_tokens_range"])
    if len(kept) < g["n_traces"]:
        log.warning(f"only {len(kept)} of {g['n_traces']} traces meet the rule; Dawar decides")
    return {"traces": [full[c["id"]] for c in kept], "candidates": cands, "selection": rule,
            "generation": dict(gen, seed=cfg["seed"], model_name=mc["model_name"],
                               model_revision_config=rev,
                               model_revision_loaded=getattr(model.config, "_commit_hash", None),
                               eos_token_ids=sorted(eos_ids), think_end_token_id=think_end,
                               chat_template="tokenizer.apply_chat_template, add_generation_prompt=True")}


def main() -> None:
    p = su.parser("Gate 0 G0-B: generate the whole traces once")
    p.set_defaults(config=os.path.join(su.ROOT, "configs", "experiments", "gate0.yaml"))
    args = p.parse_args()
    cfg = su.load_yaml(args.config)
    out = traces_path(args, cfg)
    if os.path.exists(out) and not args.force:
        print(f"{out} exists; refusing to overwrite (use --force)", file=sys.stderr)
        sys.exit(1)
    log = get_logger("gen_traces", os.path.join(su.ROOT, "logs"))
    t0 = time.perf_counter()
    res = gen_tiny(args, cfg) if args.tiny else gen_real(args, cfg, log)
    res["selection_rule"] = ("in file order: ended with EOS and n_generated in generated_tokens_range; "
                             "if fewer than n_traces, fill from candidates in fallback range (no regeneration)")
    res["packages"] = package_versions()
    res["run"] = run_record({"g0b": cfg["g0b"], "tiny": args.tiny}, cfg["seed"], time.perf_counter() - t0)
    atomic_write_json(out, res)
    log.info(f"wrote {len(res['traces'])} traces to {out}")
    if not args.tiny:
        rel = os.path.relpath(out, su.ROOT).replace(os.sep, "/")
        print("Commit the traces before any union run:")
        print(f"  git add {rel}")
        print(f'  git commit -m "Gate 0 G0-B: generated traces ({len(res["traces"])} traces, rule '
              f'{res["selection"]})"')


if __name__ == "__main__":
    main()
