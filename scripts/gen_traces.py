"""Gate 0 premise tests: generate the thinking traces ONCE on the VM (GATE0_PREMISES.md Section 2),
then commit the traces file. Settings: configs/experiments/gate0_premises.yaml, section traces.

Model: Hugging Face transformers AutoModelForCausalLM + AutoTokenizer (the model of the model set:
default sets.qwen, or --model-set qwen3_4b_plt for the SWITCH-MODEL path), dtype from
traces.generation, chat template with enable_thinking=True, greedy decoding (do_sample=False),
max_new_tokens 512, fixed seed. Downloads go under $HF_HOME.

Selection rule (fixed before running; do not change after seeing any trace): ONE trace per category
of traces.categories (arithmetic, multi_hop, logic), in that order.
1. Per category, generate that category's problems of traces.problems_file in file order, and stop
   at the first one that ended with an EOS token (the model finished, so the answer after </think>
   is complete) and whose number of generated tokens (EOS included) lies in
   traces.generated_tokens_range (inclusive). That problem is the category's trace.
2. If no problem of the category qualifies, the first generated candidate of that category (file
   order) that ended with EOS and whose length lies in traces.fallback_generated_tokens_range is
   used (no regeneration).
3. If still none, the category has no trace; the script writes what it has, says so, and exits with
   code 3 (Dawar decides).
Correctness of the answer is recorded (answer_found) but is not a selection criterion.

Per trace: id, category, question, expected answer, full token ids (prompt + generated),
n_prompt_tokens, n_generated, decoded text, think_start_index (first token after <think>),
think_end_index (the </think> token), answer_found, answer_position (first token of the LAST
occurrence of the answer after </think>; null if absent), sentences (Thought Anchors rule on the
thinking text, cct.premise_traces.split_sentences, half-open token spans), sentence_rule (source),
sampled_outputs (cct.premise_traces.sampled_outputs: 17 outputs). Also a summary row for every
generated candidate, the generation settings, the model revision and package versions.

Output: results/gate0/traces.json (results/gate0/qwen3_4b/traces.json for Qwen3-4B), written
atomically and never overwritten without --force.

--tiny: tiny pure-PyTorch model (cct.premise_traces.TINY_PREMISE_MODEL), one random-token prompt per
category, greedy generation from clean_forward logits for 72 tokens; fixed 4-token "sentences"; the
"answer position" is the last generated token. Writes results/tiny/gate0[/<tag>]/traces.json.
"""
from __future__ import annotations

import os
import sys
import time

import torch

from cct import scriptutil as su
from cct.gate0_union import find_answer_position
from cct.killtest import load_prompts
from cct.premise_traces import (THOUGHT_ANCHORS_SOURCE, TINY_NEW_TOKENS, TINY_PREMISE_MODEL, TINY_PROMPT_LENGTHS,
                                model_tag, sampled_outputs, split_sentences, tiny_premise_model, tiny_sentences,
                                traces_file)
from cct.utils.io import atomic_write_json
from cct.utils.logging import get_logger, package_versions, run_record

DEFAULT_CONFIG = os.path.join(su.ROOT, "configs", "experiments", "gate0_premises.yaml")


def in_range(n: int, r: list[int]) -> bool:
    return r[0] <= n <= r[1]


def eligible(c: dict, r: list[int]) -> bool:
    return bool(c["ended_with_eos"]) and in_range(int(c["n_generated"]), r)


def select_category(cands: list[dict], primary: list[int], fallback: list[int]) -> tuple[dict | None, str]:
    """Selection rule steps 1 to 2 for one category; cands in file order."""
    for c in cands:
        if eligible(c, primary):
            return c, "primary"
    for c in cands:
        if eligible(c, fallback):
            return c, "fallback"
    return None, "none"


def add_sentences(rec: dict, sentences: list[tuple[int, int]], tc: dict) -> None:
    rec["sentences"] = [list(s) for s in sentences]
    rec["n_sentences"] = len(sentences)
    try:
        rec["sampled_outputs"] = sampled_outputs(rec, sentences, tc["sentences_per_trace"], tc["tokens_per_sentence"])
    except ValueError as e:  # recorded; premise_B refuses such a trace
        rec["sampled_outputs"] = None
        rec["sampled_outputs_error"] = str(e)


# -------------------------------------------------------------------------------------------------
def gen_tiny(args, cfg: dict, log) -> dict:
    from cct.operators.tiny import random_prompts

    tc = cfg["traces"]
    model = tiny_premise_model()
    cats = tc["categories"]
    prompts = random_prompts(len(cats), TINY_PREMISE_MODEL["vocab"], TINY_PROMPT_LENGTHS, seed=cfg["seed"] + 100)
    traces = []
    for i, (cat, p) in enumerate(zip(cats, prompts)):
        ids = p.tolist()
        for _ in range(TINY_NEW_TOKENS):
            logits = model.clean_forward(torch.tensor(ids))["logits"][0, -1]
            ids.append(int(torch.argmax(logits)))
        L = len(ids)
        rec = {"id": f"tiny_trace_{i:02d}", "category": cat, "question": None, "answer": None, "token_ids": ids,
               "n_prompt_tokens": len(p), "n_generated": TINY_NEW_TOKENS, "n_total_tokens": L, "text": None,
               "ended_with_eos": True, "answer_found": False, "answer_position": L - 1, "end_position": L - 1,
               "think_start_index": len(p), "think_end_index": None}
        add_sentences(rec, tiny_sentences(len(p), L - 1), tc)
        rec["sentence_rule"] = "tiny: fixed 4-token sentences over the generated part before the last token"
        traces.append(rec)
    return {"traces": traces,
            "candidates": [{k: t[k] for k in ("id", "category", "n_generated", "ended_with_eos", "answer_found")}
                           for t in traces],
            "selection": {c: "tiny" for c in cats},
            "generation": {"greedy": True, "max_new_tokens": TINY_NEW_TOKENS, "model": TINY_PREMISE_MODEL}}


# -------------------------------------------------------------------------------------------------
def gen_real(args, cfg: dict, model_set: str, log) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # VM only

    from cct.utils.gpu import cap_memory, nvidia_smi

    tc = cfg["traces"]
    gen = tc["generation"]
    mc = su.model_config(model_set)
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
    think_start = tok.convert_tokens_to_ids("<think>")
    think_end = tok.convert_tokens_to_ids("</think>")
    for name, t in (("<think>", think_start), ("</think>", think_end)):
        if t is None or t == tok.unk_token_id:
            raise RuntimeError(f"tokenizer has no {name} token")
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    eos_ids.discard(None)
    problems = load_prompts(os.path.join(su.ROOT, tc["problems_file"]))
    decode = lambda ids: tok.decode(ids, skip_special_tokens=False)  # noqa: E731

    cands: list[dict] = []
    full: dict[str, dict] = {}
    chosen: dict[str, dict] = {}
    rules: dict[str, str] = {}
    for cat in tc["categories"]:
        cat_cands = []
        for pr in [p for p in problems if p["category"] == cat]:
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
            ts = [i for i, x in enumerate(all_ids) if x == think_start and i >= n_prompt]
            te = [i for i, x in enumerate(all_ids) if x == think_end and i >= n_prompt]
            ts_idx = ts[0] + 1 if ts else n_prompt
            te_idx = te[-1] if te else None
            ans_pos = find_answer_position(all_ids, te_idx + 1, pr["answer"], decode) if te_idx is not None else None
            rec = {"id": pr["id"], "category": pr["category"], "question": pr["question"], "answer": pr["answer"],
                   "token_ids": all_ids, "n_prompt_tokens": n_prompt, "n_generated": len(gen_ids),
                   "n_total_tokens": len(all_ids), "text": decode(all_ids), "ended_with_eos": ended,
                   "think_start_index": ts_idx, "think_end_index": te_idx, "answer_found": ans_pos is not None,
                   "answer_position": ans_pos, "end_position": ans_pos if ans_pos is not None else len(all_ids) - 1,
                   "first_token_is_special": all_ids[0] in tok.all_special_ids, "generation_seconds": secs}
            s_end = te_idx if te_idx is not None else (ans_pos if ans_pos is not None else len(all_ids) - 1)
            add_sentences(rec, split_sentences(all_ids, decode, ts_idx, s_end), tc)
            rec["sentence_rule"] = THOUGHT_ANCHORS_SOURCE
            full[pr["id"]] = rec
            row = {k: rec[k] for k in ("id", "category", "n_generated", "ended_with_eos", "answer_found",
                                       "answer_position", "think_end_index", "n_sentences")}
            cands.append(row)
            cat_cands.append(row)
            log.info(f"{pr['id']} ({cat}): generated {len(gen_ids)} eos={ended} answer_found={ans_pos is not None} "
                     f"sentences={rec['n_sentences']} {secs:.1f}s")
            if eligible(row, tc["generated_tokens_range"]):
                break
        c, rule = select_category(cat_cands, tc["generated_tokens_range"], tc["fallback_generated_tokens_range"])
        rules[cat] = rule
        if c is not None:
            chosen[cat] = full[c["id"]]
        else:
            log.warning(f"category {cat}: no candidate meets the rule; Dawar decides")
    return {"traces": [chosen[c] for c in tc["categories"] if c in chosen], "candidates": cands, "selection": rules,
            "generation": dict(gen, seed=cfg["seed"], model_set=model_set, model_name=mc["model_name"],
                               model_revision_config=rev,
                               model_revision_loaded=getattr(model.config, "_commit_hash", None),
                               eos_token_ids=sorted(eos_ids), think_start_token_id=think_start,
                               think_end_token_id=think_end,
                               chat_template="tokenizer.apply_chat_template, add_generation_prompt=True")}


def main() -> None:
    p = su.parser("Gate 0 premise tests: generate the thinking traces once")
    p.set_defaults(config=DEFAULT_CONFIG)
    p.add_argument("--model-set", default=None, help="default: sets.qwen of the config")
    args = p.parse_args()
    cfg = su.load_yaml(args.config)
    default_set = cfg["sets"]["qwen"]
    model_set = args.model_set or default_set
    tag = model_tag(model_set, default_set)
    out = traces_file(su.ROOT, args.tiny, tag, cfg)
    if os.path.exists(out) and not args.force:
        print(f"{out} exists; refusing to overwrite (use --force)", file=sys.stderr)
        sys.exit(1)
    log = get_logger("gen_traces", os.path.join(su.ROOT, "logs"))
    t0 = time.perf_counter()
    res = gen_tiny(args, cfg, log) if args.tiny else gen_real(args, cfg, model_set, log)
    res["selection_rule"] = ("one trace per category in traces.categories: the first problem of the category in "
                             "file order that ended with EOS and has n_generated in generated_tokens_range; else "
                             "the first generated candidate of the category in fallback_generated_tokens_range")
    res["sentence_rule"] = THOUGHT_ANCHORS_SOURCE
    res["packages"] = package_versions()
    res["run"] = run_record({"traces": cfg["traces"], "model_set": model_set, "tiny": args.tiny}, cfg["seed"],
                            time.perf_counter() - t0)
    atomic_write_json(out, res)
    n_cat = len(cfg["traces"]["categories"])
    log.info(f"wrote {len(res['traces'])} traces to {out}")
    if not args.tiny:
        rel = os.path.relpath(out, su.ROOT).replace(os.sep, "/")
        print("Commit the traces before any analysis:")
        print(f"  git add {rel}")
        print(f'  git commit -m "Gate 0: generated traces ({len(res["traces"])} traces, {model_set})"')
    if len(res["traces"]) < n_cat:
        sys.exit(3)


if __name__ == "__main__":
    main()
