"""Gate 0 premise tests: traces, sentences and sampled outputs (GATE0_PREMISES.md Section 2).

Shared by scripts/gen_traces.py, premise_B.py, premise_E.py and premise_F.py. Public API (keep
stable): split_sentences, sampled_outputs, load_traces. The rest are helpers.

Sentence rule
-------------
Thought Anchors (Bogdan, Macar, Nanda, Conmy 2025), function `split_solution_into_chunks` in
utils.py of https://github.com/interp-reasoning/thought-anchors (commit b53ed8c, read on 16 Nov
2026; see THOUGHT_ANCHORS_SOURCE). Their rule, on the thinking text:
1. walk the text character by character; a chunk ends after a character that is ".", "?" or "!"
   and is followed by a space or a newline, or after the first character of a paragraph break
   ("\\n\\n" or "\\r\\n\\r\\n"); each chunk is stripped of surrounding whitespace, and a boundary
   whose chunk is only whitespace does not end a chunk (the whitespace is carried over);
2. text after the last boundary is dropped (their code to keep it is commented out);
3. chunks shorter than 10 characters are merged into the next chunk (the last one into the
   previous chunk), joined by one space, in one left-to-right pass, as in their loop.

We reproduce this exactly on character spans, with two documented differences, both needed to map
the chunks onto the token sequence:
- text after the last boundary is KEPT as a final chunk (added before the merge step, as their
  commented-out code did); for a thinking text that ends in "." plus a newline it is empty anyway;
- whitespace between chunks is given to the following chunk (leading whitespace of the text to the
  first chunk), so the sentences tile the text with no gaps.
Tokens are then mapped to sentences by their first character: a token belongs to the sentence that
contains the first character it decodes to (zero-length tokens go with the next character). Spans
are half-open token ranges [tok_start, tok_end) covering [start, end) without gaps.
"""
from __future__ import annotations

import bisect
import json
import os
import warnings
from typing import Callable

THOUGHT_ANCHORS_SOURCE = {
    "paper": "Thought Anchors: Which LLM Reasoning Steps Matter? (Bogdan, Macar, Nanda, Conmy 2025)",
    "repository": "https://github.com/interp-reasoning/thought-anchors",
    "url": "https://github.com/interp-reasoning/thought-anchors/blob/"
           "b53ed8c75d3f6112f68adfaec9a13d4d708c442e/utils.py",
    "function": "split_solution_into_chunks",
    "commit": "b53ed8c75d3f6112f68adfaec9a13d4d708c442e",
    "checked": "2026-11-16",
    "differences": ["text after the last boundary is kept as a final chunk (Thought Anchors drops it)",
                    "inter-chunk whitespace is assigned to the following chunk so the spans tile the text"],
}
SENTENCE_ENDINGS = (".", "?", "!")
PARAGRAPH_BREAKS = ("\n\n", "\r\n\r\n")
MIN_CHUNK_CHARS = 10
TINY_SENTENCE_TOKENS = 4

# tiny backend used by the premise rehearsals (smaller than the kill-test tiny model, so that 51
# graphs on ~80-token prefixes run in seconds on a laptop; numbers are meaningless)
TINY_PREMISE_MODEL = {"transcoder": "plt", "n_layers": 2, "d_model": 16, "n_heads": 2, "d_head": 8,
                      "vocab": 64, "d_mlp": 32, "n_features": 16, "active_fraction": 0.1, "seed": 0}
TINY_NEW_TOKENS = 72
TINY_PROMPT_LENGTHS = (8, 12)

_TINY_MODEL = None


def tiny_premise_model():
    global _TINY_MODEL
    if _TINY_MODEL is None:
        from .operators.tiny import TinyConfig, TinyModel

        _TINY_MODEL = TinyModel(TinyConfig(**TINY_PREMISE_MODEL))
    return _TINY_MODEL


# -------------------------------------------------------------------------------------------------
# Thought Anchors chunking on character spans
# -------------------------------------------------------------------------------------------------
def _strip_span(text: str, a: int, b: int) -> tuple[int, int]:
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return a, b


def ta_chunk_spans(text: str, keep_trailing: bool = True) -> list[tuple[int, int]]:
    """Character spans [a, b) of the Thought Anchors chunks of `text` (stripped, after merging). A
    merged chunk spans from the start of its first part to the end of its last part. With
    keep_trailing=False the result matches split_solution_into_chunks exactly (chunk strings equal
    text[a:b] up to the joins of merged chunks, whose original separator is replaced by one space
    in their output)."""
    raw: list[list[int]] = []  # [a, b, length of their chunk string]
    cur_start = 0
    n = len(text)
    for i in range(n):
        para = any(text[i:i + len(p)] == p for p in PARAGRAPH_BREAKS)
        sent = i < n - 1 and text[i] in SENTENCE_ENDINGS and text[i + 1] in (" ", "\n")
        if para or sent:
            a, b = _strip_span(text, cur_start, i + 1)
            if b > a:
                raw.append([a, b, b - a])
                cur_start = i + 1
    if keep_trailing:
        a, b = _strip_span(text, cur_start, n)
        if b > a:
            raw.append([a, b, b - a])
    # merge pass, exactly as their loop (lengths of the joined strings: a + 1 + b)
    ch = raw
    i = 0
    while i < len(ch):
        if ch[i][2] < MIN_CHUNK_CHARS:
            if i == len(ch) - 1:
                if i > 0:
                    ch[i - 1] = [ch[i - 1][0], ch[i][1], ch[i - 1][2] + 1 + ch[i][2]]
                    ch.pop(i)
            else:
                ch[i + 1] = [ch[i][0], ch[i + 1][1], ch[i][2] + 1 + ch[i + 1][2]]
                ch.pop(i)
            if i == 0 and len(ch) == 1:
                break
        else:
            i += 1
    return [(a, b) for a, b, _ in ch]


def _token_char_offsets(token_ids: list[int], decode: Callable[[list[int]], str]) -> tuple[list[int], str]:
    """First-character offset of each token in decode(token_ids), from decoding growing prefixes
    (made non-decreasing and clamped, because a prefix ending inside a multi-byte character can
    decode to a replacement character)."""
    text = decode(list(token_ids))
    offs, hi = [], 0
    for k in range(len(token_ids)):
        o = min(len(decode(list(token_ids[:k]))), len(text))
        hi = max(hi, o)
        offs.append(hi)
    return offs, text


def split_sentences(token_ids, tokenizer_decode_fn: Callable[[list[int]], str], start: int,
                    end: int) -> list[tuple[int, int]]:
    """Half-open token spans covering [start, end) of token_ids, one per Thought Anchors sentence of
    the decoded text of tokens[start:end] (module docstring). Sentences that receive no token (their
    first character lies inside a token that starts in the previous sentence) are absorbed."""
    if end <= start:
        return []
    seg = [int(x) for x in token_ids[start:end]]
    offs, text = _token_char_offsets(seg, tokenizer_decode_fn)
    chunks = ta_chunk_spans(text)
    # tiling: sentence i covers [end of chunk i - 1, end of chunk i), so the whitespace before a chunk
    # (and a token such as " Then" whose first character is that space) goes to the following chunk
    starts = [0] + [b for _, b in chunks[:-1]]
    sid = []
    for o in offs:
        j = bisect.bisect_right(starts, o) - 1
        sid.append(min(max(j, 0), max(len(starts) - 1, 0)))
    spans: list[tuple[int, int]] = []
    k0 = 0
    for k in range(1, len(seg) + 1):
        if k == len(seg) or sid[k] != sid[k0]:
            spans.append((start + k0, start + k))
            k0 = k
    return spans


def tiny_sentences(start: int, end: int, size: int = TINY_SENTENCE_TOKENS) -> list[tuple[int, int]]:
    """Tiny traces have no text: fixed `size`-token sentences over [start, end), a short remainder
    joined to the last sentence."""
    spans = [(s, min(s + size, end)) for s in range(start, end, size)]
    if len(spans) > 1 and spans[-1][1] - spans[-1][0] < size:
        a, _ = spans.pop()
        spans[-1] = (spans[-1][0], end)
    return spans


# -------------------------------------------------------------------------------------------------
# traces and sampled outputs
# -------------------------------------------------------------------------------------------------
def n_total(trace: dict) -> int:
    return int(trace.get("n_total_tokens") or len(trace["token_ids"]))


def thinking_span(trace: dict) -> tuple[int, int]:
    """[start, end) of the thinking text: after the <think> token (or the prompt) up to the </think>
    token (or the answer position, or the end of the trace)."""
    s = trace.get("think_start_index")
    s = int(s) if s is not None else int(trace["n_prompt_tokens"])
    e = trace.get("think_end_index")
    if e is None:
        e = trace.get("answer_position")
        e = int(e) if e is not None else n_total(trace) - 1
    return s, int(e)


def trace_sentences(trace: dict) -> list[tuple[int, int]]:
    """Stored sentences (gen_traces writes them), else the tiny fallback."""
    if trace.get("sentences"):
        return [(int(a), int(b)) for a, b in trace["sentences"]]
    s, e = thinking_span(trace)
    return tiny_sentences(s, e)


def answer_position(trace: dict) -> int:
    a = trace.get("answer_position")
    return int(a) if a is not None else n_total(trace) - 1


def sampled_outputs(trace: dict, sentences, n_sentences: int = 8, tokens_per_sentence: int = 2) -> list[dict]:
    """The sampled outputs of one trace (GATE0_PREMISES.md Section 2): the final answer token first,
    then the first `tokens_per_sentence` tokens of `n_sentences` sentences spread evenly (first and
    last included) over the sentences that start in the second half of the generated part, in
    sentence order. position q = index of the token whose PREDICTION is explained (graph on
    tokens[:q]). Only sentences with at least tokens_per_sentence tokens qualify. If the second half
    holds fewer than n_sentences such sentences, the latest qualifying sentences before it are
    added (with a warning). Answer: trace["answer_position"], else the last generated token."""
    sentences = [(int(a), int(b)) for a, b in sentences]
    n_prompt = int(trace["n_prompt_tokens"])
    n_gen = int(trace.get("n_generated") or (n_total(trace) - n_prompt))
    mid = n_prompt + n_gen // 2
    ans = answer_position(trace)
    ok = [i for i, (a, b) in enumerate(sentences) if b - a >= tokens_per_sentence and a + tokens_per_sentence <= ans]
    elig = [i for i in ok if sentences[i][0] >= mid]
    if len(elig) < n_sentences:
        before = [i for i in ok if sentences[i][0] < mid]
        need = n_sentences - len(elig)
        if len(before) < need:
            raise ValueError(f"trace {trace.get('id')}: only {len(ok)} sentences with >= {tokens_per_sentence} "
                             f"tokens; need {n_sentences}")
        warnings.warn(f"trace {trace.get('id')}: {len(elig)} sentences in the second half; adding the last "
                      f"{need} before it")
        elig = before[-need:] + elig
    M = len(elig)
    if n_sentences == 1:
        pick = [elig[-1]]
    else:
        pick = [elig[(j * (M - 1) + (n_sentences - 1) // 2) // (n_sentences - 1)] for j in range(n_sentences)]
    out = [{"position": ans, "sentence_index": None, "kind": "answer"}]
    for i in pick:
        for t in range(tokens_per_sentence):
            out.append({"position": sentences[i][0] + t, "sentence_index": i, "kind": "sentence_token"})
    pos = [o["position"] for o in out]
    if len(set(pos)) != len(pos):
        raise ValueError(f"trace {trace.get('id')}: duplicate sampled positions {pos}")
    return out


def trace_outputs(trace: dict, n_sentences: int = 8, tokens_per_sentence: int = 2) -> list[dict]:
    """sampled_outputs on the trace's stored (or tiny) sentences, each tagged with the trace id."""
    outs = trace.get("sampled_outputs")
    if not outs:
        outs = sampled_outputs(trace, trace_sentences(trace), n_sentences, tokens_per_sentence)
    return [dict(o, trace_id=trace["id"]) for o in outs]


def load_traces(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d["traces"] if isinstance(d, dict) else d


# -------------------------------------------------------------------------------------------------
# paths per model set
# -------------------------------------------------------------------------------------------------
def model_tag(model_set: str, default_set: str) -> str:
    """"" for the default set (Qwen3-1.7B), else a short tag ("qwen3_4b" for qwen3_4b_plt)."""
    if model_set == default_set:
        return ""
    return model_set[:-4] if model_set.endswith("_plt") else model_set


def gate0_results_dir(root: str, tiny: bool, tag: str = "") -> str:
    d = os.path.join(root, "results", "tiny", "gate0") if tiny else os.path.join(root, "results", "gate0")
    return os.path.join(d, tag) if tag else d


def traces_file(root: str, tiny: bool, tag: str = "", cfg: dict | None = None) -> str:
    if not tiny and not tag and cfg is not None:
        return os.path.join(root, cfg["traces"]["traces_file"])
    return os.path.join(gate0_results_dir(root, tiny, tag), "traces.json")
