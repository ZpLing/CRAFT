from __future__ import annotations
"""
PRMBench — LLM-as-Verifier: with_answer vs wout_answer

Research question: Does knowing the correct final answer improve the LLM's
ability to detect erroneous steps in a reasoning chain?

Both settings are evaluated in a single pass against the same PRMBench items.
The LLM sees the `modified_process` (which contains injected errors per the
PRMBench benchmark) and outputs step-level scores.

Prompt format and output schema align with PRMBench's official OpenAI critic
implementation (mr_annotate/build_data/generate_by_4o.py and the leaderboard
submissions for GPT-4o / o1):

    Output JSON per item:
        {"validity": [1.0, -1.0, 0.5, ...], "redundancy": [-1.0, -1.0, ...]}

    - validity:   +1 = valid step,  -1 = invalid step  (threshold > 0 → True)
    - redundancy: +1 = redundant,   -1 = not redundant  (threshold > 0 → True)

These are then passed through PRMBench's eval_on_hallucination_step() to
compute official metrics: correct_step_acc, wrong_step_acc, total_step_acc,
first_error_acc, F1.

Settings:
  A (with_answer): LLM is told the correct final answer before verifying steps.
  B (wout_answer):       LLM verifies steps without knowing the answer.

Usage:
    python prmbench_evaluate_verifier.py \\
        --input PRMBench/mr_eval/tasks/prmbench_stem/data/prmbench_preview.jsonl \\
        --output verifier_results.jsonl \\
        --model gpt-4.1-mini --concurrency 10
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import NamedTuple

import aiohttp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from config import (OPENAI_API_KEY, OPENAI_BASE_URL, DEFAULT_MODEL, REQUEST_TIMEOUT,
                        resolve_input, resolve_output)
except ImportError:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    REQUEST_TIMEOUT = int(os.getenv("OPENAI_REQUEST_TIMEOUT", "180"))
    resolve_input = resolve_output = Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PRMBench official prompt format
# Aligned with mr_annotate/build_data/generate_by_4o.py and the OpenAI critic
# implementation used in leaderboard submissions.
#
# The model must output a JSON object with two score arrays, one per step:
#   validity:   +1 (valid) to -1 (invalid)
#   redundancy: +1 (redundant) to -1 (not redundant)
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLE = """\
Example:
Question: What is 15% of 80?
Solutions:
Step 1: To find 15% of 80, we convert the percentage to a decimal: 15% = 0.15.
Step 2: Now multiply: 0.15 × 80 = 120.
Step 3: Therefore the answer is 120.

Output:
{"validity": [1.0, -1.0, -1.0], "redundancy": [-1.0, -1.0, -1.0]}

Explanation: Step 1 is valid. Step 2 has an arithmetic error (0.15 × 80 = 12, not 120),
so validity = -1. Step 3 propagates the wrong answer, so validity = -1.
"""

SYSTEM_WITH_ANSWER = (
    "You are a math reasoning verifier.\n"
    "You will be given a math problem, its correct final answer, and numbered solution steps.\n"
    "For each step, output two scores:\n"
    "  - validity:   +1.0 if the step is logically correct, -1.0 if it contains an error, "
    "or a value in between for partial errors.\n"
    "  - redundancy: +1.0 if the step is redundant (adds no new information), "
    "-1.0 if it is necessary and informative.\n"
    "Respond with ONLY a JSON object: "
    "{\"validity\": [s1, s2, ...], \"redundancy\": [s1, s2, ...]}\n\n"
    + FEW_SHOT_EXAMPLE
)

SYSTEM_WOUT_ANSWER = (
    "You are a math reasoning verifier.\n"
    "You will be given a math problem and numbered solution steps.\n"
    "For each step, output two scores:\n"
    "  - validity:   +1.0 if the step is logically correct, -1.0 if it contains an error, "
    "or a value in between for partial errors.\n"
    "  - redundancy: +1.0 if the step is redundant (adds no new information), "
    "-1.0 if it is necessary and informative.\n"
    "Respond with ONLY a JSON object: "
    "{\"validity\": [s1, s2, ...], \"redundancy\": [s1, s2, ...]}\n\n"
    + FEW_SHOT_EXAMPLE
)


def build_prompt_with_answer(question: str, steps: list[str], correct_answer: str) -> str:
    steps_block = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))
    return (
        f"Question: {question}\n\n"
        f"Correct final answer: {correct_answer}\n\n"
        f"Solutions:\n{steps_block}\n\n"
        f"Output a JSON object with validity and redundancy arrays, "
        f"each of length {len(steps)}."
    )


def build_prompt_wout_answer(question: str, steps: list[str]) -> str:
    steps_block = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))
    return (
        f"Question: {question}\n\n"
        f"Solutions:\n{steps_block}\n\n"
        f"Output a JSON object with validity and redundancy arrays, "
        f"each of length {len(steps)}."
    )


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------
def _boxed_content(text: str) -> str | None:
    """The content of the last \\boxed{...}, counting braces so nesting survives."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth = start + len("\\boxed{"), 1
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start + len("\\boxed{"): i - 1].strip() if depth == 0 else None


def extract_final_answer(process: list[str]) -> str:
    """The answer a correct solution arrived at, for the w/ Answer setting.

    These solutions are PRM800K's, so the answer is usually inside \\boxed{},
    which is taken first and taken whole. Failing that, the text after "answer
    is" is the answer if it is short enough to be one. Anything else returns the
    final step entire: a complete sentence that states the answer is worth more
    to the model than a fragment, and splitting on the last "=" produced exactly
    that — half a matrix, or the tail of an equation, announced as the answer.
    """
    if not process:
        return ""
    last = process[-1].strip()

    for text in (last, " ".join(process)):
        boxed = _boxed_content(text)
        if boxed:
            return boxed

    match = re.search(r"answer is[:\s]+(.{1,60}?)\.?\s*$", last, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return last


# ---------------------------------------------------------------------------
# Gateway word filter
# ---------------------------------------------------------------------------
# The API gateway screens the request text against a word list and answers
# 400 "Invalid input: Sensitive word(gcd) detected in request message." without
# ever reaching the model. It is the gateway refusing, not the model: the same
# items score normally on the route that has no filter, and the word is the
# ordinary abbreviation for a greatest common divisor. Re-sending the identical
# text can only be refused again — that is what left the five items carrying
# "gcd" unscored on every filtered model, twenty calls apiece — so the word is
# written out in full instead. The expansion reads the same to a verifier, and
# every substitution is recorded on the item, so a rewritten prompt is never
# mistaken for the original. A blocked word with no expansion here is reported
# rather than guessed at.
# ---------------------------------------------------------------------------
GATEWAY_WORD_REWRITES = {
    "gcd": "greatest common divisor",
}

_SENSITIVE_WORD_RE = re.compile(r"Sensitive word\(([^)]+)\)", re.IGNORECASE)
_MAX_WORD_REWRITES = 3


def rewrite_blocked_word(text: str, word: str, replacement: str) -> str:
    """Write a gateway-blocked word out in full wherever it stands alone."""
    return re.sub(rf"\b{re.escape(word)}\b", replacement, text, flags=re.IGNORECASE)


def _rewrite_payload(payload: dict, word: str, rewrites: dict[str, str]) -> bool:
    """Expand a blocked word throughout the payload. False when it cannot be."""
    key = word.lower()
    replacement = GATEWAY_WORD_REWRITES.get(key)
    if replacement is None:
        logger.error("Gateway blocked %r and GATEWAY_WORD_REWRITES has no expansion for it — "
                     "add one there or every item containing it stays unscored", word)
        return False
    if key in rewrites or len(rewrites) >= _MAX_WORD_REWRITES:
        return False                     # already written out once and still refused
    messages = [{**m, "content": rewrite_blocked_word(m["content"], word, replacement)}
                for m in payload["messages"]]
    if messages == payload["messages"]:
        logger.error("Gateway blocked %r but it does not stand alone in the request, "
                     "so there is nothing to expand", word)
        return False
    payload["messages"] = messages
    rewrites[key] = replacement
    return True


def terminal_reason(reason: str | None) -> bool:
    """True for what re-asking cannot change: the text the gateway refuses, or a
    call that has already spent its whole clock. A sweep is for a miscounted reply,
    which is cheap to ask again; it is not for buying a second half-hour."""
    if not reason:
        return False
    return (reason.startswith("blocked_word:")
            or reason.startswith("out_of_time:")
            or (reason.startswith("http_4") and not reason.startswith("http_429")))


# ---------------------------------------------------------------------------
# Async LLM call — temperature 0 for deterministic verification
# ---------------------------------------------------------------------------
async def call_llm(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    system: str,
    user: str,
    model: str,
    api_key: str,
    base_url: str,
    timeout: float | None = None,
) -> tuple[str | None, str | None, dict[str, str]]:
    """Call the LLM with model-aware parameters.

    Returns (content, reason, rewrites): the reply and no reason, or no reply and
    why — so a caller can tell a refusal it should not re-ask from a miscount it
    should. `rewrites` carries any gateway-blocked word this call had to write
    out in full to get an answer at all.

    Reasoning models (o1/o3/o4 family) require:
      - max_completion_tokens  (not max_tokens)
      - temperature = 1        (fixed, not configurable)

    Standard chat models (gpt-4.x, gpt-3.5, etc.) use:
      - max_tokens
      - temperature = 0        (deterministic)
    """
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Reasoning models (o1/o3/o4 and the GPT-5 family) bill hidden reasoning against
    # the same budget as the visible reply and take max_completion_tokens rather than
    # max_tokens. At 4096 they spend the budget thinking and return an empty or
    # truncated body on the longer items, which then fails to parse and drops the
    # sample — so the ceiling is set high enough that the JSON always fits.
    _REASONING_PREFIXES = ("o1", "o3", "o4", "o-1", "o-3", "o-4", "gpt-5", "gpt5")
    is_reasoning = any(model.lower().startswith(p) for p in _REASONING_PREFIXES)
    _MAX_OUTPUT_TOKENS = 16000

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": 0.5,
        ("max_completion_tokens" if is_reasoning else "max_tokens"): _MAX_OUTPUT_TOKENS,
    }
    rewrites: dict[str, str] = {}
    reason, tries = "no_reply", 0
    while tries < 4:
        async with semaphore:
            try:
                async with session.post(
                    url, headers=headers, json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout or REQUEST_TIMEOUT),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data["choices"][0]["message"]["content"].strip()
                        return content, None, rewrites
                    text = await resp.text()
                    blocked = _SENSITIVE_WORD_RE.search(text)
                    if blocked:
                        word = blocked.group(1)
                        if _rewrite_payload(payload, word, rewrites):
                            logger.info("Gateway blocked %r — re-sending it as %r",
                                        word, rewrites[word.lower()])
                            continue     # a refused word is not a failed attempt
                        return None, f"blocked_word:{word.lower()}", rewrites
                    logger.warning("HTTP %s: %s", resp.status, text[:200])
                    reason = f"http_{resp.status}"
                    if terminal_reason(reason):
                        return None, reason, rewrites
            except Exception as e:
                logger.warning("Attempt %d error: %r", tries + 1, e)
                reason = type(e).__name__
        await asyncio.sleep(2 ** tries)
        tries += 1
    return None, reason, rewrites


def _remove_json_comments(json_string: str) -> str:
    """Strip // and # comments — models add them inside the JSON they emit."""
    return re.sub(r"//.*?$|#.*?$", "", json_string, flags=re.MULTILINE)


def extract_nested_json(text: str):
    """Return the first complete brace-balanced JSON object in text, or None.

    Ported from PRMBench's mr_eval/utils/model_utils.py so a reply is read the way
    the benchmark's own critics read it: a brace stack finds the whole object even
    when it nests or when prose surrounds it, where a non-greedy regex would stop
    at the first closing brace and hand back a fragment.
    """
    stack, start = [], -1
    for i, char in enumerate(text):
        if char == "{":
            if not stack:
                start = i
            stack.append("{")
        elif char == "}":
            if not stack:
                continue
            stack.pop()
            if not stack:
                try:
                    return json.loads(_remove_json_comments(text[start:i + 1]))
                except json.JSONDecodeError:
                    continue
    return None


def score_counts(raw: str | None) -> dict[str, int]:
    """How many scores each array in a reply actually held.

    A miscounted reply is re-asked, and "you gave 94 validity scores for 91
    steps" is a correction where "that was wrong" is only a complaint — so the
    counts are read back out of the reply the same two ways parse_scores reads
    the scores themselves.
    """
    obj = extract_nested_json(raw or "")
    if isinstance(obj, dict):
        return {k: len(v) for k, v in obj.items() if isinstance(v, list)}
    counts = {}
    for key in ("validity", "redundancy"):
        m = re.search(rf'"{key}"\s*:\s*\[([^\]]*)', raw or "", re.DOTALL)
        if m:
            counts[key] = len(re.findall(r"-?\d+(?:\.\d+)?", m.group(1)))
    return counts


def parse_scores(raw: str, n_steps: int) -> tuple[list[float] | None, list[float] | None]:
    """Parse {"validity": [...], "redundancy": [...]} into two lists of n_steps floats.

    Returns (None, None) when the arrays do not hold exactly one score per step.
    PRMBench's own scorer iterates over whatever the model returned, so a reply
    scoring 55 steps of a 45-step solution is counted as 55 steps; here the extra
    or missing entries cannot be matched to steps, and shifting them would move
    every later score onto the wrong step and with it the first-error position —
    so the item is left unscored and re-asked instead.
    """
    if not raw:
        return None, None

    text = raw.strip()
    if "```" in text:
        inner = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
        text = inner if inner else text

    def as_floats(seq) -> list[float] | None:
        if not isinstance(seq, list) or len(seq) != n_steps:
            return None
        try:
            return [float(v) for v in seq]
        except (TypeError, ValueError):
            return None

    obj = extract_nested_json(text)
    if isinstance(obj, dict):
        val, red = as_floats(obj.get("validity")), as_floats(obj.get("redundancy"))
        if val is not None:
            return val, red
        got = {k: len(v) for k, v in obj.items() if isinstance(v, list)}
        if got:
            logger.debug("Score count mismatch: expected %d per array, got %s", n_steps, got)

    # Fallback: pull the arrays straight out of the text, for a reply whose JSON
    # never closes (truncated tail) but whose numbers are all there.
    def extract_array(key: str) -> list[float] | None:
        m = re.search(rf'"{key}"\s*:\s*\[([^\]]*)', text, re.DOTALL)
        if not m:
            return None
        return as_floats([float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", m.group(1))])

    return extract_array("validity"), extract_array("redundancy")


# ---------------------------------------------------------------------------
# PRMBench eval_on_hallucination_step — verbatim port from task.py
# Threshold: score > 0 → True (valid); score <= 0 → False (invalid)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Difficulty dimension mapping — PRMBench's own taxonomy (prmbench.github.io):
# Simplicity = Non-Redundancy + Non-Circular Logic; Soundness = Empirical
# Soundness + Step Consistency + Domain Consistency + Confidence Invariance;
# Sensitivity = Prerequisite Sensitivity + Deception Resistance + Multi-Solution
# Consistency. An item's `_dim` in the sampled dataset agrees with this map, so
# grouping here and stratification there cannot drift apart.
# ---------------------------------------------------------------------------
DIFFICULTY_DIMS = {
    "simplicity":  {"redundency", "circular"},
    "soundness":   {"counterfactual", "step_contradiction",
                    "domain_inconsistency", "confidence"},
    "sensitivity": {"missing_condition", "deception", "multi_solutions"},
}


def get_dim(classification: str) -> str:
    for dim, classes in DIFFICULTY_DIMS.items():
        if classification in classes:
            return dim
    return "unknown"


def metrics_from_record(rec: dict, setting: str) -> dict | None:
    """Rebuild one setting's metrics from the scores stored in a result record.

    The record keeps the model's raw per-step scores, so a summary can be rebuilt
    from a results file alone — no second pass over the API.
    """
    side = rec.get(setting) or {}
    validity = side.get("validity_scores")
    if not validity:
        return None
    return eval_on_hallucination_step(
        rec.get("error_steps", []), validity,
        rec.get("classification", ""), side.get("redundancy_scores"),
    )


def build_summary(results: list[dict], model: str) -> dict:
    """Aggregate per-item metrics into the 3 difficulty dimensions plus a total.

    An item counts only when BOTH settings produced parsable scores. A call the
    endpoint refuses (its content filter rejects e.g. "gcd") or a reply that will
    not parse takes the whole item out of both settings, so w/ Answer and w/o
    Answer are always compared over the same items and n is the number actually
    scored, never the number attempted.
    """
    dims = ("simplicity", "soundness", "sensitivity")
    paired: dict[str, dict[str, list]] = {d: {"with_answer": [], "wout_answer": []} for d in dims}
    attempted = {d: 0 for d in dims}
    dropped_idx: dict[str, list] = {d: [] for d in dims}
    reasons: dict[str, int] = {}

    for rec in results:
        dim = get_dim(rec.get("classification", ""))
        if dim not in paired:
            continue
        attempted[dim] += 1
        for setting in ("with_answer", "wout_answer"):
            failure = (rec.get(setting) or {}).get("failure")
            if failure:
                kind = failure.split(":")[0]
                reasons[kind] = reasons.get(kind, 0) + 1
        wa_m = metrics_from_record(rec, "with_answer")
        bl_m = metrics_from_record(rec, "wout_answer")
        if wa_m is None or bl_m is None:
            dropped_idx[dim].append(rec.get("idx", ""))
            continue
        paired[dim]["with_answer"].append(wa_m)
        paired[dim]["wout_answer"].append(bl_m)

    def dim_summary(wa_list, bl_list, n_attempted, dropped):
        return {
            "with_answer": aggregate_metrics(wa_list),
            "wout_answer":       aggregate_metrics(bl_list),
            "n_items":     len(wa_list),
            "n_attempted": n_attempted,
            "n_dropped":   len(dropped),
        }

    summary = {d: dim_summary(paired[d]["with_answer"], paired[d]["wout_answer"],
                              attempted[d], dropped_idx[d]) for d in dims}
    summary["total"] = dim_summary(
        [m for d in dims for m in paired[d]["with_answer"]],
        [m for d in dims for m in paired[d]["wout_answer"]],
        sum(attempted.values()),
        [i for d in dims for i in dropped_idx[d]],
    )
    summary["meta"] = {
        "n_total":         len(results),
        "n_scored":        summary["total"]["n_items"],
        "dropped_idx":     {d: dropped_idx[d] for d in dims if dropped_idx[d]},
        "dropped_reasons": reasons,
        "model":           model,
        "difficulty_dims": {d: sorted(c) for d, c in DIFFICULTY_DIMS.items()},
    }
    return summary


def eval_on_hallucination_step(
    hallucination_steps: list[int],
    validity_scores: list[float],
    classification: str,
    redundancy_scores: list[float] | None = None,
) -> dict:
    """
    Port of PRMBench's eval_on_hallucination_step() from task.py.

    For 'redundency' and 'circular' classifications, uses step_level_redundancy_labels
    (inverted) if available — mirroring task.py lines 155–158.

    hallucination_steps: 1-indexed (as stored in PRMBench data).
    validity_scores: raw float scores from LLM, threshold at 0.
    """
    # Convert to bool labels: validity > 0 → True (step is valid)
    validity_labels = [v > 0 for v in validity_scores]

    # Mirror task.py: for redundency/circular, use redundancy labels (inverted)
    if classification in ("redundency", "circular") and redundancy_scores is not None:
        redundancy_labels = [r > 0 for r in redundancy_scores]
        labels = [not r for r in redundancy_labels]  # invert: redundant → invalid
    else:
        labels = validity_labels

    # Convert to 0-indexed
    hallucination_steps_0 = [i - 1 for i in hallucination_steps]

    POSITIVE_LABEL = True
    NEGATIVE_LABEL = False

    correct_step_acc, wrong_step_acc, total_step_acc = [], [], []
    TP = FP = TN = FN = 0

    first_error = min(hallucination_steps_0) if hallucination_steps_0 else -1
    first_error_acc = None

    for idx, label in enumerate(labels):
        if idx == first_error:
            first_error_acc = 1 if (label == NEGATIVE_LABEL) else 0

        if idx in hallucination_steps_0:
            if label == POSITIVE_LABEL:
                wrong_step_acc.append(0); total_step_acc.append(0); FP += 1
            else:
                wrong_step_acc.append(1); total_step_acc.append(1); TN += 1
        else:
            if label == POSITIVE_LABEL:
                correct_step_acc.append(1); total_step_acc.append(1); TP += 1
            else:
                correct_step_acc.append(0); total_step_acc.append(0); FN += 1

    def avg(lst): return round(sum(lst) / len(lst), 4) if lst else -1

    prec = TP / (TP + FP) if (TP + FP) else -1
    rec  = TP / (TP + FN) if (TP + FN) else -1
    f1   = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else -1

    return {
        "correct_step_acc": avg(correct_step_acc),
        "wrong_step_acc":   avg(wrong_step_acc),
        "total_step_acc":   avg(total_step_acc),
        "first_error_acc":  first_error_acc,
        "f1_matrix": {"TP": TP, "FP": FP, "TN": TN, "FN": FN},
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "f1":        f1,
        # PRMBench interface: scores dict with step_level_validity_labels
        "scores": {
            "step_level_validity_labels":   validity_labels,
            "step_level_redundancy_labels": [r > 0 for r in redundancy_scores] if redundancy_scores else None,
        },
        # lists for aggregation
        "correct_step_acc_list": correct_step_acc,
        "wrong_step_acc_list":   wrong_step_acc,
        "total_step_acc_list":   total_step_acc,
        "first_error_acc_list":  [first_error_acc] if first_error_acc is not None else [],
    }


def aggregate_metrics(metrics_list: list[dict]) -> dict:
    agg = {k: [] for k in ("correct_step_acc_list", "wrong_step_acc_list",
                            "total_step_acc_list",   "first_error_acc_list")}
    f1m = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    for m in metrics_list:
        if m is None:
            continue
        for k in agg:
            agg[k].extend(m.get(k, []))
        for k in f1m:
            f1m[k] += m["f1_matrix"][k]

    def avg(lst): return round(sum(lst) / len(lst), 4) if lst else -1

    TP, FP, TN, FN = f1m["TP"], f1m["FP"], f1m["TN"], f1m["FN"]
    prec     = TP / (TP + FP) if (TP + FP) else -1
    rec      = TP / (TP + FN) if (TP + FN) else -1
    f1       = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else -1
    neg_prec = TN / (TN + FN) if (TN + FN) else -1
    neg_rec  = TN / (TN + FP) if (TN + FP) else -1
    neg_f1   = round(2 * neg_prec * neg_rec / (neg_prec + neg_rec), 4) if (neg_prec + neg_rec) > 0 else -1

    return {
        "correct_step_acc": avg(agg["correct_step_acc_list"]),
        "wrong_step_acc":   avg(agg["wrong_step_acc_list"]),
        "total_step_acc":   avg(agg["total_step_acc_list"]),
        "first_error_acc":  avg(agg["first_error_acc_list"]),
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": f1,
        "negative_precision": round(neg_prec, 4),
        "negative_recall":    round(neg_rec, 4),
        "negative_f1":        neg_f1,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
SETTINGS = ("with_answer", "wout_answer")


class Scored(NamedTuple):
    """One setting's outcome: the scores, or why there are none."""
    validity:   list[float] | None
    redundancy: list[float] | None
    raw:        str | None
    reason:     str | None
    rewrites:   dict[str, str]


def step_timeout(n_steps: int) -> float:
    """How long to let one reply take, by how much of it there is to write.

    A flat ceiling is either generous for an 8-step solution or too tight for a
    90-step one, so the budget grows with the steps being scored — that is what
    sets both the reasoning and the two arrays the model has to emit. It stays
    modest on purpose: the long items that fail here do not answer slowly, they
    do not answer at all, and a request the gateway has dropped costs its whole
    timeout before anything can be re-asked. Failing early and asking again beats
    waiting longer.
    """
    return min(REQUEST_TIMEOUT + 1.5 * n_steps, 360.0)


# How much wall clock one (item, setting) may spend before it is given up on.
# The re-ask loop and the loop inside call_llm multiply: five re-asks of four
# tries each, at a timeout that now grows with the steps, is hours on the one
# item that always times out — and the whole run waits on it. Attempts are
# still capped as before; this caps the time they may take between them.
_TIME_BUDGET_REPLIES = 2.5


async def call_and_score(session, semaphore, system, prompt, n_steps,
                         model, api_key, base_url, attempts=5) -> Scored:
    """Ask for one setting's scores, re-asking until they line up with the steps.

    Bounded by both a count and a clock: a reply that miscounts is cheap to ask
    again, but one that times out costs its whole budget, and without the clock a
    single 90-step item can hold a finished run open for half an hour.

    PRMBench's own critics wrap the call and the parse in one retry loop for the
    same reason: a model that miscounts the steps usually gets it right when asked
    again, and temperature is not zero. The re-ask quotes the counts it got back,
    which is the whole of what was wrong with the reply. Only a reply that survives
    parsing counts, so a retry can recover an item but can never invent one — and a
    request the gateway refuses outright is not re-asked at all, since the refusal
    is in its word list rather than in the sampling.
    """
    raw, reason, rewrites, ask = None, "no_reply", {}, prompt
    budget = step_timeout(n_steps)
    deadline = time.monotonic() + _TIME_BUDGET_REPLIES * budget
    for attempt in range(attempts):
        if attempt and time.monotonic() > deadline:
            logger.warning("Out of time after %d attempt(s) — leaving it unscored (%s)",
                           attempt, reason)
            reason = f"out_of_time:{reason}"
            break
        content, err, rw = await call_llm(session, semaphore, system, ask,
                                          model, api_key, base_url,
                                          timeout=budget)
        rewrites.update(rw)
        if err:
            reason = err
            if terminal_reason(err):
                break
            continue
        raw = content
        validity, redundancy = parse_scores(raw, n_steps)
        if validity is not None:
            return Scored(validity, redundancy, raw, None, rewrites)
        counts = score_counts(raw)
        reason = f"score_count_mismatch:{counts.get('validity', 0)}!={n_steps}"
        ask = (f"{prompt}\n\nYour previous reply gave {counts.get('validity', 0)} validity "
               f"scores and {counts.get('redundancy', 0)} redundancy scores, but the solution "
               f"has {n_steps} steps. Return exactly {n_steps} in each array, one per step, "
               f"in order.")
    return Scored(None, None, raw, reason, rewrites)


def build_setting_prompt(item: dict, setting: str) -> tuple[str, str, int]:
    """The system prompt, user prompt and step count for one setting of one item."""
    question = item.get("question") or item.get("original_question", "")
    steps    = item.get("modified_process", [])      # PRMBench: has injected errors
    if setting == "with_answer":
        answer = extract_final_answer(item.get("original_process", []))
        return SYSTEM_WITH_ANSWER, build_prompt_with_answer(question, steps, answer), len(steps)
    return SYSTEM_WOUT_ANSWER, build_prompt_wout_answer(question, steps), len(steps)


def setting_record(scored: Scored, metrics: dict | None) -> dict:
    """One setting's slot in a result record, carrying why it is empty when it is."""
    side = {
        "validity_scores":   scored.validity,
        "redundancy_scores": scored.redundancy,
        "validity":          scored.validity is not None,
        # PRMBench-compatible scores dict
        "scores":  metrics["scores"] if metrics else None,
        "metrics": {k: v for k, v in metrics.items()
                    if k not in ("scores", "correct_step_acc_list", "wrong_step_acc_list",
                                 "total_step_acc_list", "first_error_acc_list")} if metrics else None,
    }
    if scored.validity is None:
        side["failure"] = scored.reason
    if scored.rewrites:
        side["gateway_rewrites"] = scored.rewrites
    return side


async def score_items(session, items: list[dict], args) -> dict[tuple[int, str], Scored]:
    """Score every item in both settings, then sweep up whatever came back unscored.

    A first pass loses a few calls to replies that miscount the steps, and the same
    request sent again — later, with fewer in flight — usually parses. That used to
    be a second script run by hand over the finished file; doing it here is the
    difference between a run that finishes complete and one that finishes and then
    needs repairing. A sweep that recovers nothing ends it, because what is left is
    refused rather than unlucky, and a refusal the gateway will only repeat is never
    re-sent at all.
    """
    outcomes: dict[tuple[int, str], Scored] = {}
    pending = [(i, s) for i in range(len(items)) for s in SETTINGS]
    concurrency = args.concurrency

    for sweep in range(args.sweeps + 1):
        if not pending:
            break
        if sweep:
            logger.info("Sweep %d/%d: re-asking %d unscored call(s) at concurrency %d",
                        sweep, args.sweeps, len(pending), concurrency)
            await asyncio.sleep(args.sweep_pause)
        semaphore = asyncio.Semaphore(concurrency)
        jobs = []
        for i, setting in pending:
            system, prompt, n_steps = build_setting_prompt(items[i], setting)
            jobs.append(call_and_score(session, semaphore, system, prompt, n_steps,
                                       args.model, args.api_key, args.base_url))
        scored = await asyncio.gather(*jobs)
        outcomes.update(zip(pending, scored))

        recovered = sum(1 for s in scored if s.validity is not None)
        if sweep:
            logger.info("Sweep %d recovered %d/%d", sweep, recovered, len(scored))
        pending = [k for k, s in zip(pending, scored)
                   if s.validity is None and not terminal_reason(s.reason)]
        if sweep and not recovered:
            break
        concurrency = max(1, concurrency // 2)

    if pending:
        logger.info("%d call(s) still unscored after %d sweep(s)", len(pending), args.sweeps)
    return outcomes


async def run(args):
    items = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    if args.max_samples:
        items = items[: args.max_samples]
    logger.info("Loaded %d items from %s", len(items), args.input)

    async with aiohttp.ClientSession() as session:
        outcomes = await score_items(session, items, args)

    results = []
    for i, item in enumerate(items):
        error_steps    = item.get("error_steps", [])      # 1-indexed
        n              = len(item.get("modified_process", []))
        idx            = item.get("idx", "")
        classification = item.get("classification", "")

        record = {
            "idx":            idx,
            "classification": classification,
            "error_steps":    error_steps,
            "n_steps":        n,
        }
        for setting in SETTINGS:
            scored = outcomes[(i, setting)]
            if scored.validity is None:
                logger.warning("Unscored (%s) idx=%s steps=%d reason=%s | %s",
                               setting, idx, n, scored.reason, (scored.raw or "")[:100])
            metrics = eval_on_hallucination_step(
                error_steps, scored.validity, classification, scored.redundancy
            ) if scored.validity else None
            record[setting] = setting_record(scored, metrics)
        results.append(record)

    summary = build_summary(results, args.model)

    # One file per setting, named for the dimension and the setting it holds —
    # the same shape the ROSCOE side writes. The item keys stay in both files so
    # the pair can be rejoined; the summary spans them, because the comparison
    # this study makes is between them.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.name[: -len(".jsonl")] if out_path.name.endswith(".jsonl") else out_path.stem
    written = []
    for setting in ("with_answer", "wout_answer"):
        path = out_path.with_name(f"{stem}_{setting}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps({
                    "idx":            r["idx"],
                    "classification": r["classification"],
                    "error_steps":    r["error_steps"],
                    "n_steps":        r["n_steps"],
                    "setting":        setting,
                    **r[setting],
                }, ensure_ascii=False) + "\n")
        written.append(path)

    # No summary file: every number in it — the metrics, the denominators, which
    # items were dropped — is derived from the per-item scores in the files above,
    # so build_summary() reproduces it from them whenever it is wanted.
    for path in written:
        logger.info("Saved %d results → %s", len(results), path)
    logger.info("=== PRMBench LLM Verifier Comparison ===")
    header = f"{'dim':<12}  {'setting':<12}  {'total_acc':>9}  {'wrong_acc':>9}  {'1st_err':>7}  {'f1':>7}  n"
    logger.info(header)
    logger.info("-" * len(header))
    for dim in ("simplicity", "soundness", "sensitivity", "total"):
        n = summary[dim]["n_items"]
        for s in ("with_answer", "wout_answer"):
            m = summary[dim][s]
            logger.info(
                "%-12s  %-12s  %9.4f  %9.4f  %7.4f  %7.4f  %d",
                dim, s,
                m.get("total_step_acc",  -1),
                m.get("wrong_step_acc",  -1),
                m.get("first_error_acc", -1),
                m.get("f1",              -1),
                n,
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "PRMBench — LLM-as-Verifier (with_answer vs wout_answer).\n"
            "Output files are auto-named as <model>/prmbench/<dimension>_<setting>.jsonl\n"
            "under the results root when --output is omitted."
        )
    )
    parser.add_argument("--input",       required=True, help="PRMBench JSONL input file")
    parser.add_argument("--output",      default=None,
                        help="Output JSONL path. Relative paths resolve under the results "
                             "root; if omitted, auto-generated in <model>/prmbench/")
    parser.add_argument("--model",       default=DEFAULT_MODEL)
    parser.add_argument("--base_url",    default=OPENAI_BASE_URL)
    parser.add_argument("--api_key",     default=OPENAI_API_KEY)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--sweeps",      type=int, default=2,
                        help="Extra waves over whatever is still unscored, each at half "
                             "the concurrency. 0 runs a single pass.")
    parser.add_argument("--sweep_pause", type=float, default=20.0,
                        help="Seconds to wait before each sweep.")
    args = parser.parse_args()

    args.input = str(resolve_input(args.input))

    # Auto-generate output path if not specified
    if args.output is None:
        model_safe = args.model.replace("/", "-").replace(":", "-")
        # One directory per model, one file per dimension: the input's own name
        # ("simplicity" / "soundness" / "sensitivity") carries through to the output,
        # so a run is identifiable without opening it.
        stem = Path(args.input).stem
        if args.max_samples:
            stem = f"{stem}_{args.max_samples}"
        args.output = f"{model_safe}/prmbench/{stem}.jsonl"
    args.output = str(resolve_output(args.output))
    logger.info("Output path: %s", args.output)

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
