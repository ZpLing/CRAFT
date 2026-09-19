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
from pathlib import Path

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
def extract_final_answer(process: list[str]) -> str:
    if not process:
        return ""
    last = process[-1]
    match = re.search(r"(?:answer is|=)\s*(.+?)\.?\s*$", last, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return last.strip()


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
) -> str | None:
    """Call the LLM with model-aware parameters.

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
    for attempt in range(4):
        async with semaphore:
            try:
                async with session.post(
                    url, headers=headers, json=payload,
                    timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"].strip()
                    text = await resp.text()
                    logger.warning("HTTP %s: %s", resp.status, text[:200])
            except Exception as e:
                logger.warning("Attempt %d error: %s", attempt + 1, e)
        await asyncio.sleep(2 ** attempt)
    return None


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

    for rec in results:
        dim = get_dim(rec.get("classification", ""))
        if dim not in paired:
            continue
        attempted[dim] += 1
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
async def call_and_score(session, semaphore, system, prompt, n_steps,
                         model, api_key, base_url, attempts=5):
    """Ask for one setting's scores, re-asking until they line up with the steps.

    PRMBench's own critics wrap the call and the parse in one retry loop for the
    same reason: a model that miscounts the steps usually gets it right when asked
    again, and temperature is not zero. Only a reply that survives parsing counts,
    so a retry can recover an item but can never invent one.
    """
    raw = None
    for attempt in range(attempts):
        ask = prompt if attempt == 0 else (
            f"{prompt}\n\nYour previous reply did not contain exactly {n_steps} "
            f"scores in each array. Return exactly {n_steps} per array, one per step."
        )
        raw = await call_llm(session, semaphore, system, ask, model, api_key, base_url)
        if raw:
            validity, redundancy = parse_scores(raw, n_steps)
            if validity is not None:
                return validity, redundancy, raw
    return None, None, raw


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

    semaphore = asyncio.Semaphore(args.concurrency)

    async with aiohttp.ClientSession() as session:
        wa_tasks, bl_tasks = [], []
        for item in items:
            question = item.get("question") or item.get("original_question", "")
            # PRMBench: verify modified_process (has injected errors)
            steps          = item.get("modified_process", [])
            correct_answer = extract_final_answer(item.get("original_process", []))

            wa_tasks.append(call_and_score(
                session, semaphore,
                SYSTEM_WITH_ANSWER,
                build_prompt_with_answer(question, steps, correct_answer),
                len(steps), args.model, args.api_key, args.base_url,
            ))
            bl_tasks.append(call_and_score(
                session, semaphore,
                SYSTEM_WOUT_ANSWER,
                build_prompt_wout_answer(question, steps),
                len(steps), args.model, args.api_key, args.base_url,
            ))

        wa_outputs, bl_outputs = await asyncio.gather(
            asyncio.gather(*wa_tasks),
            asyncio.gather(*bl_tasks),
        )

    results = []
    wa_metrics_list, bl_metrics_list = [], []

    for item, wa_out, bl_out in zip(items, wa_outputs, bl_outputs):
        steps          = item.get("modified_process", [])
        error_steps    = item.get("error_steps", [])      # 1-indexed
        n              = len(steps)
        idx            = item.get("idx", "")
        classification = item.get("classification", "")

        wa_val, wa_red, wa_raw = wa_out
        bl_val, bl_red, bl_raw = bl_out

        if wa_val is None:
            logger.warning("Unscored after retries (with_answer) idx=%s steps=%d | %s",
                           idx, n, (wa_raw or "")[:100])
        if bl_val is None:
            logger.warning("Unscored after retries (wout_answer) idx=%s steps=%d | %s",
                           idx, n, (bl_raw or "")[:100])

        wa_m = eval_on_hallucination_step(error_steps, wa_val, classification, wa_red) if wa_val else None
        bl_m = eval_on_hallucination_step(error_steps, bl_val, classification, bl_red) if bl_val else None

        wa_metrics_list.append(wa_m)
        bl_metrics_list.append(bl_m)

        results.append({
            "idx":            idx,
            "classification": classification,
            "error_steps":    error_steps,
            "n_steps":        n,
            "with_answer": {
                "validity_scores":   wa_val,
                "redundancy_scores": wa_red,
                "validity":          wa_val is not None,
                # PRMBench-compatible scores dict
                "scores": wa_m["scores"] if wa_m else None,
                "metrics": {k: v for k, v in wa_m.items()
                            if k not in ("scores", "correct_step_acc_list",
                                         "wrong_step_acc_list", "total_step_acc_list",
                                         "first_error_acc_list")} if wa_m else None,
            },
            "wout_answer": {
                "validity_scores":   bl_val,
                "redundancy_scores": bl_red,
                "validity":          bl_val is not None,
                "scores": bl_m["scores"] if bl_m else None,
                "metrics": {k: v for k, v in bl_m.items()
                            if k not in ("scores", "correct_step_acc_list",
                                         "wrong_step_acc_list", "total_step_acc_list",
                                         "first_error_acc_list")} if bl_m else None,
            },
        })

    summary = build_summary(results, args.model)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary_path = out_path.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved %d results → %s", len(results), out_path)
    logger.info("Summary → %s", summary_path)
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
            "Output files are auto-named as <model>/prmbench/<dimension>_<api>_results.jsonl\n"
            "under the results root when --output is omitted. "
            "Use --api_name to label the API in the filename."
        )
    )
    parser.add_argument("--input",       required=True, help="PRMBench JSONL input file")
    parser.add_argument("--output",      default=None,
                        help="Output JSONL path. Relative paths resolve under the results "
                             "root; if omitted, auto-generated in <model>/prmbench/")
    parser.add_argument("--api_name",    default=None,
                        help="API label for filename. "
                             "Derived from a short hash of base_url if omitted.")
    parser.add_argument("--model",       default=DEFAULT_MODEL)
    parser.add_argument("--base_url",    default=OPENAI_BASE_URL)
    parser.add_argument("--api_key",     default=OPENAI_API_KEY)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    args.input = str(resolve_input(args.input))

    # Auto-generate output path if not specified
    if args.output is None:
        # Derive api_name from a short hash of base_url if not given
        api_label = args.api_name
        if not api_label:
            url = (args.base_url or "").strip().lower()
            api_label = "api_" + hashlib.md5(url.encode("utf-8")).hexdigest()[:6] if url else "api"
        model_safe = args.model.replace("/", "-").replace(":", "-")
        # One directory per model, one file per dimension: the input's own name
        # ("simplicity" / "soundness" / "sensitivity") carries through to the output,
        # so a run is identifiable without opening it.
        stem = Path(args.input).stem
        if args.max_samples:
            stem = f"{stem}_{args.max_samples}"
        args.output = f"{model_safe}/prmbench/{stem}_{api_label}_results.jsonl"
    args.output = str(resolve_output(args.output))
    logger.info("Output path: %s", args.output)

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
