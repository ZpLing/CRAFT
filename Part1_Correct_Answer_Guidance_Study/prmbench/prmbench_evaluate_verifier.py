from __future__ import annotations
"""
PRMBench — LLM-as-Verifier: with_answer vs blind

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
  B (blind):       LLM verifies steps without knowing the answer.

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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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

SYSTEM_BLIND = (
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


def build_prompt_blind(question: str, steps: list[str]) -> str:
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

    # Detect reasoning model by name prefix
    _REASONING_PREFIXES = ("o1", "o3", "o4", "o-1", "o-3", "o-4")
    is_reasoning = any(model.lower().startswith(p) for p in _REASONING_PREFIXES)

    # Detect reasoning model — requires max_completion_tokens instead of max_tokens
    _REASONING_PREFIXES = ("o1", "o3", "o4", "o-1", "o-3", "o-4")
    is_reasoning = any(model.lower().startswith(p) for p in _REASONING_PREFIXES)

    if is_reasoning:
        # o1/o3/o4 family: must use max_completion_tokens (not max_tokens)
        payload = {
            "model":                 model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "temperature":           0.5,
            "max_completion_tokens": 4096,
        }
    else:
        # Standard chat models (gpt-4.x etc.): use max_tokens
        payload = {
            "model":       model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "temperature": 0.5,
            "max_tokens":  4096,
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


def parse_scores(raw: str, n_steps: int) -> tuple[list[float] | None, list[float] | None]:
    """
    Parse the JSON output: {"validity": [...], "redundancy": [...]}.
    Returns (validity_scores, redundancy_scores) or (None, None) on failure.

    Handles:
    - Plain JSON object
    - JSON wrapped in ```json ... ``` code fences (Gemini-style)
    - Truncated JSON arrays (partial output due to token limits)
    """
    if not raw:
        return None, None

    # Strip code fences if present
    text = raw.strip()
    if "```" in text:
        # Extract content between first ``` and last ```
        inner = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
        text = inner if inner else text

    # Try to find a complete JSON object first
    match = re.search(r"\{[\s\S]*?\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            val = obj.get("validity", [])
            red = obj.get("redundancy", [])
            if isinstance(val, list) and len(val) == n_steps:
                val = [float(v) for v in val]
            else:
                val = None
            if isinstance(red, list) and len(red) == n_steps:
                red = [float(v) for v in red]
            else:
                red = None
            return val, red
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: extract arrays directly from raw text (handles truncated JSON)
    def extract_array(key: str) -> list[float] | None:
        m = re.search(rf'"{key}"\s*:\s*\[([^\]]*)', text, re.DOTALL)
        if not m:
            return None
        nums = re.findall(r"-?\d+(?:\.\d+)?", m.group(1))
        floats = [float(x) for x in nums]
        return floats if len(floats) == n_steps else None

    val = extract_array("validity")
    red = extract_array("redundancy")
    return val, red


# ---------------------------------------------------------------------------
# PRMBench eval_on_hallucination_step — verbatim port from task.py
# Threshold: score > 0 → True (valid); score <= 0 → False (invalid)
# ---------------------------------------------------------------------------
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

            wa_tasks.append(call_llm(
                session, semaphore,
                SYSTEM_WITH_ANSWER,
                build_prompt_with_answer(question, steps, correct_answer),
                args.model, args.api_key, args.base_url,
            ))
            bl_tasks.append(call_llm(
                session, semaphore,
                SYSTEM_BLIND,
                build_prompt_blind(question, steps),
                args.model, args.api_key, args.base_url,
            ))

        wa_outputs, bl_outputs = await asyncio.gather(
            asyncio.gather(*wa_tasks),
            asyncio.gather(*bl_tasks),
        )

    results = []
    wa_metrics_list, bl_metrics_list = [], []

    for item, wa_raw, bl_raw in zip(items, wa_outputs, bl_outputs):
        steps          = item.get("modified_process", [])
        error_steps    = item.get("error_steps", [])      # 1-indexed
        n              = len(steps)
        idx            = item.get("idx", "")
        classification = item.get("classification", "")

        wa_val, wa_red = parse_scores(wa_raw, n) if wa_raw else (None, None)
        bl_val, bl_red = parse_scores(bl_raw, n) if bl_raw else (None, None)

        if wa_val is None:
            logger.warning("Parse failed (with_answer) idx=%s | %s", idx, (wa_raw or "")[:100])
        if bl_val is None:
            logger.warning("Parse failed (blind) idx=%s | %s", idx, (bl_raw or "")[:100])

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
            "blind": {
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

    # ---------------------------------------------------------------------------
    # Difficulty dimension mapping (PRMBench paper taxonomy)
    # ---------------------------------------------------------------------------
    DIFFICULTY_DIMS = {
        "simplicity":  {"redundency"},
        "soundness":   {"step_contradiction", "domain_inconsistency", "counterfactual"},
        "sensitivity": {"circular", "confidence", "deception", "missing_condition"},
    }

    def get_dim(classification: str) -> str:
        for dim, classes in DIFFICULTY_DIMS.items():
            if classification in classes:
                return dim
        return "unknown"

    # Group per-item metrics by difficulty dimension
    # Structure: {dim: {"with_answer": [...], "blind": [...]}}
    dim_metrics: dict[str, dict[str, list]] = {
        dim: {"with_answer": [], "blind": []}
        for dim in ("simplicity", "soundness", "sensitivity")
    }

    for item_result, wa_m, bl_m in zip(results, wa_metrics_list, bl_metrics_list):
        dim = get_dim(item_result["classification"])
        if dim in dim_metrics:
            dim_metrics[dim]["with_answer"].append(wa_m)
            dim_metrics[dim]["blind"].append(bl_m)

    # Build summary: 3 difficulty dims + total, each with with_answer and blind
    def build_dim_summary(wa_list, bl_list, n_items):
        return {
            "with_answer": aggregate_metrics([m for m in wa_list if m]),
            "blind":       aggregate_metrics([m for m in bl_list if m]),
            "n_items": n_items,
        }

    summary = {
        "simplicity": build_dim_summary(
            dim_metrics["simplicity"]["with_answer"],
            dim_metrics["simplicity"]["blind"],
            len(dim_metrics["simplicity"]["with_answer"]),
        ),
        "soundness": build_dim_summary(
            dim_metrics["soundness"]["with_answer"],
            dim_metrics["soundness"]["blind"],
            len(dim_metrics["soundness"]["with_answer"]),
        ),
        "sensitivity": build_dim_summary(
            dim_metrics["sensitivity"]["with_answer"],
            dim_metrics["sensitivity"]["blind"],
            len(dim_metrics["sensitivity"]["with_answer"]),
        ),
        "total": build_dim_summary(
            wa_metrics_list,
            bl_metrics_list,
            len(results),
        ),
        "meta": {
            "n_total":             len(results),
            "n_with_answer_valid": sum(1 for r in results if r["with_answer"]["validity"]),
            "n_blind_valid":       sum(1 for r in results if r["blind"]["validity"]),
            "model":               args.model,
            "difficulty_dims":     {dim: list(classes) for dim, classes in DIFFICULTY_DIMS.items()},
        },
    }

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
        for s in ("with_answer", "blind"):
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
            "PRMBench — LLM-as-Verifier (with_answer vs blind).\n"
            "Output files are auto-named as prmbench/results/<N>_<model>_<api>_results.jsonl\n"
            "under the results root when --output is omitted. "
            "Use --api_name to label the API in the filename."
        )
    )
    parser.add_argument("--input",       required=True, help="PRMBench JSONL input file")
    parser.add_argument("--output",      default=None,
                        help="Output JSONL path. Relative paths resolve under the results "
                             "root; if omitted, auto-generated in prmbench/results/")
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
        n_label = str(args.max_samples) if args.max_samples else "all"
        # Derive api_name from a short hash of base_url if not given
        api_label = args.api_name
        if not api_label:
            url = (args.base_url or "").strip().lower()
            api_label = "api_" + hashlib.md5(url.encode("utf-8")).hexdigest()[:6] if url else "api"
        model_safe = args.model.replace("/", "-").replace(":", "-")
        args.output = f"prmbench/results/prmbench_{n_label}_{model_safe}_{api_label}_results.jsonl"
    args.output = str(resolve_output(args.output))
    logger.info("Output path: %s", args.output)

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
