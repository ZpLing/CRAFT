#!/usr/bin/env python3
"""
baseline_comparisons.py
-----------------------
Baseline Comparison Study — Settings F through K.

Six published reasoning baselines that require NO pipeline trace files.
Evaluated on the same held-out test set, same model (gpt-4.1-mini).

Settings:
  F: Self-Consistency              (Wang et al., 2022)           — k × generation → majority vote
  G: Universal Self-Consistency    (Chen et al., 2023)          — k × generation → LLM selects best
  H: Self-Refine                  (Madaan et al., NeurIPS 2023) — generate → critique → revise
  I: LLM Self-Aggregation         (Li et al., 2025)             — k × generation → LLM synthesises
  J: Self-Eval Beam Search        (Xie et al., NeurIPS 2023)   — per-step self-score + beam search
  K: Faithful CoT + Symbolic      (Lyu et al., IJCNLP-AACL 2023) — translate to symbols → solve

Usage:
    # Quick test on 10 samples per dataset
    python baseline_comparisons.py \\
        --datasets ../FLD.json ../FOLIO.json \\
        --settings F G H I J K \\
        --shots 3 \\
        --max_samples 10 \\
        --model gpt-4.1-mini \\
        --output baseline_results.json

    # Full run
    python baseline_comparisons.py \\
        --datasets ../FLD.json ../FOLIO.json \\
        --shots 3 \\
        --model gpt-4.1-mini \\
        --output baseline_results.json
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np

# ---------------------------------------------------------------------------
# Config — read ROOT config.py WITHOUT using env vars to avoid shell overrides
# The root config.py uses os.getenv which picks up wrong MiniMax env vars.
# We parse the file directly for the literal default values instead.
# ---------------------------------------------------------------------------
_ROOT_CFG = Path(__file__).resolve().parent.parent / "config.py"
if _ROOT_CFG.exists():
    content = _ROOT_CFG.read_text()
    # Extract hardcoded defaults from the ACTIVE (uncommented) block only
    # Pattern: line must NOT start with # (after optional whitespace)
    m = re.search(r'^(?!.*#.*OPENAI_BASE_URL)\s*OPENAI_BASE_URL\s*=\s*os\.getenv\([^,]+,\s*"([^"]+)"\)', content, re.MULTILINE)
    m2 = re.search(r'^(?!.*#.*DEFAULT_MODEL)\s*DEFAULT_MODEL\s*=\s*os\.getenv\([^,]+,\s*"([^"]+)"\)', content, re.MULTILINE)
    OPENAI_BASE_URL = m.group(1)  if m  else "https://api.nuwaapi.com/v1"
    DEFAULT_MODEL   = m2.group(1) if m2 else "gemini-2.5-flash-lite"
else:
    OPENAI_BASE_URL = "https://api.nuwaapi.com/v1"
    DEFAULT_MODEL   = "gemini-2.5-flash-lite"
_m3 = re.search(r'^\s*OPENAI_API_KEY\s*=\s*os\.getenv\([^,]+,\s*"([^"]+)"\)', content, re.MULTILINE)
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY") or (_m3.group(1) if _m3 else "")
REQUEST_TIMEOUT = 180

# All baseline runs write to one place: <repo>/Part2_CRAFT/results/baseline_results/
BASELINE_RESULTS_DIR = (
    Path(__file__).resolve().parent.parent / "Part2_CRAFT" / "results" / "baseline_results"
)


def model_slug(model) -> str:
    """Directory name for one backbone: its id, lowercased and path-safe."""
    slug = re.sub(r"[^a-z0-9._-]+", "-", str(model or "").strip().lower()).strip("-.")
    return slug or "unknown-model"


def default_output(name: str, model: str) -> str:
    """Default --output for a baseline: one directory per run.

    A run belongs to the model that produced it, as in Part 1's results/<model>/,
    and each baseline then gets a directory of its own, so its data and its
    intermediate results sit together and the directory name says which baseline
    they came from. name="tree_of_thought", model="o4-mini" ->

        Part2_CRAFT/results/baseline_results/o4-mini/tree_of_thought/results.json
                                                                    /traces.jsonl
    """
    return str(BASELINE_RESULTS_DIR / model_slug(model) / name / "results.json")


# Whether a prediction is the gold answer is decided in one place, shared with
# Part 2's own scorer, so a baseline and CRAFT are measured by the same rule.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "Part2_CRAFT" / "evaluation" / "label_prediction"))
from answer_match import answers_match  # noqa: E402
# Steps are counted by the same rule for every system, CRAFT included.
from step_count import count_steps, count_tokens, basis as step_basis  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Run artifacts
# ---------------------------------------------------------------------------
# A run costs hours of wall clock and real tokens, and until now it kept only the
# extracted label: nine of the thirteen baselines dropped every generation on the
# floor, so a question asked afterwards — why did the vote split, what did the
# beam discard, which sub-problem did DICE get wrong — could only be answered by
# paying for the run again. Each runner now hangs its generations and the state
# that produced the prediction off the prediction itself, under ARTIFACTS_KEY.
#
# They are written to their own file rather than into the results JSON: the
# traces are two orders of magnitude larger than the predictions, and the
# results file has to stay small enough to load whole.
ARTIFACTS_KEY = "artifacts"


def resume_filter(samples: List[Dict], output, enabled: bool = True):
    """Split samples into the ones still to run and the results already held.

    A run of this size is not done in one sitting: 50 samples per dataset go
    first to check the setup, then the remaining 450 follow, and a run that dies
    halfway should cost only what it had not yet done. The unit of resumption is
    the sample id, taken from the traces file, which carries one line per sample
    whether or not that sample produced an answer — so a sample that failed is
    recorded as attempted rather than retried forever.

    Sample ids are stable across --per_dataset: loading 50 per dataset yields a
    subset of loading 500, with the same ids, as long as the seed and the list of
    datasets are unchanged. Change either and the ids no longer line up, so a
    resumed run must repeat both.
    """
    op = Path(output)
    traces = (op.with_name("traces.jsonl") if op.name == "results.json"
              else op.with_suffix(".traces.jsonl"))
    if not enabled or not op.exists():
        return samples, []

    done_ids = set()
    if traces.exists():
        with open(traces, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["sample_id"])
                except Exception:
                    continue
    try:
        previous = json.load(open(op, encoding="utf-8")).get("predictions", [])
    except Exception:
        previous = []
    # Keep only the predictions whose traces are also on disk, so the two files
    # never disagree about what the run contains.
    previous = [p for p in previous if p.get("sample_id") in done_ids]
    kept_ids = {p["sample_id"] for p in previous}
    todo = [s for s in samples if s["sample_id"] not in kept_ids]
    logger.info("Resuming: %d samples already done, %d to run", len(kept_ids), len(todo))
    return todo, previous


async def by_domain(runner, samples: List[Dict], *args, **kwargs) -> List[Dict]:
    """Run a baseline once per domain present, then reassemble in input order.

    Every runner reads the domain once, from the first sample, and applies it to
    the whole batch: the system prompt, the answer format it demands and the
    parser it extracts with all follow from that one read. Handing it FLD and
    GSM8K in the same call therefore asks the model for __PROVED__/__DISPROVED__
    on arithmetic, and the math datasets score zero without anything looking
    wrong — 101 of 104 math predictions in a four-dataset smoke run came back
    label-shaped.

    Splitting here rather than in each of the thirteen runners keeps every runner
    receiving what it already assumes: a batch of one domain.
    """
    groups: Dict[str, List[Dict]] = {}
    for s in samples:
        groups.setdefault(s.get("domain", "logical"), []).append(s)

    if len(groups) <= 1:
        preds = await runner(samples, *args, **kwargs)
    else:
        by_id: Dict[str, Dict] = {}
        for dom, group in groups.items():
            logger.info("Running %d %s samples", len(group), dom)
            for pred in await runner(group, *args, **kwargs):
                by_id[pred["sample_id"]] = pred
        preds = [by_id[s["sample_id"]] for s in samples if s["sample_id"] in by_id]
    return mark_blocked(preds, samples)


def mark_blocked(predictions: List[Dict], samples: List[Dict]) -> List[Dict]:
    """Flag the samples the gateway refused, so scoring can leave them out.

    A refusal is recognised by the problem text appearing in a prompt the
    gateway rejected. Matching on the text rather than on an id is what lets
    this live in one place: no runner has to thread a sample id through its
    calls, and a sample refused in any one of its calls is flagged.
    """
    if not BLOCKED_PROMPTS:
        return predictions
    text_by_id = {s["sample_id"]: s.get("problem_text", "") for s in samples}
    blocked_ids = set()
    for sid, text in text_by_id.items():
        if text and any(text in prompt for prompt in BLOCKED_PROMPTS):
            blocked_ids.add(sid)
    if blocked_ids:
        logger.warning("Gateway refused %d samples; they are excluded from the "
                       "denominator rather than counted wrong", len(blocked_ids))
    for p in predictions:
        if p.get("sample_id") in blocked_ids:
            p["gateway_blocked"] = True
    return predictions


def attach(prediction: Dict, **artifacts) -> Dict:
    """Record what produced this prediction: raw generations and intermediate state."""
    prediction[ARTIFACTS_KEY] = artifacts
    return prediction


def save_run(output, label: str, metrics: Dict, predictions: List[Dict],
             setting_key: str, append_traces: bool = False, **meta) -> Tuple[Path, Path]:
    """Write the two files a baseline run produces, both under
    Part2_CRAFT/results/baseline_results/<model>/.

        <model>/<baseline>/results.json   the data: the run's metrics, overall
                              and per dataset, and one record per sample with
                              the answer it predicted
        <model>/<baseline>/traces.jsonl   the intermediate results: every
                              generation this baseline made for the sample and
                              the state it derived the answer from

    They are two files rather than one because the traces are two orders of
    magnitude larger than the predictions, and the results file has to stay small
    enough to load whole. The traces file is newline-delimited so it can be
    streamed instead, and carries a line per sample even when that sample
    produced nothing — a missing line means a missing sample, not a silent one.
    """
    op = Path(output)
    op.parent.mkdir(parents=True, exist_ok=True)
    # The run's directory names the baseline, so the two files inside it do not
    # have to. An --output of another shape still gets its traces beside it.
    if op.name == "results.json":
        traces_path = op.with_name("traces.jsonl")
    elif op.name.endswith("_results.json"):
        traces_path = op.with_name(op.name.replace("_results.json", "_traces.jsonl"))
    else:
        traces_path = op.with_suffix(".traces.jsonl")

    # Append: a resumed run adds its samples to the ones already recorded. Only
    # the predictions made this time carry artifacts, so only those are written,
    # and the lines already on disk are left alone.
    fresh = [p for p in predictions if p.get(ARTIFACTS_KEY)]
    mode = "a" if (append_traces and traces_path.exists()) else "w"
    n_art = 0
    with open(traces_path, mode, encoding="utf-8") as tf:
        for pred in fresh:
            art = pred.get(ARTIFACTS_KEY)
            if art:
                n_art += 1
            tf.write(json.dumps({
                "sample_id":      pred.get("sample_id"),
                "source_dataset": pred.get("source_dataset"),
                "domain":         pred.get("domain"),
                "ground_truth":   pred.get("ground_truth"),
                "predicted":      pred.get("predicted"),
                "setting":        setting_key,
                **(art or {}),
            }, ensure_ascii=False) + "\n")

    slim = [{k: v for k, v in p.items() if k != ARTIFACTS_KEY} for p in predictions]
    with open(op, "w", encoding="utf-8") as f:
        json.dump({"setting": label, "setting_key": setting_key,
                   "overall": metrics, "run": meta,
                   "traces_file": traces_path.name,
                   "predictions": slim}, f, ensure_ascii=False, indent=1)

    logger.info("Saved data:          %s  (%d samples)", op, len(slim))
    logger.info("Saved intermediates: %s  (%d new this run, mode=%s)",
                traces_path, n_art, mode)
    return op, traces_path


VALID_LABELS  = {"__PROVED__", "__DISPROVED__"}
BINARY_LABELS = ["__PROVED__", "__DISPROVED__"]
_LABEL_RE     = re.compile(r"(__PROVED__|__DISPROVED__)", re.IGNORECASE)
def _extract_boxed_content(text: str) -> list:
    """Extract \\boxed{...} contents handling nested braces."""
    results = []
    i = 0
    while i < len(text):
        idx = text.find('\\boxed{', i)
        if idx == -1:
            break
        start = idx + 7
        depth, j = 1, start
        while j < len(text) and depth > 0:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1
        if depth == 0:
            results.append(text[start:j - 1].strip())
        i = j
    return results
_HASH_ANS_RE  = re.compile(r"####\s*(.+)")
_FINAL_ANS_RE = re.compile(
    r"(?:the\s+answer\s+is|final\s+answer\s*[:\=]|answer\s*[:\=])\s*([^\n\.]+)",
    re.IGNORECASE,
)

ALL_SETTINGS = ["A", "F", "G", "H", "I", "J", "K"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalise_label(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    label = label.strip().upper()
    if label in VALID_LABELS:
        return label
    if "DISPROVED" in label:
        return "__DISPROVED__"
    if "PROVED" in label:
        return "__PROVED__"
    return None


def extract_label(text: str) -> Optional[str]:
    # Primary: exact __PROVED__ or __DISPROVED__ format
    matches = _LABEL_RE.findall(text or "")
    if matches:
        m = matches[-1].upper()
        return m if m in ("__PROVED__", "__DISPROVED__") else None
    # Secondary: markdown bold without underscores: **PROVED**, **DISPROVED**
    bold_matches = re.findall(r"\*\*(PROVED|DISPROVED)\*\*", text or "", re.I)
    if bold_matches:
        b = bold_matches[-1].upper()
        return {"PROVED": "__PROVED__", "DISPROVED": "__DISPROVED__"}.get(b)
    # Fallback: search last 3 lines for label keywords
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    tail = " ".join(lines[-3:]) if len(lines) >= 3 else " ".join(lines)
    if re.search(r"\bdisproved\b|\bfalse\b|\brefuted\b", tail, re.I):
        return "__DISPROVED__"
    if re.search(r"\bproved\b|\btrue\b|\bholds\b", tail, re.I):
        return "__PROVED__"
    return None


def extract_math_answer(text: str) -> Optional[str]:
    """Extract numeric answer: \\boxed{} → #### N → bare number.
    Commas are preserved (multi-answer support for OlympiadBench).
    """
    if not text:
        return None
    m = _extract_boxed_content(text)
    if m:
        raw = m[-1].strip()
        raw = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', raw)  # thousands sep only
        return raw
    m2 = _HASH_ANS_RE.findall(text)
    if m2:
        return m2[-1].strip()
    m3 = _FINAL_ANS_RE.findall(text)
    if m3:
        return m3[-1].strip()
    # Bare-number / short-answer fallback (USC selector, bare responses, OlympiadBench format)
    # Try the whole text and the last line; use _normalise_single to handle $42$, 18.8°, \frac etc.
    for candidate in [text.strip(), text.strip().splitlines()[-1].strip() if text.strip() else ""]:
        if candidate and len(candidate) <= 60:
            norm = _normalise_single(candidate)
            if norm is not None:
                return candidate   # return raw; normalise_math_answer will process it
    return None


def _eval_latex_frac(text: str) -> str:
    """Evaluate \\frac{a}{b} to a decimal string."""
    def _replace(m: re.Match) -> str:
        try:
            n, d = float(m.group(1).strip()), float(m.group(2).strip())
            if d == 0:
                return m.group(0)
            r = n / d
            return str(int(r)) if r == int(r) else str(round(r, 8)).rstrip("0").rstrip(".")
        except (ValueError, TypeError):
            return m.group(0)
    return re.sub(r'\\(?:[tdc])?frac\{([^}]+)\}\{([^}]+)\}', _replace, text)


def _normalise_single(s: str) -> Optional[str]:
    """Normalize one numeric/expression token."""
    s = s.strip().strip("$").strip()
    if not s:
        return None
    s = _eval_latex_frac(s)
    s = re.sub(r'\^?\{?\\circ\}?|°|\\degree', '', s)
    s = re.sub(r'\\(?:text|mathrm|mbox)\{[^}]*\}', '', s)
    s = re.sub(
        r'\s*\b(?:km|cm|mm|m|kg|g|mg|l|ml|hours?|hrs?|minutes?|mins?|seconds?|secs?|'
        r'days?|weeks?|months?|years?|dollars?|cents?|degrees?)\b.*$',
        '', s, flags=re.IGNORECASE,
    )
    s = re.sub(r'\\[a-zA-Z]+\*?', '', s)
    s = re.sub(r'[{}]', '', s)
    s = s.strip().lstrip("$\\")
    if s.endswith('%'):
        s = s[:-1]
    s = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', s)
    s = s.strip().lower()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, OverflowError):
        pass
    # Evaluate plain a/b fractions (e.g. "505/8076" → decimal)
    _frac_m = re.match(r'^(-?\d+\.?\d*)\s*/\s*(-?\d+\.?\d*)$', s)
    if _frac_m:
        try:
            _n, _d = float(_frac_m.group(1)), float(_frac_m.group(2))
            if _d != 0:
                _r = _n / _d
                return str(int(_r)) if _r == int(_r) else str(round(_r, 8)).rstrip("0").rstrip(".")
        except (ValueError, ZeroDivisionError):
            pass
    # Only return non-numeric strings that look like math expressions.
    # Reject English prose (e.g. step titles like "step 8: state the result.")
    if s and not re.search(r'[a-z]{4,}', s):
        return s
    return None


def normalise_math_answer(ans: Optional[str]) -> Optional[str]:
    """Normalize a math answer.
    Handles GSM8K (plain integers), OlympiadBench (fractions, degrees,
    multi-answer tuples), and general LaTeX formatting.
    """
    if ans is None:
        return None
    ans = str(ans).strip()
    if ans.startswith('$') and ans.endswith('$') and len(ans) > 2:
        inner = ans[1:-1].strip()
        if not inner.startswith('$'):
            ans = inner
    # Strip outer parentheses or brackets enclosing a tuple: (1,2,3) → 1,2,3
    if ((ans.startswith('(') and ans.endswith(')')) or
            (ans.startswith('[') and ans.endswith(']'))):
        inner = ans[1:-1].strip()
        if inner:
            ans = inner
    if ',' in ans:
        parts = [p.strip() for p in ans.split(',') if p.strip()]
        if len(parts) >= 2:
            norm_parts = [_normalise_single(p) for p in parts]
            if all(p is not None for p in norm_parts):
                try:
                    sorted_parts = sorted(norm_parts, key=float)
                except (ValueError, TypeError):
                    sorted_parts = sorted(norm_parts)
                return ','.join(sorted_parts)
    return _normalise_single(ans)


def _legacy_count_tokens(text: str) -> int:
    return len(text.split()) if text else 0


def _extract_pred(text: str, domain: str = "logical") -> Optional[str]:
    if domain == "math":
        return normalise_math_answer(extract_math_answer(text))
    return extract_label(text)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_raw_dataset(paths: List[Path], seed: int = 42, per_dataset: int = 250,
                    max_samples: Optional[int] = None) -> List[Dict]:
    rng = random.Random(seed)
    n_per_class = per_dataset // 2
    all_samples = []

    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else data.get("results", [])
        stem  = path.stem

        first = items[0] if items else {}
        is_math = (first.get("domain") == "math" or
                   ("answer" in first and "proof_label" not in first and "Label" not in first))

        if is_math:
            # Detect OlympiadBench by presence of answer_type field
            is_olympiad = 'answer_type' in first
            rng_items = list(items)
            rng.shuffle(rng_items)
            limit = min(len(rng_items), per_dataset * 4 if is_olympiad else per_dataset * 2)
            chosen = []
            for idx, item in enumerate(rng_items[:limit]):
                problem_text = (item.get("input") or item.get("question", "")).strip()
                atype = None
                if is_olympiad:
                    # Proof-type items (answer_type=None) have no answer to score
                    # against and are still skipped. Expression, Tuple and
                    # Interval answers are kept: answer_match compares them as
                    # what they are, which the old string comparison could not.
                    atype = item.get('answer_type')
                    if not atype:
                        continue
                    # Use answer field directly, never fall back to long solution text
                    raw_gt = item.get('answer', '').strip()
                else:
                    raw_gt = (item.get("answer") or "").strip()
                # The gold answer is carried as written. Normalising it here is
                # what lost information before: the comparison is done by
                # answer_match, which needs the original text.
                gt = raw_gt if is_olympiad else normalise_math_answer(raw_gt)
                if not problem_text or not gt:
                    continue
                chosen.append({
                    "sample_id":      f"{stem}_{idx}",
                    "source_dataset": stem,
                    "problem_text":   problem_text,
                    "ground_truth":   gt,
                    "answer_type":    atype,
                    "domain":         "math",
                })
                if len(chosen) >= per_dataset:
                    break
            logger.info("%s (math%s): %d samples loaded",
                        stem, "/olympiad" if is_olympiad else "", len(chosen))
            all_samples.extend(chosen)
        else:
            proved, disproved = [], []
            for idx, item in enumerate(items):
                gt = normalise_label(
                    item.get("proof_label") or item.get("Label") or item.get("label")
                )
                problem_text = item.get("input", "").strip()
                if not gt or not problem_text:
                    continue
                entry = {
                    "sample_id":      f"{stem}_{idx}",
                    "source_dataset": stem,
                    "problem_text":   problem_text,
                    "ground_truth":   gt,
                    "domain":         "logical",
                }
                (proved if gt == "__PROVED__" else disproved).append(entry)

            rng.shuffle(proved)
            rng.shuffle(disproved)
            chosen = proved[:n_per_class] + disproved[:n_per_class]
            logger.info("%s (logical): PROVED=%d, DISPROVED=%d, total=%d",
                        stem, len(proved), len(disproved), len(chosen))
            all_samples.extend(chosen)

    if max_samples:
        # For math datasets: just truncate (no label balancing needed)
        # For logical datasets: ensure strict PROVED/DISPROVED balance
        has_math = any(s.get("domain") == "math" for s in all_samples)
        has_logical = any(s.get("domain") == "logical" for s in all_samples)
        if has_math and not has_logical:
            rng.shuffle(all_samples)
            all_samples = all_samples[:max_samples]
            logger.info("Capped to --max_samples=%d (math, random truncation)", max_samples)
        else:
            half = max_samples // 2
            proved_list = [s for s in all_samples if s["ground_truth"] == "__PROVED__"]
            disproved_list = [s for s in all_samples if s["ground_truth"] == "__DISPROVED__"]
            all_samples = proved_list[:half] + disproved_list[:half]
            rng.shuffle(all_samples)
            logger.info("Capped to --max_samples=%d (strict %d PROVED + %d DISPR)", max_samples, half, half)
    return all_samples


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_MATH = (
    "You are an expert mathematician. Solve the math problem step by step. "
    "Show all calculations clearly. "
    "On the very last line write ONLY the numeric answer in the format: \\boxed{<answer>}"
)
SYSTEM_LOGICIAN = (
    "You are an expert logician. Given premises and a hypothesis, decide whether "
    "the hypothesis is __PROVED__ or __DISPROVED__ based solely on the premises. "
    "There are exactly TWO possible answers: __PROVED__ or __DISPROVED__. "
    "Output your step-by-step reasoning, then on the very last line write ONLY: "
    "__PROVED__ or __DISPROVED__."
)

def build_zeroshot_prompt(problem: str, domain: str = "logical") -> str:
    if domain == "math":
        return f"Problem:\n{problem}\n\nSolve step by step. Last line must be: \\boxed{{<numeric answer>}}"
    return f"Problem:\n{problem}\n\nOutput ONLY __PROVED__ or __DISPROVED__ on the last line (two choices only)."


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

# The API gateway refuses some requests before the model sees them — a content
# filter that trips on words like "gcd" or "benzodiazepine" appearing in a
# problem. The model never got to answer, so such a sample is not evidence about
# the model and does not belong in the denominator. The refused prompts are
# recorded here and the samples carrying them are marked, so they can be
# excluded from the accuracy rather than counted as failures.
BLOCKED_PROMPTS: set = set()
_BLOCK_MARKERS = ("sensitive word", "content filter", "content_filter",
                  "invalid input", "safety")


def _is_gateway_refusal(body: str) -> bool:
    low = (body or "").lower()
    return any(m in low for m in _BLOCK_MARKERS)


async def call_llm(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    system: str,
    user: str,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> Optional[str]:
    # o4-mini / deepseek-r1: reasoning tokens eat into max_tokens budget, leaving content empty.
    # Use a large max_tokens for the total budget, and add max_output_tokens to reserve space
    # for the visible response (Bosch API specific parameter, mirrors OpenAI SDK extra_body usage).
    is_reasoning_model = model in ("o4-mini", "deepseek-r1", "o4-mini-2025-04-16")
    if is_reasoning_model:
        max_tokens = max(max_tokens, 16000)

    url     = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model":    model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user",   "content": user}],
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    if is_reasoning_model:
        payload["max_output_tokens"] = min(max_tokens, 4096)
    for attempt in range(5):
        async with semaphore:
            try:
                async with session.post(
                    url, headers=headers, json=payload,
                    timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        msg = data["choices"][0]["message"]
                        # o4-mini / deepseek-r1 put reasoning in reasoning_content, not content
                        if model in ("o4-mini", "deepseek-r1", "o4-mini-2025-04-16"):
                            text = msg.get("reasoning_content") or msg.get("content") or ""
                        else:
                            text = msg.get("content") or ""
                        text = text.strip()
                        # Retry on empty content — Bosch proxy sometimes returns empty body under load
                        if not text:
                            logger.warning("Attempt %d: empty response, retrying", attempt + 1)
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return text
                    body = await resp.text()
                    logger.warning("HTTP %s: %s", resp.status, body[:200])
                    # 400 = client error (content filter, invalid request) — don't retry
                    if resp.status == 400:
                        if _is_gateway_refusal(body):
                            BLOCKED_PROMPTS.add(user)
                        return None
            except Exception as e:
                logger.warning("Attempt %d: %s", attempt + 1, e)
        await asyncio.sleep(2 ** attempt)
    return None


# ===========================================================================
# SETTING A: Zero-shot (single call)
# ===========================================================================

async def run_setting_zeroshot(
    test_samples: List[Dict],
    model: str, api_key: str, base_url: str,
    semaphore: asyncio.Semaphore,
    session: aiohttp.ClientSession,
) -> List[Dict]:
    """A: single zero-shot call per sample, no trace files needed."""
    domain = test_samples[0].get("domain", "logical") if test_samples else "logical"
    system = SYSTEM_MATH if domain == "math" else SYSTEM_LOGICIAN
    coros = [
        call_llm(session, semaphore, system,
                 build_zeroshot_prompt(s["problem_text"], domain=domain),
                 model, api_key, base_url,
                 temperature=0.0, max_tokens=1024)
        for s in test_samples
    ]
    responses = await asyncio.gather(*coros)
    predictions = []
    for s, r in zip(test_samples, responses):
        pred = _extract_pred(r or "", domain)
        predictions.append(attach({
            "sample_id":       s["sample_id"],
            "source_dataset":  s["source_dataset"],
            "ground_truth":    s["ground_truth"],
            "predicted":       pred,
            "raw_response":    r or "",
            "response_steps":  count_steps(r or ""),
            "response_tokens": count_tokens(r or ""),
            "domain":          domain,
        }, traces=[r or ""]))
    return predictions


# ===========================================================================
# SETTING F: Self-Consistency (Wang et al., 2022)
# ===========================================================================

async def run_setting_self_consistency(
    test_samples: List[Dict],
    k: int,
    model: str, api_key: str, base_url: str,
    semaphore: asyncio.Semaphore,
    session: aiohttp.ClientSession,
) -> List[Dict]:
    """F: generate k responses → majority vote on label."""
    domain = test_samples[0].get("domain", "logical") if test_samples else "logical"
    system = SYSTEM_MATH if domain == "math" else SYSTEM_LOGICIAN
    all_coros = []
    for s in test_samples:
        user = build_zeroshot_prompt(s["problem_text"], domain=domain)
        for _ in range(k):
            all_coros.append(
                call_llm(session, semaphore, system,
                         user, model, api_key, base_url,
                         temperature=0.7, max_tokens=1024)
            )
    all_responses = await asyncio.gather(*all_coros)

    predictions = []
    for i, s in enumerate(test_samples):
        resps  = all_responses[i * k: (i + 1) * k]
        preds  = [_extract_pred(r, domain) for r in resps if r]
        valid  = [p for p in preds if p is not None]
        votes  = Counter(valid)
        pred   = votes.most_common(1)[0][0] if valid else None
        # The reported answer came from the traces that voted for it. Averaging
        # over all k, including the ones that were outvoted, measures the length
        # of the sampling rather than the length of the answer.
        winners = [r for r, lab in zip(resps, preds) if r and lab == pred] or [r for r in resps if r]
        predictions.append(attach({
            "sample_id":       s["sample_id"],
            "source_dataset":  s["source_dataset"],
            "ground_truth":    s["ground_truth"],
            "predicted":       pred,
            "all_labels":      preds,
            "response_steps":  float(np.mean([count_steps(r) for r in winners])) if winners else 0.0,
            "response_tokens": float(np.mean([count_tokens(r) for r in winners])) if winners else 0.0,
            "domain":          domain,
        }, traces=[r or "" for r in resps], labels=preds, votes=dict(votes)))
    return predictions


# ===========================================================================
# SETTING G: Universal Self-Consistency (Chen et al., 2023)
# ===========================================================================

_USC_SEL_LOGICAL = (
    "You are an expert logician. You will see multiple reasoning attempts. "
    "Select the single most consistent and correct answer. "
    "Output ONLY: __PROVED__ or __DISPROVED__"
)
_USC_SEL_MATH = (
    "You are an expert mathematician. You will see multiple solution attempts. "
    "Select the single most consistent and correct numeric answer. "
    "Output ONLY the number (e.g. 42 or 3.5), no other text."
)


async def run_setting_usc(
    test_samples: List[Dict],
    k: int,
    model: str, api_key: str, base_url: str,
    semaphore: asyncio.Semaphore,
    session: aiohttp.ClientSession,
) -> List[Dict]:
    """G: k × generation → LLM selector picks the best answer."""
    domain = test_samples[0].get("domain", "logical") if test_samples else "logical"
    system = SYSTEM_MATH if domain == "math" else SYSTEM_LOGICIAN
    sel_system = _USC_SEL_MATH if domain == "math" else _USC_SEL_LOGICAL

    gen_coros = []
    for s in test_samples:
        user = build_zeroshot_prompt(s["problem_text"], domain=domain)
        for _ in range(k):
            gen_coros.append(
                call_llm(session, semaphore, system,
                         user, model, api_key, base_url,
                         temperature=0.7, max_tokens=1024)
            )
    all_gen = await asyncio.gather(*gen_coros)

    sel_coros, grouped = [], []
    for i, s in enumerate(test_samples):
        resps = all_gen[i * k: (i + 1) * k]
        grouped.append(resps)
        block = "\n\n".join(
            f"Candidate {j+1}:\n{r[:600]}" for j, r in enumerate(resps) if r
        )
        if domain == "math":
            sel_user = (f"Problem:\n{s['problem_text']}\n\n"
                        f"Solution attempts:\n{block}\n\n"
                        f"What is the most consistent correct numeric answer? Output ONLY the number:")
        else:
            sel_user = (f"Problem:\n{s['problem_text']}\n\n"
                        f"Reasoning attempts:\n{block}\n\n"
                        f"Based on the above candidates, what is the most consistent and correct answer? "
                        f"Output ONLY: __PROVED__ or __DISPROVED__")
        sel_coros.append(
            call_llm(session, semaphore, sel_system,
                     sel_user, model, api_key, base_url,
                     temperature=0.0, max_tokens=2048)
        )
    sel_responses = await asyncio.gather(*sel_coros)

    predictions = []
    for s, resps, sel_r in zip(test_samples, grouped, sel_responses):
        _sel_pred = _extract_pred(sel_r, domain) if sel_r else None
        # The selector returns only an answer, so the trace behind it is the
        # candidate that argued for that answer.
        _selected = next((r for r in resps
                          if r and _extract_pred(r, domain) == _sel_pred), sel_r or "")
        predictions.append(attach({
            "sample_id":       s["sample_id"],
            "source_dataset":  s["source_dataset"],
            "ground_truth":    s["ground_truth"],
            "predicted":       _extract_pred(sel_r, domain) if sel_r else None,
            "selector_response": sel_r or "",
            "response_steps":  count_steps(_selected),
            "response_tokens": count_tokens(_selected),
            "domain":          domain,
        }, traces=[r or "" for r in resps],
           candidate_labels=[_extract_pred(r or "", domain) for r in resps],
           selector_response=sel_r or ""))
    return predictions


# ===========================================================================
# SETTING H: Self-Refine (Madaan et al., NeurIPS 2023)
# ===========================================================================

_REFINE_CRITIC = (
    "You are a strict reasoning critic. "
    "Review the reasoning below and identify any errors, unsupported leaps, or incorrect conclusions. "
    "Be specific and concise. If the reasoning is correct, say 'The reasoning is correct.'"
)
_REFINE_REVISE_LOGICAL = (
    "You are an expert logician. Revise the reasoning chain based on the critique provided. "
    "Produce an improved reasoning chain that fixes all identified errors. "
    "On the very last line write ONLY: __PROVED__ or __DISPROVED__."
)
_REFINE_REVISE_MATH = (
    "You are an expert mathematician. Revise the solution based on the critique provided. "
    "Show all corrected calculations step by step. "
    "On the very last line write ONLY: \\boxed{<numeric answer>}"
)


async def run_setting_self_refine(
    test_samples: List[Dict],
    model: str, api_key: str, base_url: str,
    semaphore: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    n_iterations: int = 1,
) -> List[Dict]:
    """H: generate → critique → revise (n_iterations rounds)."""
    domain  = test_samples[0].get("domain", "logical") if test_samples else "logical"
    system  = SYSTEM_MATH if domain == "math" else SYSTEM_LOGICIAN
    rev_sys = _REFINE_REVISE_MATH if domain == "math" else _REFINE_REVISE_LOGICAL

    init_coros = [
        call_llm(session, semaphore, system,
                 build_zeroshot_prompt(s["problem_text"], domain=domain),
                 model, api_key, base_url, temperature=0.0, max_tokens=1024)
        for s in test_samples
    ]
    current = list(await asyncio.gather(*init_coros))
    # The refinement is the point of this baseline, so every draft is kept, not
    # just the last one: traces[i] is the draft after round i, and critiques[i]
    # is what prompted the revision into traces[i+1]. Keeping the critique text
    # out of traces avoids storing every draft twice.
    drafts: List[List[str]] = [[c or ""] for c in current]
    critiques_by_sample: List[List[Dict]] = [[] for _ in current]

    for _it in range(n_iterations):
        crit_coros = []
        for s, resp in zip(test_samples, current):
            crit_user = (
                f"Problem:\n{s['problem_text']}\n\n"
                f"Reasoning:\n{resp or '(empty)'}\n\n"
                f"Identify any errors or unsupported steps:")
            crit_coros.append(
                call_llm(session, semaphore, _REFINE_CRITIC,
                         crit_user, model, api_key, base_url,
                         temperature=0.0, max_tokens=256)
            )
        critiques = await asyncio.gather(*crit_coros)

        revise_coros = []
        for s, resp, crit in zip(test_samples, current, critiques):
            hint = ("Last line must be ONLY: \\boxed{<numeric answer>}"
                    if domain == "math" else "Last line must be ONLY: __PROVED__ or __DISPROVED__")
            rev_user = (
                f"Problem:\n{s['problem_text']}\n\n"
                f"Original reasoning:\n{resp or '(empty)'}\n\n"
                f"Critique:\n{crit or '(none)'}\n\n"
                f"Revise to fix all issues. {hint}")
            revise_coros.append(
                call_llm(session, semaphore, rev_sys,
                         rev_user, model, api_key, base_url,
                         temperature=0.0, max_tokens=1024)
            )
        current = list(await asyncio.gather(*revise_coros))
        for ds, cs, crit, rev in zip(drafts, critiques_by_sample, critiques, current):
            cs.append({"round": _it + 1, "critique": crit or ""})
            ds.append(rev or "")

    predictions = []
    for s, r, ds, cs in zip(test_samples, current, drafts, critiques_by_sample):
        predictions.append(attach({
            "sample_id":       s["sample_id"],
            "source_dataset":  s["source_dataset"],
            "ground_truth":    s["ground_truth"],
            "predicted":      _extract_pred(r, domain),
            "raw_response":   r or "",
            "response_steps": count_steps(r or ""),
            "response_tokens": count_tokens(r or ""),
            "domain":         domain,
        }, traces=ds, critiques=cs,
           labels_by_round=[_extract_pred(d, domain) for d in ds]))
    return predictions


# ===========================================================================
# SETTING I: LLM Self-Aggregation (Li et al., 2025)
# ===========================================================================

_AGGREGATE_LOGICAL = (
    "You are an expert logician. You have seen multiple independent reasoning attempts "
    "for the same problem. Synthesize the best elements into a single coherent chain. "
    "On the very last line write ONLY: __PROVED__ or __DISPROVED__."
)
_AGGREGATE_MATH = (
    "You are an expert mathematician. You have seen multiple solution attempts. "
    "Synthesize the correct approach into a single coherent solution. "
    "On the very last line write ONLY: \\boxed{<numeric answer>}"
)


async def run_setting_self_aggregation(
    test_samples: List[Dict],
    k: int,
    model: str, api_key: str, base_url: str,
    semaphore: asyncio.Semaphore,
    session: aiohttp.ClientSession,
) -> List[Dict]:
    """I: k × generation → LLM synthesises them into one best answer."""
    domain     = test_samples[0].get("domain", "logical") if test_samples else "logical"
    system     = SYSTEM_MATH if domain == "math" else SYSTEM_LOGICIAN
    agg_system = _AGGREGATE_MATH if domain == "math" else _AGGREGATE_LOGICAL

    gen_coros = []
    for s in test_samples:
        for _ in range(k):
            gen_coros.append(
                call_llm(session, semaphore, system,
                         build_zeroshot_prompt(s["problem_text"], domain=domain),
                         model, api_key, base_url,
                         temperature=0.7, max_tokens=1024)
            )
    all_gen = await asyncio.gather(*gen_coros)

    agg_coros, grouped = [], []
    for i, s in enumerate(test_samples):
        resps = all_gen[i * k: (i + 1) * k]
        grouped.append(resps)
        block = "\n\n".join(
            f"Response {j+1}:\n{r[:600]}" for j, r in enumerate(resps) if r
        )
        hint = ("Last line must be ONLY: \\boxed{<numeric answer>}"
                if domain == "math" else "Last line must be ONLY: __PROVED__ or __DISPROVED__")
        agg_user = (f"Problem:\n{s['problem_text']}\n\n"
                    f"Multiple independent attempts:\n{block}\n\n"
                    f"Synthesize into the best possible solution. {hint}")
        agg_coros.append(
            call_llm(session, semaphore, agg_system,
                     agg_user, model, api_key, base_url,
                     temperature=0.0, max_tokens=1024)
        )
    agg_responses = await asyncio.gather(*agg_coros)

    predictions = []
    for s, resps, agg_r in zip(test_samples, grouped, agg_responses):
        predictions.append(attach({
            "sample_id":       s["sample_id"],
            "source_dataset":  s["source_dataset"],
            "ground_truth":    s["ground_truth"],
            "predicted":      _extract_pred(agg_r, domain),
            "raw_response":   agg_r or "",
            "response_steps": count_steps(agg_r or ""),
            "response_tokens": count_tokens(agg_r or ""),
            "domain":         domain,
        }, traces=[r or "" for r in resps],
           candidate_labels=[_extract_pred(r or "", domain) for r in resps],
           aggregated=agg_r or ""))
    return predictions


# ===========================================================================
# SETTING J: Self-Eval Beam Search (Xie et al., NeurIPS 2023)
# ===========================================================================

_SEBS_GEN_LOGICAL = (
    "You are an expert logician reasoning step by step. "
    "Continue the reasoning chain by writing ONLY the next single reasoning step. "
    "If you have reached a definitive conclusion, write the conclusion step ending with "
    "__PROVED__ or __DISPROVED__. Do not write more than one step."
)
_SEBS_GEN_MATH = (
    "You are an expert mathematician solving a math problem step by step. "
    "Continue the solution by writing ONLY the next single calculation step. "
    "If you have computed the final answer, write the last step ending with \\boxed{<answer>}. "
    "Do not write more than one step."
)
_SEBS_SCR_LOGICAL = (
    "You are a logical reasoning evaluator. "
    "Given a problem and a partial reasoning chain, assess how likely this reasoning "
    "path leads to a correct conclusion. "
    "Output ONLY a number between 0.0 and 1.0 representing your confidence."
)
_SEBS_SCR_MATH = (
    "You are a mathematical reasoning evaluator. "
    "Given a math problem and a partial solution chain, assess how likely this solution "
    "path leads to the correct numeric answer. "
    "Output ONLY a number between 0.0 and 1.0 representing your confidence."
)


def _parse_score(text: str) -> float:
    m = re.search(r"\b(0?\.\d+|1\.0|0|1)\b", text or "")
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            pass
    return 0.5


async def run_setting_sebs(
    test_samples: List[Dict],
    model: str, api_key: str, base_url: str,
    semaphore: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    beam_width: int = 2,
    max_steps: int = 8,
) -> List[Dict]:
    """J: per-step generation + self-scoring + beam search."""
    domain = test_samples[0].get("domain", "logical") if test_samples else "logical"
    sebs_gen = _SEBS_GEN_MATH if domain == "math" else _SEBS_GEN_LOGICAL
    sebs_scr = _SEBS_SCR_MATH if domain == "math" else _SEBS_SCR_LOGICAL

    async def _beam_one(problem: str) -> Tuple[str, float, List[Dict]]:
        beams = [{"trace": "", "score": 1.0, "done": False}]
        # What the search saw at each round, including the candidates it dropped:
        # a beam baseline's answer is only interpretable next to what it rejected.
        history: List[Dict] = []

        for _round in range(max_steps):
            active = [b for b in beams if not b["done"]]
            if not active:
                break

            gen_coros = []
            for b in active:
                prev = b["trace"] if b["trace"] else "(start)"
                gen_coros.append(
                    call_llm(session, semaphore, sebs_gen,
                             f"Problem:\n{problem}\n\nReasoning so far:\n{prev}\n\nWrite the next reasoning step only:",
                             model, api_key, base_url,
                             temperature=0.7, max_tokens=200)
                )
            next_steps = await asyncio.gather(*gen_coros)

            score_coros, candidates = [], []
            for b, step in zip(active, next_steps):
                new_trace = (b["trace"] + "\n" + (step or "")).strip()
                candidates.append({"trace": new_trace, "step": step or "",
                                   "parent_score": b["score"]})
                score_coros.append(
                    call_llm(session, semaphore, sebs_scr,
                             f"Problem:\n{problem}\n\nPartial reasoning:\n{new_trace}\n\n"
                             f"Confidence this reasoning path is correct (0.0–1.0):",
                             model, api_key, base_url,
                             temperature=0.0, max_tokens=2048)
                )
            score_texts = await asyncio.gather(*score_coros)

            new_beams = []
            for cand, score_text in zip(candidates, score_texts):
                step_score  = _parse_score(score_text)
                combined    = 0.6 * step_score + 0.4 * cand["parent_score"]
                label_hit   = _extract_pred(cand["trace"], domain) is not None
                new_beams.append({"trace": cand["trace"], "score": combined, "done": label_hit})

            inactive = [b for b in beams if b["done"]]
            new_beams.sort(key=lambda x: -x["score"])
            history.append({
                "round": _round,
                "candidates": [{"step": c["step"], "score": round(b["score"], 4),
                                "done": b["done"]}
                               for c, b in zip(candidates, new_beams)],
                "kept": [round(b["score"], 4) for b in new_beams[:beam_width]],
            })
            beams = inactive + new_beams[:beam_width]

        best = max(beams, key=lambda x: x["score"])
        return best["trace"], best["score"], history

    tasks = [_beam_one(s["problem_text"]) for s in test_samples]
    results_bs = await asyncio.gather(*tasks)

    predictions = []
    for s, (trace, score, history) in zip(test_samples, results_bs):
        predictions.append(attach({
            "sample_id":       s["sample_id"],
            "source_dataset":  s["source_dataset"],
            "ground_truth":    s["ground_truth"],
            "predicted":      _extract_pred(trace, domain),
            "beam_score":     round(score, 4),
            "raw_response":  trace,
            "response_steps": count_steps(trace),
            "response_tokens": count_tokens(trace),
            "domain":         domain,
        }, traces=[trace], beam_history=history))
    return predictions


# ===========================================================================
# SETTING K: Faithful CoT + Symbolic (Lyu et al., IJCNLP-AACL 2023)
# ===========================================================================

_SYMB_LOGICAL = (
    "You are a logical reasoning expert. Analyze the following problem carefully.\n\n"
    "First, identify the key facts and rules.\n"
    "Then reason step by step to determine whether the hypothesis is proved or disproved.\n\n"
    "You MUST end your response with exactly one of these two lines:\n"
    "CONCLUSION: __PROVED__\n"
    "CONCLUSION: __DISPROVED__\n\n"
    "Always commit to PROVED or DISPROVED based on available evidence."
)
_SYMB_MATH = (
    "You are an expert mathematician. Solve the following math problem using "
    "step-by-step arithmetic reasoning, expressing each operation explicitly.\n\n"
    "Output format:\n"
    "STEPS:\n"
    "step1: <equation>\n"
    "step2: <equation>\n"
    "...\n\n"
    "ANSWER: <numeric value>\n\n"
    "The ANSWER line must contain only the final numeric answer."
)


def _run_symbolic_solver(llm_output: str, domain: str = "logical") -> Optional[str]:
    if domain == "math":
        m = re.search(r"ANSWER\s*[:\-]\s*([^\n]+)", llm_output or "", re.IGNORECASE)
        if m:
            return normalise_math_answer(m.group(1))
        return normalise_math_answer(extract_math_answer(llm_output))
    conclusion_match = re.search(
        r"CONCLUSION\s*[:\-]\s*(__PROVED__|__DISPROVED__)",
        llm_output or "", re.IGNORECASE
    )
    if conclusion_match:
        return normalise_label(conclusion_match.group(1))
    return extract_label(llm_output)


async def run_setting_faithful_cot(
    test_samples: List[Dict],
    model: str, api_key: str, base_url: str,
    semaphore: asyncio.Semaphore,
    session: aiohttp.ClientSession,
) -> List[Dict]:
    """K: translate to symbolic form → structured output → extract answer."""
    domain = test_samples[0].get("domain", "logical") if test_samples else "logical"
    sym_system = _SYMB_MATH if domain == "math" else _SYMB_LOGICAL

    coros = []
    for s in test_samples:
        if domain == "math":
            user = (f"Problem:\n{s['problem_text']}\n\n"
                    "Solve using explicit arithmetic steps. "
                    "Follow the output format with STEPS: and ANSWER: sections.")
        else:
            user = (f"Problem:\n{s['problem_text']}\n\n"
                    "Reason step by step to determine whether the hypothesis is PROVED or DISPROVED. "
                    "End with exactly: CONCLUSION: __PROVED__ or CONCLUSION: __DISPROVED__")
        coros.append(
            call_llm(session, semaphore, sym_system, user,
                     model, api_key, base_url,
                     temperature=0.0, max_tokens=2048)
        )
    responses = await asyncio.gather(*coros)

    predictions = []
    for s, r in zip(test_samples, responses):
        predictions.append(attach({
            "sample_id":       s["sample_id"],
            "source_dataset": s["source_dataset"],
            "ground_truth":   s["ground_truth"],
            "predicted":      _run_symbolic_solver(r or "", domain=domain),
            "raw_response":   r or "",
            "response_steps": count_steps(r or ""),
            "response_tokens": count_tokens(r or ""),
            "domain":         domain,
        }, traces=[r or ""], solver_output=_run_symbolic_solver(r or "", domain=domain)))
    return predictions


# ===========================================================================
# Metrics
# ===========================================================================

def compute_metrics(predictions: List[Dict]) -> Dict:
    """Score a set of predictions. Each sample is judged by its own domain.

    The domain used to be read once, from the first prediction, and applied to
    the whole set. A run over all four datasets returns logical samples first, so
    the overall figure silently covered FLD and FOLIO only and left GSM8K and
    OlympiadBench out of it entirely — 0.810 overall against per-dataset numbers
    of .72 / .90 / .88 / .38, which is the mean of the first two.
    """
    domains = {p.get("domain", "logical") for p in predictions}
    if len(domains) > 1:
        parts = {d: compute_metrics([p for p in predictions
                                     if p.get("domain", "logical") == d])
                 for d in sorted(domains)}
        n_total   = sum(m["n_total"]   for m in parts.values())
        n_correct = sum(m["n_correct"] for m in parts.values())
        n_no_pred = sum(m["n_no_pred"] for m in parts.values())
        n_blocked = sum(m.get("n_blocked", 0) for m in parts.values())
        steps  = [p["response_steps"]  for p in predictions if p.get("response_steps")  is not None]
        tokens = [p["response_tokens"] for p in predictions if p.get("response_tokens") is not None]
        return {
            "accuracy":     round(n_correct / n_total, 4) if n_total else 0.0,
            # F1 belongs to the label datasets only, so a run spanning both
            # domains has none at the top level; FLD's and FOLIO's are in
            # per_dataset and in by_domain["logical"].
            "macro_f1":     None,
            "avg_steps":    round(float(np.mean(steps)),  2) if steps  else 0.0,
            "std_steps":    round(float(np.std(steps)),   2) if steps  else 0.0,
            "avg_tokens":   round(float(np.mean(tokens)), 1) if tokens else 0.0,
            "std_tokens":   round(float(np.std(tokens)),  1) if tokens else 0.0,
            "n_total":      n_total,
            "n_correct":    n_correct,
            "n_no_pred":    n_no_pred,
            "no_pred_rate": round(n_no_pred / n_total, 4) if n_total else 0.0,
            "n_blocked":    n_blocked,
            "by_domain":    parts,
            "domain":       "mixed",
        }

    domain = predictions[0].get("domain", "logical") if predictions else "logical"
    all_steps, all_tokens = [], []
    correct = no_pred = valid_total = n_blocked = 0

    if domain == "math":
        for p in predictions:
            gt   = str(p.get("ground_truth", "") or "").strip()
            pred = str(p.get("predicted", "") or "").strip()
            if not gt:
                continue
            # The gateway refused this one before the model saw it, so it is not
            # evidence either way and is left out of the denominator.
            if p.get("gateway_blocked"):
                n_blocked += 1
                continue
            valid_total += 1
            if not pred:
                no_pred += 1
            elif answers_match(pred, gt, dataset=p.get("source_dataset"),
                               answer_type=p.get("answer_type"), domain="math"):
                correct += 1
            if p.get("response_steps") is not None:
                all_steps.append(float(p["response_steps"]))
            if p.get("response_tokens") is not None:
                all_tokens.append(float(p["response_tokens"]))
        accuracy = correct / valid_total if valid_total else 0.0
        total_predicted = valid_total - no_pred
        precision = correct / total_predicted if total_predicted > 0 else 0.0
        # No F1 on the math datasets. There are no classes to average over, and
        # the quantity that used to be reported here, 2PA/(P+A) with P the
        # accuracy among answered samples, equals the accuracy exactly whenever
        # every sample is answered — it carried no information the accuracy and
        # the no-answer rate did not already carry. GSM8K and OlympiadBench are
        # reported by accuracy; FLD and FOLIO keep a real macro-F1 over
        # PROVED/DISPROVED.
        return {
            "accuracy":     round(accuracy, 4),
            "macro_f1":     None,
            "precision_answered": round(precision, 4),
            "avg_steps":    round(float(np.mean(all_steps)),  2) if all_steps  else 0.0,
            "std_steps":    round(float(np.std(all_steps)),   2) if all_steps  else 0.0,
            "avg_tokens":   round(float(np.mean(all_tokens)), 1) if all_tokens else 0.0,
            "std_tokens":   round(float(np.std(all_tokens)),  1) if all_tokens else 0.0,
            "n_total":      valid_total,
            "n_correct":    correct,
            "n_no_pred":    no_pred,
            "no_pred_rate": round(no_pred / valid_total if valid_total else 0.0, 4),
            "n_blocked":    n_blocked,
            "domain":       "math",
        }
    else:
        confusion = {gt: {p: 0 for p in BINARY_LABELS + ["none"]} for gt in BINARY_LABELS}
        for p in predictions:
            gt   = p.get("ground_truth")
            pred = p.get("predicted")
            if gt not in BINARY_LABELS:
                continue
            if p.get("gateway_blocked"):
                n_blocked += 1
                continue
            valid_total += 1
            if pred not in BINARY_LABELS:
                no_pred += 1
                confusion[gt]["none"] += 1
            else:
                confusion[gt][pred] += 1
                if pred == gt:
                    correct += 1
            if p.get("response_steps") is not None:
                all_steps.append(float(p["response_steps"]))
            if p.get("response_tokens") is not None:
                all_tokens.append(float(p["response_tokens"]))
        accuracy = correct / valid_total if valid_total else 0.0
        f1s = {}
        for cls in BINARY_LABELS:
            tp = confusion[cls][cls]
            fp = sum(confusion[g][cls] for g in BINARY_LABELS if g != cls)
            fn = sum(confusion[cls][p] for p in BINARY_LABELS if p != cls)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec  = tp / (tp + fn) if (tp + fn) else 0.0
            f1s[cls] = round(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0, 4)
        return {
            "accuracy":     round(accuracy, 4),
            "macro_f1":     round(sum(f1s.values()) / len(BINARY_LABELS), 4),
            "avg_steps":    round(float(np.mean(all_steps)),  2) if all_steps  else 0.0,
            "std_steps":    round(float(np.std(all_steps)),   2) if all_steps  else 0.0,
            "avg_tokens":   round(float(np.mean(all_tokens)), 1) if all_tokens else 0.0,
            "std_tokens":   round(float(np.std(all_tokens)),  1) if all_tokens else 0.0,
            "n_total":      valid_total,
            "n_correct":    correct,
            "n_no_pred":    no_pred,
            "no_pred_rate": round(no_pred / valid_total if valid_total else 0.0, 4),
            "n_blocked":    n_blocked,
            "per_class_f1": f1s,
            "confusion_matrix": confusion,
            "domain":       "logical",
        }


def compute_per_dataset(predictions: List[Dict]) -> Dict[str, Dict]:
    groups: Dict[str, List] = defaultdict(list)
    for p in predictions:
        groups[p.get("source_dataset", "unknown")].append(p)
    return {ds: compute_metrics(preds) for ds, preds in groups.items()}


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_metrics(m: Dict, label: str = "", indent: str = "") -> None:
    domain = m.get("domain", "logical")
    short  = {"__PROVED__": "PROVED", "__DISPROVED__": "DISPR"}
    if label:
        print(f"\n{indent}{'─'*70}\n{indent}  {label}\n{indent}{'─'*70}")
    print(f"{indent}  Accuracy         : {m['accuracy']:.4f}  ({m['n_correct']}/{m['n_total']})")
    if m.get("macro_f1") is None:
        # F1 is reported for FLD and FOLIO only — the math datasets have no
        # classes to average over. A mixed run therefore has none at this level.
        print(f"{indent}  Macro-F1         : FLD/FOLIO only (see per-dataset)")
    else:
        print(f"{indent}  Macro-F1         : {m['macro_f1']:.4f}")
    print(f"{indent}  Avg steps/resp   : {m['avg_steps']:.2f} ± {m['std_steps']:.2f}")
    print(f"{indent}  Avg tokens/resp  : {m['avg_tokens']:.1f} ± {m['std_tokens']:.1f}")
    print(f"{indent}  No-pred rate      : {m['no_pred_rate']:.4f}")
    if m.get("n_blocked"):
        print(f"{indent}  Gateway-refused  : {m['n_blocked']}  (excluded from the denominator)")
    if domain == "logical" and m.get("per_class_f1"):
        f1_str = "  ".join(f"{short[c]}={m['per_class_f1'].get(c, 0):.3f}" for c in BINARY_LABELS)
        print(f"{indent}  Per-class F1     : {f1_str}")
    for dom, sub in (m.get("by_domain") or {}).items():
        f1 = sub.get("macro_f1")
        f1s = f"macro_f1={f1:.4f}" if f1 is not None else "macro_f1=n/a (math)"
        print(f"{indent}    {dom:8s} acc={sub['accuracy']:.4f} "
              f"({sub['n_correct']}/{sub['n_total']})  {f1s}")


def print_comparison_table(results: List[Tuple[str, Dict]]) -> None:
    print("\n" + "═" * 100)
    print("  BASELINE COMPARISONS — SETTINGS F through K")
    print("═" * 100)
    hdr = (f"  {'Setting':<46}  {'Accuracy':>9}  {'Macro-F1':>9}"
           f"  {'Avg Steps':>11}  {'Avg Tokens':>12}  N")
    print(hdr)
    print("  " + "─" * 96)
    for label, m in results:
        steps_str  = f"{m['avg_steps']:.2f}±{m['std_steps']:.2f}"
        tokens_str = f"{m['avg_tokens']:.1f}±{m['std_tokens']:.1f}"
        f1 = m.get("macro_f1")
        f1_cell = f"{f1:>9.4f}" if f1 is not None else f"{'—':>9s}"
        print(f"  {label:<46}  {m['accuracy']:>9.4f}  {f1_cell}"
              f"  {steps_str:>11}  {tokens_str:>12}  {m['n_total']}")
    print("═" * 100)


# ===========================================================================
# MAIN
# ===========================================================================

SETTING_NAMES = {
    "A": "A: Zero-shot",
    "F": "F: Self-Consistency (k={k})",
    "G": "G: Universal Self-Consistency (k={k})",
    "H": "H: Self-Refine (1 iter)",
    "I": "I: LLM Self-Aggregation (k={k})",
    "J": "J: Self-Eval Beam Search (beam={bw})",
    "K": "K: Faithful CoT + Symbolic Solver",
}


async def run(args: argparse.Namespace) -> None:
    all_samples = load_raw_dataset(
        [Path(p) for p in args.datasets],
        seed=args.seed,
        per_dataset=args.per_dataset,
        max_samples=args.max_samples,
    )
    logger.info("Loaded %d samples", len(all_samples))

    # Domain distribution
    domain_counts: Dict[str, int] = defaultdict(int)
    for s in all_samples:
        domain_counts[s.get("domain", "logical")] += 1
    logger.info("Domain distribution: %s", dict(domain_counts))

    k          = args.shots
    model      = args.model
    api_key    = args.api_key
    base_url   = args.base_url
    semaphore  = asyncio.Semaphore(args.concurrency)

    all_results: List[Tuple[str, Dict]] = []
    full_output: Dict[str, Any] = {}

    setting_names = {
        "A": "A: Zero-shot",
        "F": f"F: Self-Consistency (k={k})",
        "G": f"G: Universal Self-Consistency (k={k})",
        "H": "H: Self-Refine (1 iter)",
        "I": f"I: LLM Self-Aggregation (k={k})",
        "J": f"J: Self-Eval Beam Search (beam={args.beam_width})",
        "K": "K: Faithful CoT + Symbolic Solver",
    }

    async with aiohttp.ClientSession() as session:
        for key in args.settings:
            name = setting_names[key]
            logger.info("=== Running Setting %s: %s ===", key, name)

            if key == "A":
                preds = await by_domain(
                    run_setting_zeroshot, all_samples, model, api_key, base_url, semaphore, session)
            elif key == "F":
                preds = await by_domain(
                    run_setting_self_consistency, all_samples, k, model, api_key, base_url, semaphore, session)
            elif key == "G":
                preds = await by_domain(
                    run_setting_usc, all_samples, k, model, api_key, base_url, semaphore, session)
            elif key == "H":
                preds = await by_domain(
                    run_setting_self_refine, all_samples, model, api_key, base_url, semaphore, session,
                    n_iterations=args.refine_iterations)
            elif key == "I":
                preds = await by_domain(
                    run_setting_self_aggregation, all_samples, k, model, api_key, base_url, semaphore, session)
            elif key == "J":
                preds = await by_domain(
                    run_setting_sebs, all_samples, model, api_key, base_url, semaphore, session,
                    beam_width=args.beam_width, max_steps=args.beam_max_steps)
            elif key == "K":
                preds = await by_domain(
                    run_setting_faithful_cot, all_samples, model, api_key, base_url, semaphore, session)
            else:
                continue

            overall = compute_metrics(preds)
            per_ds  = compute_per_dataset(preds)

            print_metrics(overall, label=f"Setting {key}: {name}")
            for ds, dm in sorted(per_ds.items()):
                print_metrics(dm, label=ds, indent="  ")

            all_results.append((name, overall))
            full_output[key] = {
                "setting":     name,
                "shots":       k if key in ("F", "G", "I") else 0,
                "overall":     overall,
                "per_dataset": per_ds,
                "predictions": preds,
            }

    print_comparison_table(all_results)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(full_output, f, indent=2, ensure_ascii=False)
        logger.info("Results saved → %s", out)

        summary = {
            "settings": [
                {"key": k, "name": v["setting"],
                 "accuracy": v["overall"]["accuracy"],
                 "macro_f1": v["overall"]["macro_f1"],
                 "avg_steps": v["overall"]["avg_steps"],
                 "avg_tokens": v["overall"]["avg_tokens"],
                 "n_total": v["overall"]["n_total"]}
                for k, v in full_output.items()
            ],
            "model":  args.model,
            "shots":  args.shots,
            "seed":   args.seed,
        }
        summary_path = out.with_suffix(".summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Summary saved → %s", summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline Comparisons — Settings F through K (no trace files needed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--datasets",   nargs="+", required=True,
                        help="FLD.json, FOLIO.json, or GSM8K.json files")
    parser.add_argument("--per_dataset", type=int, default=250,
                        help="Samples per dataset (balanced, default 250)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap total samples (for quick testing)")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--shots",       type=int,   default=3,
                        help="k for Self-Consistency / USC / Self-Aggregation (default 3)")
    parser.add_argument("--settings",    nargs="+", choices=ALL_SETTINGS,
                        default=ALL_SETTINGS,
                        help="Settings to run (default: all F-K)")
    parser.add_argument("--refine_iterations", type=int, default=1,
                        help="Self-Refine iterations (default 1)")
    parser.add_argument("--beam_width",      type=int, default=2,
                        help="Beam width for Setting J (default 2)")
    parser.add_argument("--beam_max_steps",   type=int, default=8,
                        help="Max steps per beam for Setting J (default 8)")
    parser.add_argument("--model",       default=DEFAULT_MODEL)
    parser.add_argument("--api_key",     default=OPENAI_API_KEY)
    parser.add_argument("--base_url",    default=OPENAI_BASE_URL)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--output",      default="baseline_results.json")

    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
