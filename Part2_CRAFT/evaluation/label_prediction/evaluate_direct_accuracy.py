#!/usr/bin/env python3
"""
evaluate_direct_accuracy.py  (CRAFT — Direct Label Prediction Accuracy)
-----------------------------------------------------------------------
Zero-training-cost evaluation: extract label prediction accuracy and
trace quality statistics from CRAFT pipeline outputs.

Reports four metrics per setting:
  1. Accuracy          — predicted label vs ground truth
  2. Macro-F1          — balanced across 3 classes
  3. Avg steps/trace   — mean ± std number of reasoning steps
  4. Avg tokens/trace  — mean ± std completion token count (word-split proxy)

Supports three trace sources:
  --source synthesized   synthesized_traces.json   (Step 5, one trace per sample)
  --source k_traces      k_traces_N_samples.json   (Step 1, k traces per sample)
  --source cleaned       cleaned_traces*.json       (Step 3/3.3, k cleaned traces)

For k_traces and cleaned, majority vote is used for the label prediction,
and per-trace stats are averaged across all k traces.

Usage:
    # Single file
    python evaluation/label_prediction/evaluate_direct_accuracy.py \\
        --input craft_runs/craft_k5_full_fld_gemini/synthesized.json \\
        --source synthesized

    # Ablation: compare multiple pipeline outputs side-by-side
    python evaluation/label_prediction/evaluate_direct_accuracy.py --compare \\
        craft_runs/<run>/synthesized_step_by_step.json \\
        craft_runs/<run>/synthesized.json \\
        --labels "Setting D: step_by_step" "Setting E: RKG" \\
        --source synthesized \\
        --output ablation_results.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Config — Part2_CRAFT/config.py, for the part-local results root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from config import resolve_input, resolve_output
except ImportError:
    resolve_input = resolve_output = Path

# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------
VALID_LABELS = {"__PROVED__", "__DISPROVED__"}

_LABEL_RE      = re.compile(r"(__PROVED__|__DISPROVED__)", re.IGNORECASE)
_STEP_RE       = re.compile(r"^Step\s*\d+\s*[:\.]", re.IGNORECASE | re.MULTILINE)
_NL_DISPROVED  = re.compile(r"\b(disproved|false|incorrect|refuted)\b", re.IGNORECASE)
_NL_PROVED     = re.compile(r"\b(proved|proven|true|correct|holds)\b", re.IGNORECASE)
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
_HASH_ANS_RE   = re.compile(r"####\s*(.+)")
_FINAL_ANS_RE  = re.compile(
    r"(?:the\s+answer\s+is|final\s+answer\s*[:\=]|answer\s*[:\=])\s*([^\n\.]+)", re.IGNORECASE
)
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
    return {"TRUE": "__PROVED__", "FALSE": "__DISPROVED__"}.get(label)


def extract_label(text: str) -> Optional[str]:
    """Extract __PROVED__ or __DISPROVED__ from text."""
    if not text:
        return None
    # Primary: exact __PROVED__ or __DISPROVED__ format (last occurrence)
    matches = _LABEL_RE.findall(text)
    if matches:
        m = matches[-1].upper()
        return m if m in ("__PROVED__", "__DISPROVED__") else None
    # Secondary: markdown bold without underscores: **PROVED**, **DISPROVED**
    bold_matches = re.findall(r"\*\*(PROVED|DISPROVED)\*\*", text, re.I)
    if bold_matches:
        b = bold_matches[-1].upper()
        return {"PROVED": "__PROVED__", "DISPROVED": "__DISPROVED__"}.get(b)
    # Tertiary: look for "Final Conclusion: PROVED/DISPROVED" (without underscores)
    concl_m = re.search(r"(?:final\s+conclusion|conclusion)\s*[:\-]\s*(PROVED|DISPROVED)", text, re.I)
    if concl_m:
        return {"PROVED": "__PROVED__", "DISPROVED": "__DISPROVED__"}.get(concl_m.group(1).upper())
    # Fallback: search last 5 lines for natural language signals
    lines = [l.strip() for l in text.splitlines() if l.strip() and "[Step" not in l]
    tail = " ".join(lines[-5:]) if len(lines) >= 5 else " ".join(lines)
    if _NL_DISPROVED.search(tail):
        return "__DISPROVED__"
    if _NL_PROVED.search(tail):
        return "__PROVED__"
    # Last resort: scan the full text for natural language signals (last occurrence)
    all_disproved = list(re.finditer(r"\b(disproved|false|incorrect|refuted|cannot be proved|not proved|hypothesis is not)\b", text, re.I))
    all_proved = list(re.finditer(r"\b(proved|proven|true|correct|holds|hypothesis is proved|hypothesis holds)\b", text, re.I))
    if all_disproved and all_proved:
        # Take whichever appears last
        if all_disproved[-1].start() > all_proved[-1].start():
            return "__DISPROVED__"
        return "__PROVED__"
    if all_disproved:
        return "__DISPROVED__"
    if all_proved:
        return "__PROVED__"
    return None


def majority_vote(labels: List[Optional[str]]) -> Optional[str]:
    valid = [l for l in labels if l in VALID_LABELS]
    return Counter(valid).most_common(1)[0][0] if valid else None


def extract_math_answer(text: str) -> Optional[str]:
    """Extract numeric answer: \\boxed{} → #### N → 'the answer is N'.
    Commas are preserved (multi-answer support for OlympiadBench).
    """
    if not text:
        return None
    m = _extract_boxed_content(text)
    if m:
        raw = m[-1].strip()
        # Remove thousands-separator commas only (e.g. 1,000 → 1000)
        raw = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', raw)
        return raw
    m2 = _HASH_ANS_RE.findall(text)
    if m2:
        return m2[-1].strip()
    m3 = _FINAL_ANS_RE.findall(text)
    if m3:
        return m3[-1].strip()
    # Bare-number / short-answer fallback (USC selector, bare responses, OlympiadBench format)
    for candidate in [text.strip(), text.strip().splitlines()[-1].strip() if text.strip() else ""]:
        if candidate and len(candidate) <= 60:
            norm = _normalise_single(candidate)
            if norm is not None:
                return candidate
    return None


def _eval_latex_frac(text: str) -> str:
    """Evaluate \\frac / \\tfrac / \\dfrac / \\cfrac{a}{b} to a decimal string."""
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
    # Normalize Unicode minus/dash variants → ASCII minus
    s = s.replace('\u2013', '-').replace('\u2014', '-').replace('\u2212', '-')  # en-dash, em-dash, minus sign
    # Strip leading LaTeX spacing artifacts (e.g. \; or \, at start after multi-answer split)
    s = re.sub(r'^[;\\]+\s*', '', s)
    s = _eval_latex_frac(s)
    s = re.sub(r'\^?\{?\\circ\}?|°|\\degree', '', s)   # degree symbols
    s = re.sub(r'\\(?:text|mathrm|mbox)\{[^}]*\}', '', s)
    s = re.sub(
        r'\s*\b(?:km|cm|mm|m|kg|g|mg|l|ml|hours?|hrs?|minutes?|mins?|seconds?|secs?|'
        r'days?|weeks?|months?|years?|dollars?|cents?|degrees?)\b.*$',
        '', s, flags=re.IGNORECASE,
    )
    s = re.sub(r'\\[a-zA-Z]+\*?', '', s)
    s = re.sub(r'[{}]', '', s)
    s = s.strip().lstrip("$\\").strip(";")
    if s.endswith('%'):
        s = s[:-1]
    s = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', s)   # thousands separators
    # Normalize internal whitespace in algebraic expressions (e.g. "2 n" → "2n", "2rc + r + c" → "2rc+r+c")
    s = re.sub(r'\s+', '', s)
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
    # Strip matching outer dollar signs
    if ans.startswith('$') and ans.endswith('$') and len(ans) > 2:
        inner = ans[1:-1].strip()
        if not inner.startswith('$'):
            ans = inner
    # Normalize Unicode minus/dash variants → ASCII minus (before split)
    ans = ans.replace('\u2013', '-').replace('\u2014', '-').replace('\u2212', '-')
    # Normalize LaTeX spacing commands used as multi-answer separators: \, \; \quad → comma
    ans = re.sub(r'\\[,;]', ',', ans)
    ans = re.sub(r'\\quad\s*', ',', ans)
    ans = re.sub(r',+', ',', ans)          # collapse duplicate commas
    ans = ans.strip(',')
    # Strip outer parentheses or brackets enclosing a tuple: (1,2,3) → 1,2,3
    if ((ans.startswith('(') and ans.endswith(')')) or
            (ans.startswith('[') and ans.endswith(']'))):
        inner = ans[1:-1].strip()
        if inner:
            ans = inner
    # Comma-separated multi-answer (e.g. OlympiadBench "-1,-9" or "0.5,3.56")
    if ',' in ans:
        parts = [p.strip() for p in ans.split(',') if p.strip()]
        if len(parts) >= 2:
            norm_parts = [_normalise_single(p) for p in parts]
            valid_parts = [p for p in norm_parts if p is not None]
            if valid_parts and len(valid_parts) == len(norm_parts):
                try:
                    sorted_parts = sorted(valid_parts, key=float)
                except (ValueError, TypeError):
                    sorted_parts = sorted(valid_parts)
                return ','.join(sorted_parts)
    return _normalise_single(ans)


# ---------------------------------------------------------------------------
# Step / token counting
# ---------------------------------------------------------------------------

def count_steps(text: str, reasoning_steps: Optional[list] = None) -> int:
    """Count reasoning steps in a trace text."""
    if reasoning_steps and isinstance(reasoning_steps, list):
        return len([s for s in reasoning_steps if s and str(s).strip()])
    if not text:
        return 0
    matches = _STEP_RE.findall(text)
    if matches:
        return len(matches)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return min(len(lines), 30)  # cap at 30 to avoid noise


def count_tokens(text: str) -> int:
    """Approximate token count via whitespace split (~0.75× real BPE tokens)."""
    return len(text.split()) if text else 0


def _infer_dataset(sample_id: str) -> str:
    if "FLD" in sample_id or "Dataset1" in sample_id or "dataset1" in sample_id:
        return "FLD"
    if "FOLIO" in sample_id or "Dataset2" in sample_id or "dataset2" in sample_id:
        return "FOLIO"
    return "unknown"


def balance_samples(samples: List[Dict], n_per_class: int = 125, seed: int = 42) -> List[Dict]:
    """Keep exactly n_per_class PROVED + n_per_class DISPROVED
    per source_dataset. Math domain samples are returned as-is (no balancing).

    n_per_class=125 → 250 per logical dataset → 500 total across two datasets.
    """
    import random
    rng = random.Random(seed)

    # Separate math and logical samples
    math_samples    = [s for s in samples if s.get("domain") == "math"]
    logical_samples = [s for s in samples if s.get("domain") != "math"]

    result = list(math_samples)  # math: return all

    # Logical: balance PROVED/DISPROVED per dataset
    groups: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
    for s in logical_samples:
        gt = s.get("ground_truth")
        if gt not in ("__PROVED__", "__DISPROVED__"):
            continue
        groups[s.get("source_dataset", "unknown")][gt].append(s)

    for ds, label_map in groups.items():
        for gt, items in label_map.items():
            rng.shuffle(items)
            result.extend(items[:n_per_class])

    return result


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("results", data) if isinstance(data, dict) else data


# Whether a prediction is the gold answer is decided in answer_match, shared
# with the baselines, so CRAFT and a baseline are measured by the same rule.
# normalise_math_answer stays for display and grouping; it is no longer what
# decides correctness, because a canonical spelling cannot represent
# \frac{1}{2}, 1/2 and 0.5 as one string without also merging answers that differ.
from answer_match import answers_match  # noqa: E402


def _is_math_sample(r: Dict) -> bool:
    """Detect math domain from a pipeline output record."""
    return (r.get("domain") == "math" or
            (r.get("target_answer") and not r.get("target_answer", "").startswith("__")))


def _extract_pred(text: str, is_math: bool) -> Optional[str]:
    """The answer the trace states, as written. Comparison happens later."""
    if is_math:
        raw = extract_math_answer(text)
        return str(raw).strip() if raw else None
    return extract_label(text)


def _get_gt(r: Dict, is_math: bool) -> Optional[str]:
    raw = r.get("ground_truth") or r.get("target_answer")
    if is_math:
        return str(raw).strip() if raw else None
    return normalise_label(raw)


def load_synthesized(path: Path) -> List[Dict[str, Any]]:
    """Step 5 synthesized_traces.json — one trace per sample."""
    samples = []
    for r in _load_json(path):
        is_math = _is_math_sample(r)
        gt      = _get_gt(r, is_math)
        text    = r.get("synthesized_trace", "")
        pred    = _extract_pred(text, is_math)
        samples.append({
            "sample_id":      r.get("sample_id", ""),
            "source_dataset": r.get("source_dataset", _infer_dataset(r.get("sample_id", ""))),
            "ground_truth":   gt,
            "predicted":      pred,
            "domain":         "math" if is_math else "logical",
            "traces": [{"predicted": pred, "text": text,
                        "n_steps":  count_steps(text),
                        "n_tokens": count_tokens(text)}],
        })
    return samples


def load_k_traces(path: Path) -> List[Dict[str, Any]]:
    """Step 1 k_traces*.json — k traces per sample, majority vote."""
    samples = []
    for r in _load_json(path):
        is_math = _is_math_sample(r)
        gt      = _get_gt(r, is_math)
        traces  = []
        for t in r.get("traces", []):
            text = t.get("reasoning_text") or t.get("raw_response", "")
            if is_math:
                _raw = t.get("label") or extract_math_answer(text)
                pred = str(_raw).strip() if _raw else None
            else:
                pred = normalise_label(t.get("label")) or extract_label(text)
            traces.append({"predicted": pred, "text": text,
                           "n_steps":  count_steps(text, t.get("reasoning_steps")),
                           "n_tokens": count_tokens(text)})
        # math majority vote: most common numeric string
        all_preds = [t["predicted"] for t in traces if t["predicted"] is not None]
        mv_pred   = Counter(all_preds).most_common(1)[0][0] if all_preds else None
        samples.append({
            "sample_id":      r.get("sample_id", ""),
            "source_dataset": r.get("source_dataset", _infer_dataset(r.get("sample_id", ""))),
            "ground_truth":   gt,
            "traces":         traces,
            "predicted":      mv_pred,
            "domain":         "math" if is_math else "logical",
        })
    return samples


def load_cleaned(path: Path) -> List[Dict[str, Any]]:
    """Step 3/3.6 cleaned_traces*.json — k cleaned traces per sample."""
    samples = []
    for r in _load_json(path):
        is_math = _is_math_sample(r)
        gt      = _get_gt(r, is_math)
        traces  = []
        for t in r.get("cleaned_traces", []):
            text = t.get("reasoning_text") or ""
            pred = _extract_pred(text, is_math)
            traces.append({"predicted": pred, "text": text,
                           "n_steps":  count_steps(text, t.get("reasoning_steps")),
                           "n_tokens": count_tokens(text)})
        all_preds = [t["predicted"] for t in traces if t["predicted"] is not None]
        mv_pred   = Counter(all_preds).most_common(1)[0][0] if all_preds else None
        samples.append({
            "sample_id":      r.get("sample_id", ""),
            "source_dataset": r.get("source_dataset", _infer_dataset(r.get("sample_id", ""))),
            "ground_truth":   gt,
            "traces":         traces,
            "predicted":      mv_pred,
            "domain":         "math" if is_math else "logical",
        })
    return samples


LOADERS = {"synthesized": load_synthesized, "k_traces": load_k_traces, "cleaned": load_cleaned}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(samples: List[Dict]) -> Dict[str, Any]:
    """Domain-aware metrics: binary F1 for logical, exact-match for math."""
    # Infer domain from majority vote across all samples (robust to leading error records)
    if samples:
        math_count = sum(1 for s in samples if s.get("domain") == "math")
        domain = "math" if math_count > len(samples) / 2 else "logical"
    else:
        domain = "logical"
    BINARY_LABELS = ["__PROVED__", "__DISPROVED__"]

    correct = no_pred = valid_total = n_blocked = 0
    per_trace_acc: List[float] = []
    all_steps:  List[float] = []
    all_tokens: List[float] = []

    if domain == "math":
        for s in samples:
            gt   = str(s.get("ground_truth", "") or "").strip()
            pred = str(s.get("predicted", "") or "").strip()
            ds   = s.get("source_dataset")
            if not gt:
                continue
            # Refused by the API gateway before the model saw it: not evidence
            # about the model, so it leaves the denominator rather than counting
            # as a failure. Same rule as the baselines use.
            if s.get("gateway_blocked"):
                n_blocked += 1
                continue
            valid_total += 1
            if not pred:
                no_pred += 1
            elif answers_match(pred, gt, dataset=ds,
                               answer_type=s.get("answer_type"), domain="math"):
                correct += 1
            for t in s.get("traces", []):
                tp = str(t.get("predicted", "") or "").strip()
                if tp:
                    per_trace_acc.append(
                        1.0 if answers_match(tp, gt, dataset=ds,
                                             answer_type=s.get("answer_type"),
                                             domain="math") else 0.0)
                if t.get("n_steps") is not None:
                    all_steps.append(float(t["n_steps"]))
                if t.get("n_tokens") is not None:
                    all_tokens.append(float(t["n_tokens"]))

        accuracy = correct / valid_total if valid_total else 0.0
        total_predicted = valid_total - no_pred
        precision = correct / total_predicted if total_predicted > 0 else 0.0
        # No F1 on the math datasets: there are no classes to average over,
        # and 2PA/(P+A) equals the accuracy whenever every sample is answered.
        # FLD and FOLIO keep a real macro-F1 over PROVED/DISPROVED.
        return {
            "accuracy":    round(accuracy, 4),
            "macro_f1":    None,
            "precision_answered": round(precision, 4),
            "avg_steps":   round(float(np.mean(all_steps)),  2) if all_steps  else 0.0,
            "std_steps":   round(float(np.std(all_steps)),   2) if all_steps  else 0.0,
            "avg_tokens":  round(float(np.mean(all_tokens)), 1) if all_tokens else 0.0,
            "std_tokens":  round(float(np.std(all_tokens)),  1) if all_tokens else 0.0,
            "mean_trace_acc": round(float(np.mean(per_trace_acc)), 4) if per_trace_acc else round(accuracy, 4),
            "std_trace_acc":  round(float(np.std(per_trace_acc)),  4) if per_trace_acc else 0.0,
            "n_total":     valid_total,
            "n_correct":   correct,
            "n_no_pred":   no_pred,
        "n_blocked":    n_blocked,
            "no_pred_rate": round(no_pred / valid_total if valid_total else 0.0, 4),
            "per_class_f1": {},
            "confusion_matrix": {},
            "domain": "math",
        }

    # Logical domain
    confusion = {gt: {p: 0 for p in BINARY_LABELS + ["none"]} for gt in BINARY_LABELS}

    for s in samples:
        gt   = s["ground_truth"]
        pred = s["predicted"]
        if gt not in BINARY_LABELS:
            continue
        if s.get("gateway_blocked"):
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
        hits = []
        for t in s.get("traces", []):
            tp = t.get("predicted")
            if tp in BINARY_LABELS:
                hits.append(1.0 if tp == gt else 0.0)
            if t.get("n_steps") is not None:
                all_steps.append(float(t["n_steps"]))
            if t.get("n_tokens") is not None:
                all_tokens.append(float(t["n_tokens"]))
        if hits:
            per_trace_acc.append(sum(hits) / len(hits))

    accuracy = correct / valid_total if valid_total else 0.0
    f1s = {}
    for cls in BINARY_LABELS:
        tp   = confusion[cls][cls]
        fp   = sum(confusion[g][cls] for g in BINARY_LABELS if g != cls)
        fn   = sum(confusion[cls][p] for p in BINARY_LABELS if p != cls)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1s[cls] = round(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0, 4)

    return {
        "accuracy":    round(accuracy, 4),
        "macro_f1":    round(sum(f1s.values()) / len(BINARY_LABELS), 4),
        "avg_steps":   round(float(np.mean(all_steps)),  2) if all_steps  else 0.0,
        "std_steps":   round(float(np.std(all_steps)),   2) if all_steps  else 0.0,
        "avg_tokens":  round(float(np.mean(all_tokens)), 1) if all_tokens else 0.0,
        "std_tokens":  round(float(np.std(all_tokens)),  1) if all_tokens else 0.0,
        "mean_trace_acc": round(float(np.mean(per_trace_acc)), 4) if per_trace_acc else round(accuracy, 4),
        "std_trace_acc":  round(float(np.std(per_trace_acc)),  4) if per_trace_acc else 0.0,
        "n_total":     valid_total,
        "n_correct":   correct,
        "n_no_pred":   no_pred,
        "n_blocked":    n_blocked,
        "no_pred_rate": round(no_pred / valid_total if valid_total else 0.0, 4),
        "per_class_f1": f1s,
        "confusion_matrix": confusion,
        "domain": "logical",
    }


def compute_per_dataset(samples: List[Dict]) -> Dict[str, Dict]:
    groups: Dict[str, List] = defaultdict(list)
    for s in samples:
        groups[s.get("source_dataset", "unknown")].append(s)
    return {ds: compute_metrics(ss) for ds, ss in groups.items()}


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_metrics(m: Dict, label: str = "", indent: str = "") -> None:
    BINARY_LABELS = ["__PROVED__", "__DISPROVED__"]
    short  = {"__PROVED__": "PROVED", "__DISPROVED__": "DISPR"}
    domain = m.get("domain", "logical")
    if label:
        print(f"\n{indent}{'─'*66}")
        print(f"{indent}  {label}")
        print(f"{indent}{'─'*66}")
    print(f"{indent}  Accuracy         : {m['accuracy']:.4f}  ({m['n_correct']}/{m['n_total']})")
    if domain == "math":
        print(f"{indent}  Metric           : exact-match on numeric answer")
    else:
        if m.get("macro_f1") is None:
            print(f"{indent}  Macro-F1         : FLD/FOLIO only")
        else:
            print(f"{indent}  Macro-F1         : {m['macro_f1']:.4f}")
    print(f"{indent}  Avg steps/trace  : {m['avg_steps']:.2f} ± {m['std_steps']:.2f}")
    print(f"{indent}  Avg tokens/trace : {m['avg_tokens']:.1f} ± {m['std_tokens']:.1f}")
    if m.get("std_trace_acc", 0) > 0:
        print(f"{indent}  Mean trace acc   : {m['mean_trace_acc']:.4f} ± {m['std_trace_acc']:.4f}")
    print(f"{indent}  No-pred rate     : {m['no_pred_rate']:.4f}  ({m['n_no_pred']} samples)")
    if domain == "logical" and m.get("per_class_f1"):
        f1_str = "  ".join(f"{short[c]}={m['per_class_f1'].get(c, 0):.3f}" for c in BINARY_LABELS)
        print(f"{indent}  Per-class F1     : {f1_str}")
        if m.get("confusion_matrix"):
            print(f"\n{indent}  Confusion matrix (rows=GT, cols=Pred):")
            hdr_label = "GT/Pred"
            hdr = f"{indent}  {hdr_label:<10}" + "".join(f"{short[l]:>8}" for l in BINARY_LABELS) + f"{'none':>7}"
            print(hdr)
            cm = m["confusion_matrix"]
            for gt in BINARY_LABELS:
                row = f"{indent}  {short[gt]:<10}" + "".join(f"{cm.get(gt,{}).get(p,0):>8}" for p in BINARY_LABELS)
                row += f"{cm.get(gt,{}).get('none',0):>7}"
                print(row)


def print_comparison_table(results: List[Tuple[str, Dict]]) -> None:
    print("\n" + "═"*112)
    print("  COMPARISON TABLE")
    print("═"*112)
    hdr = (f"  {'Setting':<42}  {'Accuracy':>9}  {'Macro-F1':>9}"
           f"  {'Avg Steps':>12}  {'Avg Tokens':>13}  N")
    print(hdr)
    print("  " + "─"*108)
    for label, m in results:
        steps_str  = f"{m['avg_steps']:.2f}±{m['std_steps']:.2f}"
        tokens_str = f"{m['avg_tokens']:.1f}±{m['std_tokens']:.1f}"
        _f1 = m.get("macro_f1")
        _f1c = f"{_f1:>9.4f}" if _f1 is not None else f"{'—':>9s}"
        print(f"  {label:<42}  {m['accuracy']:>9.4f}  {_f1c}"
              f"  {steps_str:>12}  {tokens_str:>13}  {m['n_total']}")
    print("═"*112)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_single(path: Path, source: str, n_per_class: int = 125, seed: int = 42) -> Tuple[Dict, Dict]:
    samples = LOADERS[source](path)
    samples = balance_samples(samples, n_per_class=n_per_class, seed=seed)
    return compute_metrics(samples), compute_per_dataset(samples)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label accuracy + trace stats from pipeline outputs (no LLM calls).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate synthesized traces (Step 5 output)
  python evaluate_direct_accuracy.py \\
      --input craft_runs/<run>/synthesized.json --source synthesized

  # Evaluate k_traces (Step 1) — majority vote across k traces
  python evaluate_direct_accuracy.py \\
      --input craft_runs/<run>/k_traces.json --source k_traces

  # Ablation: compare multiple outputs (Settings B-E)
  python evaluate_direct_accuracy.py --compare \\
      craft_runs/<run>/synthesized_step_by_step.json \\
      craft_runs/<run>/synthesized.json \\
      --labels "Setting D: step_by_step" "Setting E: RKG" \\
      --source synthesized --output ablation.json
""",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input",   type=Path, help="Single trace file")
    group.add_argument("--compare", type=Path, nargs="+", help="Multiple files for ablation")

    parser.add_argument("--source", choices=["synthesized", "k_traces", "cleaned"],
                        default="synthesized",
                        help="Trace type: synthesized (Step 5) | k_traces (Step 1) | cleaned (Step 3/3.6)")
    parser.add_argument("--labels",      nargs="+", default=None,
                        help="Display labels for --compare mode")
    parser.add_argument("--output",       type=Path, default=None,
                        help="Save JSON results to this path (optional)")
    parser.add_argument("--per_dataset",  action="store_true", default=True,
                        help="Print per-dataset breakdown (default: True)")
    parser.add_argument("--n_per_class",  type=int, default=125,
                        help="Samples per class per dataset after balancing (default 125 → 250/dataset)")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    # Relative paths resolve against the part's results root (see config.resolve_*).
    if args.input:
        args.input = Path(resolve_input(args.input))
    if args.compare:
        args.compare = [Path(resolve_input(c)) for c in args.compare]
    if args.output:
        args.output = Path(resolve_output(args.output))

    if args.input:
        print(f"\nEvaluating: {args.input}  [source={args.source}]")
        overall, per_ds = evaluate_single(args.input, args.source, args.n_per_class, args.seed)
        print_metrics(overall, label=args.input.name)
        if args.per_dataset:
            for ds, dm in sorted(per_ds.items()):
                print_metrics(dm, label=ds, indent="  ")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump({"file": str(args.input), "source": args.source,
                           "overall": overall, "per_dataset": per_ds}, f, indent=2)
            print(f"\nResults saved → {args.output}")
    else:
        files  = args.compare
        labels = args.labels or [f.name for f in files]
        labels += [f.name for f in files[len(labels):]]  # pad if short

        all_results = []
        full_output = []
        for path, lbl in zip(files, labels):
            print(f"\nEvaluating: {path}  [{lbl}]")
            overall, per_ds = evaluate_single(path, args.source, args.n_per_class, args.seed)
            print_metrics(overall, label=lbl)
            if args.per_dataset:
                for ds, dm in sorted(per_ds.items()):
                    print_metrics(dm, label=f"{lbl} / {ds}", indent="  ")
            all_results.append((lbl, overall))
            full_output.append({"label": lbl, "file": str(path),
                                 "overall": overall, "per_dataset": per_ds})

        print_comparison_table(all_results)

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(full_output, f, indent=2, ensure_ascii=False)
            print(f"\nComparison saved → {args.output}")


if __name__ == "__main__":
    main()
