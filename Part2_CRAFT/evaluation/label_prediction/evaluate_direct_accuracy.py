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

# One copy of the answer readers, shared with the baselines and with
# evaluate_label_accuracy. Three copies of them had drifted apart.
from extract_label import (normalise_label, extract_label,  # noqa: E402
                          extract_math_answer, extract_pred,
                          normalise_math_answer, majority_vote)


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
from step_count import count_steps, count_tokens  # noqa: E402


def _is_math_sample(r: Dict) -> bool:
    """Detect math domain from a pipeline output record."""
    return (r.get("domain") == "math" or
            (r.get("target_answer") and not r.get("target_answer", "").startswith("__")))


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
        pred    = extract_pred(text, "math" if is_math else "logical")
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
            pred = extract_pred(text, "math" if is_math else "logical")
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
