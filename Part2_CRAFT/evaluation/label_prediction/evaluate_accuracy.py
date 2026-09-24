#!/usr/bin/env python3
"""Label-prediction accuracy for CRAFT.

Two scripts once asked the same question of the same files and answered it
differently. evaluate_direct_accuracy scored one pipeline output;
evaluate_label_accuracy ran a Settings A-E ablation whose B through E are that
same scoring applied to four files. Each carried its own compute_metrics,
compute_per_dataset, print_metrics and answer reader, and the copies had
drifted, so the two could disagree about one file.

They disagreed in ways that mattered. The ablation compared maths answers by
normalising both sides to a string and testing equality, which loses whatever
the normaliser strips: 3\\pi/4 became 0.75 and no longer matched. It reported
macro-F1 for maths as equal to accuracy, where there are no classes to average
over. It read the domain from the first prediction and applied it to the whole
batch, the same read that once sent a hundred maths samples through the
logical parser. And it counted a sample the gateway refused as one the model
got wrong.

This file keeps one of each, the version without those faults: answers_match
decides maths equivalence symbolically, macro-F1 is None for maths and real
over PROVED/DISPROVED for logic, the domain is the majority of the batch, and
a refused sample leaves the denominator rather than counting as a failure.

The A-E ablation is gone with it. The paper's ablation is five named rows, and
evaluation/Ablation_Study/ablation_study.py builds them from the loaders and the metric
here, so there is one scorer and one definition of the ablation rather than a
second table that can drift from the first.

    python evaluate_accuracy.py score \\
        --input craft_runs/<run>/synthesized.json --source synthesized
"""


from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
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
# Config — Part2_CRAFT/config.py, which reads the gitignored repo-root config.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from config import (OPENAI_API_KEY, OPENAI_BASE_URL, DEFAULT_MODEL, REQUEST_TIMEOUT,
                        resolve_input, resolve_output)
except ImportError:
    resolve_input = resolve_output = Path
    OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    DEFAULT_MODEL   = os.getenv("OPENAI_MODEL", "gemini-3.1-flash-lite")
    REQUEST_TIMEOUT = int(os.getenv("OPENAI_REQUEST_TIMEOUT", "180"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


VALID_LABELS = {"__PROVED__", "__DISPROVED__"}

# One copy of the answer readers, shared with the baselines and with
# evaluate_label_accuracy. Three copies of them had drifted apart.
from extract_label import (normalise_label, extract_label,  # noqa: E402
                          extract_math_answer, extract_pred,
                          normalise_math_answer, majority_vote)


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
        # FLD and ProofWriter keep a real macro-F1 over PROVED/DISPROVED.
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
    """The one summary line for a run, and the datasets it is made of.

    This used to print ten lines — tokens, mean trace accuracy, per-class F1,
    a confusion matrix — and then repeat all ten for each of the four
    datasets, forty lines in which the three numbers that get compared were
    never adjacent. What a run is compared on is accuracy, macro-F1 where
    there are classes to average, and steps. Those are what it prints.
    """
    if label:
        print(f"\n{indent}{'─'*62}")
        print(f"{indent}  {label}")
        print(f"{indent}{'─'*62}")
    f1 = m.get("macro_f1")
    f1_cell = f"{f1:.3f}" if f1 is not None else "—"
    print(f"{indent}  overall        acc {m['accuracy']:.3f}   macro-F1 {f1_cell:>5}   "
          f"steps {m['avg_steps']:.1f}   n {m['n_total']}"
          + (f"   no-pred {m['n_no_pred']}" if m.get("n_no_pred") else "")
          + (f"   blocked {m['n_blocked']}" if m.get("n_blocked") else ""))
    if m.get("per_dataset"):
        print()
        print_per_dataset(m["per_dataset"], indent=indent)


def print_per_dataset(per_ds: Dict[str, Dict], indent: str = "") -> None:
    """One line per dataset: accuracy always, macro-F1 where there are classes.

    A logical dataset carries a two-class label, so it reports accuracy and
    macro-F1 over PROVED and DISPROVED — the two answer different questions
    when the classes are unbalanced in what a method predicts, so both are
    reported. A maths dataset has no classes to average over, and an F1 there
    would be accuracy under another name, so the column is left blank. Steps
    are reported for every dataset, since that is the column the methods are
    compared on and the only place a method's cost shows.
    """
    order = ["FLD", "ProofWriter", "OmniMATH", "OlympiadBench"]
    names = [d for d in order if d in per_ds] + [d for d in sorted(per_ds) if d not in order]
    if not names:
        return
    print(f"{indent}  {'dataset':<16}{'acc':>8}{'macro-F1':>10}{'steps':>9}{'n':>7}")
    for ds in names:
        m = per_ds[ds]
        f1 = m.get("macro_f1")
        f1_cell = "     —" if f1 is None else f"{f1:>10.3f}"
        if f1 is not None:
            f1_cell = f"{f1:>10.3f}"
        else:
            f1_cell = f"{'—':>10}"
        print(f"{indent}  {ds:<16}{m['accuracy']:>8.3f}{f1_cell}{m['avg_steps']:>9.1f}{m['n_total']:>7}")


def _infer_dataset(sample_id: str) -> str:
    if "FLD" in sample_id or "Dataset1" in sample_id or "dataset1" in sample_id:
        return "FLD"
    if "ProofWriter" in sample_id or "Dataset2" in sample_id or "dataset2" in sample_id:
        return "ProofWriter"
    return "unknown"


def balance_samples(samples: List[Dict], n_per_class: int = 250, seed: int = 42) -> List[Dict]:
    """Keep at most n_per_class PROVED + n_per_class DISPROVED
    per source_dataset. Math domain samples are returned as-is (no balancing).

    The default, 250 per class, keeps every problem of a 500-problem logical
    dataset drawn 250/250, so a bare `score` reports the same 500 problems the
    baselines are scored on. A smaller value draws a balanced subset.
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

    # Sort before shuffling, and shuffle each group from its own seed. Synthesis
    # writes its results in completion order, which asyncio does not fix, so two
    # runs over the same samples produce files whose record order differs. The
    # shuffle then drew a different 125 from each, and two settings scored on
    # different subsets were reported side by side as if they were comparable —
    # on one FLD pair that read as 0.960 against 0.896 where scoring every
    # sample puts both at 0.887. Ordering by sample_id makes the subset a
    # function of the samples alone, so any two settings over the same run are
    # scored on the same rows.
    for ds in sorted(groups):
        label_map = groups[ds]
        for gt in sorted(label_map):
            items = sorted(label_map[gt], key=lambda s: str(s.get("sample_id", "")))
            random.Random(f"{seed}:{ds}:{gt}").shuffle(items)
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
            # The surviving steps and the trace's own closing statement, which is
            # the other half of what survived filtering. Reading the steps alone
            # left 78% of the maths traces with nothing to extract — a filter
            # drops the concluding sentence from the step list, not from the
            # trace — and "w/o Synthesis" then scored the extraction failure
            # rather than the vote. The stored generation-time label is still
            # not used: this is re-derived from surviving text, which is what
            # separates this row from "w/o Filter & Synthesis".
            text = "\n".join(
                part for part in (t.get("reasoning_text"), t.get("final_statement"))
                if part
            )
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


def evaluate_single(path: Path, source: str, n_per_class: int = 250, seed: int = 42) -> Tuple[Dict, Dict]:
    samples = LOADERS[source](path)
    samples = balance_samples(samples, n_per_class=n_per_class, seed=seed)
    return compute_metrics(samples), compute_per_dataset(samples)


def main_score(argv) -> None:
    parser = argparse.ArgumentParser(
        description="Label accuracy + trace stats from pipeline outputs (no LLM calls).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate synthesized traces (Module III output)
  python evaluate_accuracy.py score \\
      --input craft_runs/<run>/synthesized.json --source synthesized

  # Evaluate the K traces of Module I — majority vote across them
  python evaluate_accuracy.py score \\
      --input craft_runs/<run>/k_traces.json --source k_traces

  # Compare several outputs, e.g. w/o RKG against CRAFT
  python evaluate_accuracy.py score --compare \\
      craft_runs/<run>/synthesized_step_by_step.json \\
      craft_runs/<run>/synthesized.json \\
      --labels "w/o RKG" "CRAFT" \\
      --source synthesized --output ablation.json
""",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input",   type=Path, help="Single trace file")
    group.add_argument("--compare", type=Path, nargs="+", help="Multiple files for ablation")

    parser.add_argument("--source", choices=["synthesized", "k_traces", "cleaned"],
                        default="synthesized",
                        help="Trace type: synthesized (Module III) | k_traces (Module I rollouts) | cleaned (after Steps Filtering)")
    parser.add_argument("--labels",      nargs="+", default=None,
                        help="Display labels for --compare mode")
    parser.add_argument("--output",       type=Path, default=None,
                        help="Save JSON results to this path (optional)")
    parser.add_argument("--per_dataset",  action="store_true", default=True,
                        help="Print per-dataset breakdown (default: True)")
    parser.add_argument("--n_per_class",  type=int, default=250,
                        help="Samples per class per logical dataset after balancing (default 250 → all 500 of a "
                             "250/250 dataset, the same problems the baselines are scored on)")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args(argv)

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
        if args.per_dataset:
            overall["per_dataset"] = per_ds
        print_metrics(overall, label=args.input.name)


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




def load_raw_dataset(paths: List[Path], seed: int = 42, per_dataset: int = 250) -> List[Dict]:
    """Load datasets with domain-aware handling.

    Logical domain (FLD/ProofWriter):
      - Balance to per_dataset/2 PROVED + per_dataset/2 DISPROVED per file.

    Math domain (GSM8K, detected via 'domain'=='math' or 'answer' field present):
      - No label filtering/balancing — use numeric answer as ground_truth.
      - Take min(len(items), per_dataset*2) samples.
      - 'input' = question, 'ground_truth' = numeric answer string.
    """
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
            # Build with ORIGINAL file indices so sample_ids match pipeline outputs.
            # Pipeline Step 1 names samples as "{stem}_{source_index}" where
            # source_index is the position in the raw JSON file.
            # Load ALL valid samples — caller (run()) applies the per_dataset cap
            # after restricting to what exists in the pipeline pools.
            for orig_idx, item in enumerate(items):
                problem_text = (item.get("input") or item.get("question", "")).strip()
                raw_gt = item.get("answer", "")
                gt = normalise_math_answer(str(raw_gt)) if raw_gt else None
                if not problem_text or not gt:
                    continue
                all_samples.append({
                    "sample_id":      item.get("sample_id") or f"{stem}_{orig_idx}",
                    "source_dataset": stem,
                    "problem_text":   problem_text,
                    "ground_truth":   gt,
                    "domain":         "math",
                })
            logger.info("%s (math): %d valid samples loaded", stem,
                        sum(1 for s in all_samples if s["source_dataset"] == stem))
        else:
            proved, disproved = [], []
            for idx, item in enumerate(items):
                gt = normalise_label(
                    item.get("proof_label") or item.get("Label") or item.get("label")
                )
                problem_text = item.get("input", "").strip()
                if not gt or not problem_text or gt not in VALID_LABELS:
                    continue
                entry = {
                    "sample_id":      item.get("sample_id") or f"{stem}_{idx}",
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

    return all_samples


def stratified_split(
    samples: List[Dict], test_ratio: float = 0.2, seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)
    groups: Dict[Tuple, List] = defaultdict(list)
    for s in samples:
        groups[(s["source_dataset"], s["ground_truth"])].append(s)
    train, test = [], []
    for group in groups.values():
        rng.shuffle(group)
        n_test = max(1, round(len(group) * test_ratio))
        test.extend(group[:n_test])
        train.extend(group[n_test:])
    return train, test


# ---------------------------------------------------------------------------
# Trace pool loading (Settings B–E)
# ---------------------------------------------------------------------------


def load_trace_pool(path: Optional[Path], source: str) -> Dict[str, Dict]:
    """Load a trace file → {sample_id: {text, n_steps, n_tokens}}."""
    if not path or not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", data) if isinstance(data, dict) else data
    pool: Dict[str, Dict] = {}

    for r in results:
        sid = r.get("sample_id", "")
        if not sid:
            continue
        if source == "synthesized":
            text = r.get("synthesized_trace", "")
        elif source == "k_traces":
            traces = r.get("traces", [])
            text   = ""
            # Majority vote among traces — no GT lookup (no leakage)
            label_counts: Counter = Counter()
            for t in traces:
                t_text = t.get("reasoning_text") or t.get("raw_response", "")
                lbl = normalise_label(t.get("label")) or extract_label(t_text)
                if lbl in VALID_LABELS:
                    label_counts[lbl] += 1
            mv_label = label_counts.most_common(1)[0][0] if label_counts else None
            for t in traces:
                t_text = t.get("reasoning_text") or t.get("raw_response", "")
                lbl = normalise_label(t.get("label")) or extract_label(t_text)
                if lbl == mv_label and t_text:
                    text = t_text
                    break
            if not text and traces:
                text = traces[0].get("reasoning_text") or traces[0].get("raw_response", "")
        elif source == "cleaned":
            cleaned = r.get("cleaned_traces", [])
            text    = ""
            if cleaned:
                # Majority vote among cleaned traces — consistent with Setting B
                label_counts_c: Counter = Counter()
                for t in cleaned:
                    t_text = t.get("reasoning_text", "")
                    lbl = normalise_label(t.get("label")) or extract_label(t_text)
                    if lbl in VALID_LABELS:
                        label_counts_c[lbl] += 1
                mv_label_c = label_counts_c.most_common(1)[0][0] if label_counts_c else None
                for t in cleaned:
                    t_text = t.get("reasoning_text", "")
                    lbl = normalise_label(t.get("label")) or extract_label(t_text)
                    if lbl == mv_label_c and t_text:
                        text = t_text
                        break
                if not text:
                    # fallback: longest
                    text = max((t.get("reasoning_text", "") for t in cleaned), key=len, default="")
        else:
            continue
        if text:
            pool[sid] = {
                "text":     text,
                "n_steps":  count_steps(text),
                "n_tokens": count_tokens(text),
            }
    return pool


def load_label_pool(path: Optional[Path], source: str) -> Dict[str, Dict]:
    """Load a pipeline output file → {sample_id: {label, n_steps, n_tokens, text}}.

    source='k_traces'   : majority vote over raw traces (Setting B)
    source='cleaned'    : majority vote over cleaned traces (Setting C)
    source='synthesized': extract label from synthesized trace (Setting D)

    Handles both logical (__PROVED__/__DISPROVED__) and math (numeric answer) domains.
    Domain is auto-detected from the first record's 'domain' field or presence of
    'target_answer' / absence of 'proof_label'.
    """
    if not path or not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", data) if isinstance(data, dict) else data
    if not results:
        return {}

    # Domain detection: scan the first 20 records, since the first may be an
    # error item with no fields.
    #
    # A record's own "domain" decides it. The fallbacks below only run when no
    # record carries one, because "target_answer" does not mean what it looks
    # like: a k_traces record has it on every dataset, holding "__PROVED__" on
    # the logical ones. Reading it as a sign of maths sent FLD and ProofWriter
    # through normalise_math_answer, which returns None for "__PROVED__", so
    # every vote came back empty and Settings B and C reported 0.0000 --- not a
    # component contributing nothing, but a domain read the wrong way.
    def _detect_math(records):
        # The record's own domain decides it, wherever it is written. A
        # k_traces record carries it at the top; a cleaned record does not, and
        # carries it on each trace instead.
        for rec in records[:20]:
            d = rec.get("domain")
            if d:
                return d == "math"
            for key in ("traces", "cleaned_traces", "original_traces"):
                for t in (rec.get(key) or [])[:3]:
                    if t.get("domain"):
                        return t["domain"] == "math"
        # No domain anywhere: read what the target answer looks like. Its
        # presence says nothing --- every dataset's k_traces has one --- but a
        # logical dataset's is __PROVED__ or __DISPROVED__.
        for rec in records[:20]:
            ta = rec.get("target_answer")
            if isinstance(ta, str) and normalise_label(ta) in VALID_LABELS:
                return False
            if ta is not None:
                return True
            if "answer" in rec and "proof_label" not in rec and "Label" not in rec:
                return True
        return False
    is_math = _detect_math(results)

    pool: Dict[str, Dict] = {}

    for r in results:
        sid = r.get("sample_id", "")
        if not sid:
            continue

        if source == "synthesized":
            text = r.get("synthesized_trace", "")
            if is_math:
                # pred_label not written for math pipelines — extract from trace text
                raw_pred = r.get("pred_label")
                if raw_pred is not None:
                    label = normalise_math_answer(str(raw_pred))
                else:
                    label = normalise_math_answer(extract_math_answer(text) or "")
            else:
                label = normalise_label(r.get("pred_label") or r.get("predicted_label"))
                if label is None:
                    label = extract_label(text)
            pool[sid] = {
                "label":    label,
                "text":     text,
                "n_steps":  count_steps(text),
                "n_tokens": count_tokens(text),
            }

        elif source in ("k_traces", "cleaned"):
            traces_key = "traces" if source == "k_traces" else "cleaned_traces"
            traces = r.get(traces_key, [])
            label_counts: Counter = Counter()
            best_text = ""

            if is_math:
                for t in traces:
                    raw_lbl = t.get("label")
                    if raw_lbl is not None:
                        lbl = normalise_math_answer(str(raw_lbl))
                    else:
                        t_text = t.get("reasoning_text") or t.get("raw_response", "")
                        lbl = normalise_math_answer(extract_math_answer(t_text) or "")
                    if lbl:
                        label_counts[lbl] += 1
                mv_label = label_counts.most_common(1)[0][0] if label_counts else None
                for t in traces:
                    raw_lbl = t.get("label")
                    lbl = (normalise_math_answer(str(raw_lbl)) if raw_lbl is not None
                           else normalise_math_answer(
                               extract_math_answer(
                                   t.get("reasoning_text") or t.get("raw_response", "")) or ""))
                    t_text = t.get("reasoning_text") or t.get("raw_response", "")
                    if lbl == mv_label and t_text:
                        best_text = t_text
                        break
            else:
                for t in traces:
                    t_text = t.get("reasoning_text") or t.get("raw_response", "")
                    lbl = normalise_label(t.get("label") or t.get("pred_label")) or extract_label(t_text)
                    if lbl in VALID_LABELS:
                        label_counts[lbl] += 1
                mv_label = label_counts.most_common(1)[0][0] if label_counts else None
                for t in traces:
                    t_text = t.get("reasoning_text") or t.get("raw_response", "")
                    lbl = normalise_label(t.get("label") or t.get("pred_label")) or extract_label(t_text)
                    if lbl == mv_label and t_text:
                        best_text = t_text
                        break

            if not best_text and traces:
                best_text = (traces[0].get("reasoning_text") or
                             traces[0].get("raw_response", ""))
            pool[sid] = {
                "label":    mv_label,
                "text":     best_text,
                "n_steps":  count_steps(best_text),
                "n_tokens": count_tokens(best_text),
            }

    return pool


# ---------------------------------------------------------------------------
# Example retrieval
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    return re.findall(r'\b[a-z]{2,}\b', text.lower())


def _bm25(query_tok, doc_tok, idf, avgdl, k1=1.5, b=0.75):
    dl = len(doc_tok)
    tf = Counter(doc_tok)
    return sum(
        idf.get(tok, 0) * (tf[tok] * (k1 + 1)) / (tf[tok] + k1 * (1 - b + b * dl / max(avgdl, 1)))
        for tok in set(query_tok)
    )


def select_examples(
    pool: List[Dict],
    trace_pool: Dict[str, Dict],
    query_text: str,
    k: int,
    exclude_ids: set,
    retrieval: str,
    seed: int = 42,
) -> List[Dict]:
    available = [s for s in pool
                 if s["sample_id"] not in exclude_ids and s["sample_id"] in trace_pool]
    if not available:
        return []
    if retrieval == "similar":
        all_docs = [_tokenize(s["problem_text"]) for s in available]
        N  = len(all_docs)
        df: Counter = Counter()
        for doc in all_docs:
            for t in set(doc):
                df[t] += 1
        idf   = {t: math.log((N - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}
        avgdl = sum(len(d) for d in all_docs) / max(N, 1)
        qtok  = _tokenize(query_text)
        scored = sorted(((s, _bm25(qtok, doc, idf, avgdl)) for s, doc in zip(available, all_docs)),
                        key=lambda x: -x[1])
        return [s for s, _ in scored[:k]]
    else:
        random.seed(seed)
        return random.sample(available, min(k, len(available)))


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


SYSTEM_LOGICIAN = (
    "You are an expert logician. Given premises and a hypothesis, decide whether "
    "the hypothesis is __PROVED__ or __DISPROVED__ based solely on the premises. "
    "Output your step-by-step reasoning, then on the very last line write ONLY: "
    "__PROVED__ or __DISPROVED__."
)

# Used for Setting E direct-context mode: model must READ the provided trace
# and extract the conclusion, NOT generate new reasoning.


SYSTEM_LOGICIAN_EXTRACT = (
    "You are an expert logician. A pre-computed reasoning chain is provided. "
    "Read it carefully and identify the final conclusion. "
    "Output ONLY the conclusion label: __PROVED__ or __DISPROVED__. "
    "Do NOT generate new reasoning — extract the answer from the chain above."
)


SYSTEM_LOGICIAN_ZEROSHOT = (
    "You are an expert logician. Given premises and a hypothesis, decide whether "
    "the hypothesis is __PROVED__ or __DISPROVED__ based solely on the premises. "
    "Output ONLY the label on the last line: __PROVED__ or __DISPROVED__."
)


SYSTEM_MATH = (
    "You are an expert mathematician. Solve the math problem step by step. "
    "Show all calculations clearly. "
    "On the very last line write ONLY the numeric answer in the format: \\boxed{<answer>}"
)


SYSTEM_MATH_ZEROSHOT = (
    "You are an expert mathematician. Solve the math problem. "
    "Output ONLY the numeric answer on the last line in the format: \\boxed{<answer>}"
)


SYSTEM_VERIFIER_LOGICAL = (
    "You are a logical reasoning verifier. Given a problem and a proposed reasoning chain, "
    "rate the logical soundness, completeness, and correctness of the reasoning on a scale "
    "from 0 to 10. A score of 10 means the reasoning is perfectly valid and the conclusion "
    "follows necessarily from the premises. A score of 0 means the reasoning is clearly "
    "invalid or the conclusion is wrong. Output ONLY a single integer from 0 to 10."
)


SYSTEM_VERIFIER_MATH = (
    "You are a math verifier. Given a math problem and a proposed solution, "
    "rate the correctness of the step-by-step reasoning and the final answer on a scale "
    "from 0 to 10. A score of 10 means every step is correct and the final answer is "
    "definitely right. A score of 0 means the reasoning is clearly wrong. "
    "Output ONLY a single integer from 0 to 10."
)


def build_verifier_prompt(problem: str, solution: str, domain: str = "logical") -> str:
    if domain == "math":
        return (f"Problem:\n{problem}\n\n"
                f"Proposed solution:\n{solution}\n\n"
                "Rate the correctness of this reasoning and final answer (0-10).\n"
                "Output ONLY a single integer:")
    return (f"Problem:\n{problem}\n\n"
            f"Proposed reasoning:\n{solution}\n\n"
            "Rate the logical soundness and correctness of this reasoning (0-10).\n"
            "Output ONLY a single integer:")


def _problem_block(problem: str) -> str:
    return f"Problem:\n{problem}"


def build_zeroshot_prompt(problem: str, domain: str = "logical") -> str:
    if domain == "math":
        return (_problem_block(problem) + "\n\n"
                "Solve step by step. Last line must be: \\boxed{<numeric answer>}")
    return (_problem_block(problem) + "\n\n"
            "Output ONLY one of: __PROVED__, __DISPROVED__")


def build_icl_prompt(problem: str, examples: List[Dict],
                     trace_pool: Dict[str, Dict], domain: str = "logical") -> str:
    parts = []
    for i, ex in enumerate(examples, 1):
        trace   = trace_pool[ex["sample_id"]]["text"]
        trimmed = trace[:800] + ("..." if len(trace) > 800 else "")
        parts += [
            f"--- Example {i} ---",
            _problem_block(ex["problem_text"]),
            f"Reasoning:\n{trimmed}",
            f"Answer: {ex['ground_truth']}",
            "",
        ]
    parts += [
        "--- New Problem ---",
        _problem_block(problem),
        "Reasoning: [your step-by-step reasoning]\nAnswer:",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

async def call_llm(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    system: str,
    user: str,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> Optional[str]:
    url     = base_url.rstrip("/") + "/chat/completions"
    auth_value = f"Bearer {api_key}"
    headers = {"Authorization": auth_value, "Content-Type": "application/json"}

    # o4-mini (and other o-series reasoning models) consume reasoning tokens internally;
    # with small max_tokens the content field returns empty. Force at least 4096.
    _is_reasoning = any(x in model.lower() for x in ("o1", "o3", "o4", "o-mini"))
    effective_max = max(max_tokens, 4096) if _is_reasoning else max_tokens

    payload = {
        "model":    model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user",   "content": user}],
        "temperature": temperature,
    }
    # o-series uses max_completion_tokens; standard models use max_tokens
    if _is_reasoning:
        payload["max_completion_tokens"] = effective_max
    else:
        payload["max_tokens"] = effective_max
    for attempt in range(4):
        async with semaphore:
            try:
                async with session.post(
                    url, headers=headers, json=payload,
                    timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        msg = data["choices"][0]["message"]
                        # o4-mini / deepseek-r1 via our API gateway put the reply in
                        # reasoning_content; content field is empty
                        content = (msg.get("reasoning_content") or
                                   msg.get("content") or "")
                        if content and content.strip():
                            return content.strip()
                        logger.warning("Empty content on attempt %d, retrying", attempt + 1)
                    else:
                        logger.warning("HTTP %s: %s", resp.status, (await resp.text())[:200])
            except Exception as e:
                logger.warning("Attempt %d: %s", attempt + 1, e)
        await asyncio.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Setting implementations
# ---------------------------------------------------------------------------

async def run_setting_best_of_n(
    test_samples: List[Dict],
    model: str, api_key: str, base_url: str,
    semaphore: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    n: int = 8,
) -> List[Dict]:
    """Setting L: Best-of-N.

    For each sample:
      1. Generate N independent solutions (temperature=0.7).
      2. Score each with a domain-aware LLM verifier (temperature=0).
      3. Return the answer of the highest-scored solution.
         Ties are broken by the most common answer among tied solutions.
    """

    async def _process_one(s: Dict) -> Dict:
        domain = s.get("domain", "logical")
        problem = s["problem_text"]

        gen_system = SYSTEM_MATH if domain == "math" else SYSTEM_LOGICIAN
        ver_system = SYSTEM_VERIFIER_MATH if domain == "math" else SYSTEM_VERIFIER_LOGICAL
        gen_prompt = build_zeroshot_prompt(problem, domain=domain)

        # Step 1: generate N solutions in parallel
        gen_coros = [
            call_llm(session, semaphore, gen_system, gen_prompt,
                     model, api_key, base_url, temperature=0.7)
            for _ in range(n)
        ]
        solutions_raw = await asyncio.gather(*gen_coros)
        solutions = [sol for sol in solutions_raw if sol and sol.strip()]

        if not solutions:
            return {
                "sample_id":       s["sample_id"],
                "source_dataset":  s["source_dataset"],
                "ground_truth":    s["ground_truth"],
                "predicted":       None,
                "raw_response":    "",
                "response_steps":  0,
                "response_tokens": 0,
                "domain":          domain,
                "bon_n_generated": 0,
                "bon_best_score":  0,
            }

        # Step 2: score each solution
        ver_coros = [
            call_llm(session, semaphore, ver_system,
                     build_verifier_prompt(problem, sol, domain=domain),
                     model, api_key, base_url, temperature=0.0)
            for sol in solutions
        ]
        scores_raw = await asyncio.gather(*ver_coros)

        def _parse_score(r: Optional[str]) -> int:
            if not r:
                return 0
            m = re.search(r'\b(\d+)\b', r)
            return min(10, max(0, int(m.group(1)))) if m else 0

        scores = [_parse_score(r) for r in scores_raw]
        best_score = max(scores)

        # Collect all solutions tied at the best score
        top_solutions = [sol for sol, sc in zip(solutions, scores) if sc == best_score]

        # Extract answers from top solutions; pick most common (minor tie-break)
        if domain == "math":
            top_answers = [normalise_math_answer(extract_math_answer(sol)) for sol in top_solutions]
            top_answers_valid = [a for a in top_answers if a]
            if top_answers_valid:
                predicted = Counter(top_answers_valid).most_common(1)[0][0]
            else:
                predicted = None
        else:
            top_answers = [extract_label(sol) for sol in top_solutions]
            top_answers_valid = [a for a in top_answers if a in VALID_LABELS]
            predicted = Counter(top_answers_valid).most_common(1)[0][0] if top_answers_valid else None

        best_solution = top_solutions[0]  # for step/token counting

        return {
            "sample_id":       s["sample_id"],
            "source_dataset":  s["source_dataset"],
            "ground_truth":    s["ground_truth"],
            "predicted":       predicted,
            "raw_response":    best_solution,
            "response_steps":  count_steps(best_solution),
            "response_tokens": count_tokens(best_solution),
            "domain":          domain,
            "bon_n_generated": len(solutions),
            "bon_best_score":  best_score,
        }

    coros = [_process_one(s) for s in test_samples]
    return list(await asyncio.gather(*coros))


async def run_setting_zeroshot(
    test_samples: List[Dict],
    model: str, api_key: str, base_url: str,
    semaphore: asyncio.Semaphore,
    session: aiohttp.ClientSession,
) -> List[Dict]:
    """Setting A: zero-shot, single call per sample. Domain-aware."""
    domain = test_samples[0].get("domain", "logical") if test_samples else "logical"
    system = SYSTEM_MATH_ZEROSHOT if domain == "math" else SYSTEM_LOGICIAN_ZEROSHOT

    coros = [
        call_llm(session, semaphore, system,
                 build_zeroshot_prompt(s["problem_text"], domain=domain),
                 model, api_key, base_url)
        for s in test_samples
    ]
    responses = await asyncio.gather(*coros)

    results = []
    for s, r in zip(test_samples, responses):
        if domain == "math":
            predicted = normalise_math_answer(extract_math_answer(r))
        else:
            predicted = extract_label(r)
        results.append({
            "sample_id":       s["sample_id"],
            "source_dataset":  s["source_dataset"],
            "ground_truth":    s["ground_truth"],
            "predicted":       predicted,
            "raw_response":    r or "",
            "response_steps":  count_steps(r or ""),
            "response_tokens": count_tokens(r or ""),
            "domain":          domain,
        })
    return results


def run_setting_offline(
    test_samples: List[Dict],
    label_pool: Dict[str, Dict],
) -> List[Dict]:
    """Settings B–E: read predicted label from pipeline output, no LLM call.

    label_pool maps sample_id → {label, text, n_steps, n_tokens}.
    Samples not found in the pool get predicted=None (counted as no-pred).
    """
    results = []
    for s in test_samples:
        sid = s["sample_id"]
        entry = label_pool.get(sid)
        if entry:
            predicted = entry["label"]
            text      = entry["text"]
            n_steps   = entry["n_steps"]
            n_tokens  = entry["n_tokens"]
        else:
            predicted = None
            text      = ""
            n_steps   = 0
            n_tokens  = 0
        results.append({
            "sample_id":       sid,
            "source_dataset":  s["source_dataset"],
            "ground_truth":    s["ground_truth"],
            "predicted":       predicted,
            "raw_response":    text,
            "response_steps":  n_steps,
            "response_tokens": n_tokens,
            "domain":          s.get("domain", "logical"),
        })
    return results


async def run_setting_icl(
    name: str,
    test_samples: List[Dict],
    train_pool: List[Dict],
    trace_pool: Dict[str, Dict],
    shots: int,
    retrieval: str,
    model: str, api_key: str, base_url: str,
    semaphore: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    seed: int,
    fixed_shots: Optional[List[Dict]] = None,
    e_direct_extract: bool = False,
) -> List[Dict]:
    """Settings B–E: ICL with a trace pool. Domain-aware.

    If fixed_shots is provided (list of dicts with problem_text, ground_truth, trace),
    those are used as-is for every query instead of selecting from train_pool.

    e_direct_extract=True (Setting E only): directly extract the label from the
    synthesized trace without an LLM call. This measures the framework's output
    quality rather than a downstream model's reasoning ability.
    e_direct_extract=False (default): use LLM with the synthesized trace as ICL context.
    """
    domain   = test_samples[0].get("domain", "logical") if test_samples else "logical"
    system   = SYSTEM_MATH if domain == "math" else SYSTEM_LOGICIAN

    test_ids = {s["sample_id"] for s in test_samples}
    coros, meta = [], []

    async def _passthrough(value: Optional[str]) -> Optional[str]:
        """Return a value directly without an LLM call."""
        return value

    for s in test_samples:
        sample_system = system

        # E-direct mode: extract label directly from synthesized trace, no LLM call.
        # Must be checked BEFORE fixed_shots branching so it works regardless of --fixed_shots_file.
        if name.startswith("E") and s["sample_id"] in trace_pool and e_direct_extract:
            synth_trace = trace_pool[s["sample_id"]]["text"]
            extracted   = extract_label(synth_trace)
            coros.append(_passthrough(extracted))
            meta.append({
                "sample":             s,
                "example_avg_steps":  float(count_steps(synth_trace)),
                "example_avg_tokens": float(count_tokens(synth_trace)),
                "e_direct":           True,
                "e_trace_steps":      count_steps(synth_trace),
                "e_trace_tokens":     count_tokens(synth_trace),
                "e_raw":              synth_trace,
            })
            continue  # skip the LLM call below

        if fixed_shots is not None:
            # Build prompt from fixed examples (used for Settings B/C/D and E-LLM fallback)
            parts = []
            for i, ex in enumerate(fixed_shots, 1):
                trace   = ex["trace"]
                trimmed = trace[:3000] + ("..." if len(trace) > 3000 else "")
                parts += [
                    f"--- Example {i} ---",
                    f"Problem:\n{ex['problem_text']}",
                    f"Reasoning:\n{trimmed}",
                    f"Answer: {ex['ground_truth']}",
                    "",
                ]

            if name.startswith("E") and s["sample_id"] in trace_pool:
                # E-LLM mode (original): provide synthesized trace as ICL context
                synth_trace  = trace_pool[s["sample_id"]]["text"]
                trimmed_synth = synth_trace[:6000] + ("..." if len(synth_trace) > 6000 else "")
                parts += [
                    "--- New Problem ---",
                    f"Problem:\n{s['problem_text']}",
                    f"Pre-computed reasoning chain (RKG-synthesized):",
                    trimmed_synth,
                    "",
                    "Based ONLY on the reasoning chain above, what is the conclusion?",
                    "Answer (output ONLY __PROVED__ or __DISPROVED__, no other text):",
                ]
                avg_steps  = float(count_steps(synth_trace))
                avg_tokens = float(count_tokens(synth_trace))
            else:
                parts += [
                    "--- New Problem ---",
                    f"Problem:\n{s['problem_text']}",
                    "Reasoning: [your step-by-step reasoning]\nAnswer:",
                ]
                avg_steps  = float(np.mean([count_steps(ex["trace"]) for ex in fixed_shots]))
                avg_tokens = float(np.mean([count_tokens(ex["trace"]) for ex in fixed_shots]))
            user = "\n".join(parts)
        else:
            examples = select_examples(train_pool, trace_pool, s["problem_text"],
                                       shots, test_ids, retrieval, seed)
            user     = build_icl_prompt(s["problem_text"], examples, trace_pool, domain=domain)
            avg_steps  = float(np.mean([trace_pool[e["sample_id"]]["n_steps"]  for e in examples])) if examples else 0.0
            avg_tokens = float(np.mean([trace_pool[e["sample_id"]]["n_tokens"] for e in examples])) if examples else 0.0
        coros.append(call_llm(session, semaphore, sample_system, user,
                              model, api_key, base_url))
        meta.append({
            "sample":             s,
            "example_avg_steps":  avg_steps,
            "example_avg_tokens": avg_tokens,
        })

    responses = await asyncio.gather(*coros)
    predictions = []
    for m, r in zip(meta, responses):
        s = m["sample"]
        if m.get("e_direct"):
            # E-direct: prediction came from trace extraction, not LLM
            predicted      = r  # already the extracted label or None
            raw_response   = m["e_raw"]
            resp_steps     = m["e_trace_steps"]
            resp_tokens    = m["e_trace_tokens"]
        else:
            predicted    = (normalise_math_answer(extract_math_answer(r))
                            if domain == "math" else extract_label(r))
            raw_response = r or ""
            resp_steps   = count_steps(r or "")
            resp_tokens  = count_tokens(r or "")
        predictions.append({
            "sample_id":           s["sample_id"],
            "source_dataset":      s["source_dataset"],
            "ground_truth":        s["ground_truth"],
            "predicted":           predicted,
            "raw_response":        raw_response,
            "response_steps":      resp_steps,
            "response_tokens":     resp_tokens,
            "example_avg_steps":   m["example_avg_steps"],
            "example_avg_tokens":  m["example_avg_tokens"],
            "domain":              domain,
        })
    return predictions

# ---------------------------------------------------------------------------
# export — the trace each cell reports, one JSONL per benchmark and backbone
# ---------------------------------------------------------------------------
# A cell's reported number does not always come from `synthesized.json`: the
# settings differ per configuration and each writes its own file, so reading
# a run directory and taking the obvious name gives the wrong trace for most
# cells. This map is the one place that says which file each cell reports
# from; the export fails rather than falling back if that file is missing.
#
# (dataset, model) -> (cell directory under --runs, file, how it was run).
# The 2026-09-23 run: the atomic synthesizer for every cell; ProofWriter goes
# through the closed-world recheck with its proof search written back as
# steps; gpt's ProofWriter trace is then restated with its answer pinned;
# gemini's mathematics has the adjudicator's derivation written back as steps
# and opens with the problem's own statement of the goal.
REPORTED_CELLS = {
    ("FLD", "gemini-3.1-flash-lite"):
        ("FLDgem", "synth", "prior_mode=verify"),
    ("FLD", "gpt-5.4-nano"):
        ("FLDgpt", "synth", "prior_mode=follow"),
    ("ProofWriter", "gemini-3.1-flash-lite"):
        ("PWgem", "cwa_resolve",
         "prior_mode=verify, weight_by=gold_depth, cwa_recheck (integrated), cwa_resolve"),
    ("ProofWriter", "gpt-5.4-nano"):
        ("PWgpt", "polish",
         "prior_mode=verify, weight_by=gold_depth, cwa_recheck (integrated), cwa_resolve, polish two3"),
    ("OmniMATH", "gemini-3.1-flash-lite"):
        ("OMgem", "adj_goal", "prior_mode=verify, adjudication (integrated), goal stated"),
    ("OmniMATH", "gpt-5.4-nano"):
        ("OMgpt", "synth", "prior_mode=follow"),
    ("OlympiadBench", "gemini-3.1-flash-lite"):
        ("OBgem", "adj_goal", "prior_mode=verify, adjudication (integrated), goal stated"),
    ("OlympiadBench", "gpt-5.4-nano"):
        ("OBgpt", "synth", "prior_mode=follow"),
}

_STEP_HEAD_RE = re.compile(r"(?m)^\s*Step\s*\d+\s*[:.\-]\s*")


def main_export(argv) -> None:
    """Write the trace CRAFT reports for each cell to <out_dir>/<model>/<dataset>_Output.jsonl.

    Each line is one sample: sample_id, source_dataset, domain, model, setting,
    ground_truth, predicted, n_steps, trace. `predicted` is re-derived from the
    trace text with the reader `score` uses, so a line cannot disagree with the
    reported table; a sample Module III produced no trace for is left out, as
    `score` leaves it out, so the line count is the n its cell reports.

    The trace is written with its restatements removed (dedup_trace), which
    leaves every sentence that states an answer alone and checks the rewrite
    against the cell's own reader, so `predicted` cannot move.
    """
    parser = argparse.ArgumentParser(
        prog="evaluate_accuracy.py export", description=main_export.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="craft_runs/full500_v4/traces",
                        help="Directory holding one folder per cell (see REPORTED_CELLS)")
    parser.add_argument("--out_dir", default="CRAFT_results/Raw_Output")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                           / "framework" / "module3_topology_guided_synthesis"))
    from dedup_trace import dedup_trace  # noqa: E402  (pulls in sympy; only export needs it)

    readers = {"FLD": extract_label, "ProofWriter": extract_label,
               "OmniMATH": extract_math_answer, "OlympiadBench": extract_math_answer}
    runs = Path(resolve_input(args.runs))
    out_dir = Path(resolve_output(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for (ds, model), (run, stem, setting) in sorted(REPORTED_CELLS.items()):
        src = runs / run / f"{stem}.json"
        if not src.exists():
            missing.append(str(src))
            continue
        rows = LOADERS["synthesized"](src)
        metrics = compute_metrics(rows)
        shrunk = total = dropped = 0
        out = out_dir / model / f"{ds}_Output.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        reader = readers[ds]
        with out.open("w", encoding="utf-8") as fh:
            for r in rows:
                t = r["traces"][0]
                original = t["text"] or ""
                if not original.strip():
                    dropped += 1
                    continue
                text = dedup_trace(original, extractor=reader)
                n_steps = len(_STEP_HEAD_RE.findall(text)) or t["n_steps"]
                shrunk += len(original) - len(text)
                total += len(original)
                fh.write(json.dumps({
                    "sample_id": r["sample_id"],
                    "source_dataset": r["source_dataset"],
                    "domain": r["domain"],
                    "model": model,
                    "setting": setting,
                    "ground_truth": r["ground_truth"],
                    "predicted": r["predicted"],
                    "n_steps": n_steps,
                    "trace": text,
                }, ensure_ascii=False) + "\n")
        print(f"  {model:<22} {ds:<14} {len(rows) - dropped:>4} traces  "
              f"acc {100*metrics['accuracy']:5.1f}  steps {metrics['avg_steps']:4.1f}  "
              f"trimmed {100*shrunk/max(total,1):4.1f}%"
              + (f"  dropped {dropped}" if dropped else "") + f"  -> {out}")

    if missing:
        raise SystemExit("These cells have no reported trace file:\n  "
                         + "\n  ".join(missing))


def main() -> None:
    """Score one pipeline output (`score`), or export the traces the cells report (`export`).

    The Settings A-E ablation that used to live here is gone. The paper's
    ablation is the five named rows, which evaluation/Ablation_Study/ablation_study.py
    builds from this module's loaders and metric, so there is one scorer and
    one place the ablation is defined.
    """
    import sys as _sys
    modes = {"score": main_score, "export": main_export}
    if len(_sys.argv) < 2 or _sys.argv[1] not in modes:
        print(__doc__)
        raise SystemExit(2)
    modes[_sys.argv[1]](_sys.argv[2:])


if __name__ == "__main__":
    main()
