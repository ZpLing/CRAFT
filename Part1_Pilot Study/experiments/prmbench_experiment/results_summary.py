#!/usr/bin/env python3
"""
prmbench_results_summary.py
----------------------------
Aggregates PRMBench evaluation results across multiple models into a single
comparison table.

Each entry in the registry corresponds to one run:
  - model:    model name used
  - api:      which API endpoint was used
  - n_per_dim: samples per dimension (simplicity / soundness / sensitivity)
  - file:     a run's results, named by dimension (both settings are read)

Usage:
    # Show current summary table
    python prmbench_results_summary.py

    # Register a new run and append to the master JSON
    python prmbench_results_summary.py --add \
        --model gpt-4.1-mini \
        --api api_a \
        --n_per_dim N \
        --results_base <dimension>

Output files:
    prmbench_master_results.json   — machine-readable master record
    (stdout)                        — formatted comparison table
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from config import RESULTS_ROOT
except ImportError:
    RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results"

# One directory per model (<results>/<model>/prmbench/); the master index spans
# all of them, so it sits at the results root rather than inside any one model.
MASTER_FILE  = RESULTS_ROOT / "prmbench_master_results.json"

DIMS    = ["simplicity", "soundness", "sensitivity", "total"]
METRICS = [
    ("total_step_acc",   "Total Step Acc"),
    ("correct_step_acc", "Correct Step Acc"),
    ("wrong_step_acc",   "Wrong Step Acc"),
    ("first_error_acc",  "First Error Acc"),
    ("precision",        "Precision"),
    ("recall",           "Recall"),
    ("f1",               "F1"),
    ("negative_f1",      "Negative F1"),
]

DIM_INFO = {
    "simplicity":  "redundency / circular",
    "soundness":   "counterfactual / step_contradiction / domain_inconsistency / confidence",
    "sensitivity": "missing_condition / deception / multi_solutions",
    "total":       "all 3 dimensions",
}


# ---------------------------------------------------------------------------
# Load / save master file
# ---------------------------------------------------------------------------

def load_master() -> List[Dict]:
    if MASTER_FILE.exists():
        with open(MASTER_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_master(records: List[Dict]) -> None:
    MASTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MASTER_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Master results saved → {MASTER_FILE}")


# ---------------------------------------------------------------------------
# Add a new run
# ---------------------------------------------------------------------------

def summarize_run(model: str, dim_files_base: Path) -> Dict[str, Any]:
    """Recompute a run's summary from its two per-setting results files.

    The verifier writes only per-item scores; everything an aggregate says is
    derived from them, so it is derived here rather than stored twice and left to
    drift apart.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "prmbench_verifier", Path(__file__).resolve().parent / "prmbench_evaluate_verifier.py")
    V = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(V)

    sides = {}
    for setting in ("with_answer", "wout_answer"):
        path = dim_files_base.with_name(f"{dim_files_base.name}_{setting}.jsonl")
        if not path.exists():
            raise FileNotFoundError(f"Missing results file: {path}")
        with path.open() as f:
            sides[setting] = {json.loads(l)["idx"]: json.loads(l) for l in f if l.strip()}

    records = []
    for idx, wa in sides["with_answer"].items():
        bl = sides["wout_answer"].get(idx)
        if bl is None:
            continue
        records.append({"idx": idx, "classification": wa["classification"],
                        "error_steps": wa["error_steps"], "n_steps": wa["n_steps"],
                        "with_answer": wa, "wout_answer": bl})
    return V.build_summary(records, model)


def add_run(
    model: str,
    api: str,
    n_per_dim: int,
    results_base: str,
    note: str = "",
) -> None:
    base = Path(results_base)
    if not base.is_absolute() and not base.exists():
        base = RESULTS_ROOT / model / "prmbench" / base
    summary = summarize_run(model, base)

    record: Dict[str, Any] = {
        "model":      model,
        "api":        api,
        "n_per_dim":  n_per_dim,
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M"),
        "note":       note,
        "dims":       {},
    }

    for dim in DIMS:
        d = summary.get(dim, {})
        n   = d.get("n_items", 0)
        wa  = d.get("with_answer", {})
        bl  = d.get("wout_answer", {})
        record["dims"][dim] = {
            "n_items":      n,
            "with_answer":  {k: wa.get(k) for k, _ in METRICS},
            "wout_answer":        {k: bl.get(k) for k, _ in METRICS},
        }

    records = load_master()
    records.append(record)
    save_master(records)
    print(f"Added: {model} ({api}) — {n_per_dim} samples/dim  [{record['timestamp']}]")


# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------

def _fmt(v: Optional[float]) -> str:
    if v is None or v == -1:
        return "   —  "
    return f"{v:.4f}"


def print_table(records: List[Dict]) -> None:
    if not records:
        print("No results registered yet.")
        return

    for dim in DIMS:
        print()
        print("=" * 90)
        print(f"  DIMENSION: {dim.upper()}  ({DIM_INFO.get(dim, '')})")
        print("=" * 90)

        # Header: each model gets two columns (A / B)
        model_labels = [f"{r['model']} ({r['api']})" for r in records]
        col_w = 14
        header = f"  {'Metric':<22}"
        for label in model_labels:
            short = label[:12]
            header += f"  {'A:'+short:>{col_w}}  {'B:'+short:>{col_w}}"
        print(header)

        subhdr = f"  {'':22}"
        for r in records:
            subhdr += f"  {'w/ answer':>{col_w}}  {'wout_answer':>{col_w}}"
        print(subhdr)
        print("  " + "-" * (22 + len(records) * (col_w * 2 + 4)))

        for key, label in METRICS:
            row = f"  {label:<22}"
            for r in records:
                d  = r["dims"].get(dim, {})
                wa = d.get("with_answer", {}).get(key)
                bl = d.get("wout_answer", {}).get(key)
                row += f"  {_fmt(wa):>{col_w}}  {_fmt(bl):>{col_w}}"
            print(row)

        # n_items row
        n_row = f"  {'n_items':<22}"
        for r in records:
            n = r["dims"].get(dim, {}).get("n_items", 0)
            n_row += f"  {str(n):>{col_w}}  {'—':>{col_w}}"
        print(n_row)

    # Registry footer
    print()
    print("=" * 90)
    print("  REGISTERED RUNS")
    print("=" * 90)
    for i, r in enumerate(records, 1):
        note = f"  [{r['note']}]" if r.get("note") else ""
        print(f"  [{i}] {r['model']:<20} api={r['api']:<12} "
              f"n_per_dim={r['n_per_dim']}  {r['timestamp']}{note}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PRMBench multi-model results summary.")
    parser.add_argument("--add", action="store_true",
                        help="Register a new run into the master results")
    parser.add_argument("--model",        default=None)
    parser.add_argument("--api",          default=None, help="API used (anonymized label)")
    parser.add_argument("--n_per_dim",    type=int, default=None)
    parser.add_argument("--results_base", default=None,
                        help="A run's results named by dimension without the setting "
                             "suffix, e.g. simplicity (relative → <model>/prmbench/)")
    parser.add_argument("--note",         default="", help="Optional note for this run")
    args = parser.parse_args()

    if args.add:
        if not args.model or not args.results_base:
            parser.error("--add requires --model and --results_base")
        add_run(
            model        = args.model,
            api          = args.api or "unknown",
            n_per_dim    = args.n_per_dim,
            results_base = args.results_base,
            note         = args.note,
        )

    records = load_master()
    print_table(records)


if __name__ == "__main__":
    main()
