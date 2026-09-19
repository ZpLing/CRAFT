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
  - file:     path to the .summary.json produced by prmbench_evaluate_verifier.py

Usage:
    # Show current summary table
    python prmbench_results_summary.py

    # Register a new run and append to the master JSON
    python prmbench_results_summary.py --add \
        --model gpt-4.1-mini \
        --api api_a \
        --n_per_dim N \
        --summary_file prmbench_<N>_<model>_results.summary.json

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

RESULTS_DIR  = RESULTS_ROOT / "prmbench" / "results"
MASTER_FILE  = RESULTS_DIR / "prmbench_master_results.json"

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

def add_run(
    model: str,
    api: str,
    n_per_dim: int,
    summary_file: str,
    note: str = "",
) -> None:
    path = Path(summary_file)
    # If not absolute, look inside the results root's prmbench/results/ first
    if not path.is_absolute() and not path.exists():
        candidate = RESULTS_DIR / path
        if candidate.exists():
            path = candidate
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")

    with open(path, encoding="utf-8") as f:
        summary = json.load(f)

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
        bl  = d.get("blind", {})
        record["dims"][dim] = {
            "n_items":      n,
            "with_answer":  {k: wa.get(k) for k, _ in METRICS},
            "blind":        {k: bl.get(k) for k, _ in METRICS},
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
            subhdr += f"  {'w/ answer':>{col_w}}  {'blind':>{col_w}}"
        print(subhdr)
        print("  " + "-" * (22 + len(records) * (col_w * 2 + 4)))

        for key, label in METRICS:
            row = f"  {label:<22}"
            for r in records:
                d  = r["dims"].get(dim, {})
                wa = d.get("with_answer", {}).get(key)
                bl = d.get("blind", {}).get(key)
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
    parser.add_argument("--summary_file", default=None,
                        help="Path to .summary.json from prmbench_evaluate_verifier.py")
    parser.add_argument("--note",         default="", help="Optional note for this run")
    args = parser.parse_args()

    if args.add:
        if not args.model or not args.summary_file:
            parser.error("--add requires --model and --summary_file")
        add_run(
            model        = args.model,
            api          = args.api or "unknown",
            n_per_dim    = args.n_per_dim,
            summary_file = args.summary_file,
            note         = args.note,
        )

    records = load_master()
    print_table(records)


if __name__ == "__main__":
    main()
