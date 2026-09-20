#!/usr/bin/env python3
"""
ablation_study.py — the ablation table of §4, one row per removed component.

    Setting                      what it removes            read from
    ---------------------------------------------------------------------------
    CRAFT (full)                 nothing                    the run's synthesized trace
    w/o CRAFT                    the whole pipeline         a single-call run
    w/o RKG                      Module II's graph          --synthesis_strategy step_by_step
    w/o Synthesis                Module III                 vote over the cleaned traces
    w/o Filter & Synthesis       the filter and Module III  vote over the raw K traces
    w/o Weighted Edges Fusion    the lambda term in W(e)    build_rkg --edge_lambda 0
    Embedding Cosine Similarity  Jaccard in the edge weight an embedding-similarity run

Scoring is not reimplemented here: the rows are read with the same loaders and
scored with the same metric as the main table, so an ablation row and a main-table
cell can never disagree about what a run achieved. That also keeps the two vote
rows honest — the filtered file re-derives each trace's prediction from the text
that survived filtering, rather than reusing the label generation stored, which is
what makes "w/o Synthesis" differ from "w/o Filter & Synthesis" at all.

Three settings need their own run and are passed in with --variant NAME=PATH.
Settings that were not run are printed as absent, so a partial table cannot be
mistaken for a complete one.

Usage:
    python ablation_study.py --craft_dir craft_runs/olympiad_gemini \\
        --variant "w/o RKG=craft_runs/olympiad_gemini/synthesized_step_by_step.json" \\
        --variant "w/o Weighted Edges Fusion=craft_runs/olympiad_gemini_lam0/synthesized.json" \\
        --zero_shot craft_runs/olympiad_gemini/zero_shot.json \\
        --output ablation/olympiad_gemini.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from config import resolve_input, resolve_output, run_model
except ImportError:
    resolve_input = resolve_output = Path

    def run_model(*paths):  # noqa: D103
        return "unknown-model"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "evaluation" / "label_prediction"))
from evaluate_direct_accuracy import LOADERS, compute_metrics

FULL = "CRAFT (full)"
VARIANT_ROWS = ("w/o RKG", "w/o Weighted Edges Fusion", "Embedding Cosine Similarity")
ROW_ORDER = (FULL, "w/o CRAFT", "w/o RKG", "w/o Synthesis", "w/o Filter & Synthesis",
             "w/o Weighted Edges Fusion", "Embedding Cosine Similarity")


def find_one(run_dir: Path, pattern: str) -> Optional[Path]:
    hits = sorted(run_dir.glob(pattern))
    return hits[0] if hits else None


def score(path: Path, source: str) -> Dict[str, Any]:
    """One row, scored exactly as evaluate_direct_accuracy would score it."""
    metrics = compute_metrics(LOADERS[source](path))
    return {"accuracy": round(100.0 * metrics["accuracy"], 1),
            "macro_f1": metrics["macro_f1"],
            "avg_steps": metrics.get("avg_steps"),
            "n_samples": metrics.get("n_samples", metrics.get("valid_total")),
            "source": source, "path": str(path)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--craft_dir", required=True,
                    help="The full CRAFT run: k_traces, cleaned traces, synthesized trace")
    ap.add_argument("--zero_shot", default=None,
                    help="'w/o CRAFT': a synthesized-schema file from a single-call run "
                         "over the same samples. Without it that row is reported as absent")
    ap.add_argument("--variant", action="append", default=[], metavar="NAME=PATH",
                    help=f"Synthesis output for one of: {', '.join(VARIANT_ROWS)}. Repeatable")
    ap.add_argument("--synth_file", default=None,
                    help="The full run's synthesis output (default: synthesized*.json in --craft_dir)")
    ap.add_argument("--output", default=None,
                    help="Write the table as JSON here. Default: "
                         "CRAFT_results/ablation_study/<model>/ablation_study.json under the results "
                         "root, with the model read from the run's own metadata")
    args = ap.parse_args()

    run_dir = Path(resolve_input(args.craft_dir))
    k_path = find_one(run_dir, "k_traces_*_samples.json")
    if k_path is None:
        raise FileNotFoundError(f"No k_traces_*_samples.json in {run_dir}")
    synth_path = (Path(resolve_input(args.synth_file)) if args.synth_file
                  else find_one(run_dir, "synthesized*.json"))
    if synth_path is None:
        raise FileNotFoundError(f"No synthesized*.json in {run_dir}")
    cleaned_path = (find_one(run_dir, "cleaned_traces_rkg.json")
                    or find_one(run_dir, "cleaned*.json"))

    rows: Dict[str, Dict[str, Any]] = {
        FULL: score(synth_path, "synthesized"),
        "w/o Filter & Synthesis": score(k_path, "k_traces"),
    }
    if cleaned_path:
        rows["w/o Synthesis"] = score(cleaned_path, "cleaned")
    if args.zero_shot:
        rows["w/o CRAFT"] = score(Path(resolve_input(args.zero_shot)), "synthesized")
    for spec in args.variant:
        if "=" not in spec:
            raise SystemExit(f"--variant takes NAME=PATH, got {spec!r}")
        name, raw = (s.strip() for s in spec.split("=", 1))
        if name not in VARIANT_ROWS:
            raise SystemExit(f"Unknown variant {name!r}; expected one of {VARIANT_ROWS}")
        rows[name] = score(Path(resolve_input(raw)), "synthesized")

    full_acc = rows[FULL]["accuracy"]
    print()
    print(f"  {'Ablation Setting':<32} {'Acc(%)':>8} {'d(%)':>8} {'steps':>7}")
    print("  " + "-" * 58)
    for name in ROW_ORDER:
        row = rows.get(name)
        if row is None:
            print(f"  {name:<32} {'not run':>8}")
            continue
        delta = "" if name == FULL else f"{row['accuracy'] - full_acc:+.1f}"
        steps = f"{row['avg_steps']:.1f}" if row.get("avg_steps") else "-"
        print(f"  {name:<32} {row['accuracy']:>8.1f} {delta:>8} {steps:>7}")
    print()
    absent = [n for n in ROW_ORDER if n not in rows]
    if absent:
        print("  Not run: " + ", ".join(absent))
        print("  w/o RKG: synthesize with --synthesis_strategy step_by_step.")
        print("  w/o Weighted Edges Fusion: build_rkg --edge_lambda 0, then synthesize.")
        print("  w/o CRAFT: pass --zero_shot.")

    model = run_model(synth_path, k_path)
    out = Path(resolve_output(args.output
                              or f"CRAFT_results/ablation_study/{model}/ablation_study.json"))
    out.write_text(json.dumps(
        {"craft_dir": str(run_dir), "full_accuracy": full_acc,
         "settings": {n: {**rows[n],
                          "delta": None if n == FULL
                          else round(rows[n]["accuracy"] - full_acc, 1)}
                      for n in ROW_ORDER if n in rows},
         "not_run": absent}, indent=2), encoding="utf-8")
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
