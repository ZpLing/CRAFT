#!/usr/bin/env python3
"""
hyperparameter_sensitivity.py — how accuracy and the size of the consensus RKG move with K.

For each K, reads a finished CRAFT run and records three numbers: label-prediction
accuracy from its synthesized traces, and the mean node and edge count of the
consensus RKG. The paper's observation is that edges grow faster than nodes, and
that RKG size tracks accuracy.

One run per K is needed. They are named on the command line, or discovered from a
directory whose subdirectories end in the K they were run with:

    python hyperparameter_sensitivity.py --runs 2=craft_runs/fld_k2 3=craft_runs/fld_k3 ...
    python hyperparameter_sensitivity.py --glob "craft_runs/fld_k*" --label FLD

Writes a JSON the paper's area chart renders from. Accuracy is scored exactly as
the main table scores it, so a point here and a cell there cannot disagree.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from config import RESULTS_ROOT, resolve_input, resolve_output, run_model
except ImportError:
    RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results"
    resolve_input = resolve_output = Path

    def run_model(*paths):  # noqa: D103
        return "unknown-model"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "evaluation" / "label_prediction"))
try:
    from evaluate_direct_accuracy import normalise_math_answer
except ImportError:
    def normalise_math_answer(s):  # noqa: D103
        return (s or "").strip() or None

_LABEL_RE = re.compile(r"(__PROVED__|__DISPROVED__)", re.IGNORECASE)


def load_records(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return raw.get("results", raw) if isinstance(raw, dict) else raw


def normalise(value: Any, domain: str) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if domain == "math":
        return normalise_math_answer(text)
    hit = _LABEL_RE.search(text)
    return hit.group(1).upper() if hit else text.upper()


def measure(run_dir: Path, domain: str) -> Dict[str, Any]:
    """Accuracy and mean consensus-RKG size for one run."""
    k_files = sorted(run_dir.glob("k_traces_*_samples.json"))
    synth_files = sorted(run_dir.glob("synthesized*.json"))
    rkg_files = sorted(run_dir.glob("rkg*.json"))
    if not k_files or not synth_files:
        raise FileNotFoundError(f"{run_dir} needs both k_traces_*_samples.json "
                                f"and synthesized*.json")

    truth = {r.get("sample_id"): r.get("target_answer") for r in load_records(k_files[0])}
    correct = scored = missing = 0
    for r in load_records(synth_files[0]):
        t = normalise(truth.get(r.get("sample_id")), domain)
        if t is None:
            continue
        p = normalise(r.get("pred_label"), domain)
        if p is None:
            missing += 1
            continue
        scored += 1
        correct += (p == t)

    # Size is the consensus RKG's, not the per-trace graphs': it is the graph
    # Module III walks, so it is the one whose growth can track accuracy.
    nodes = edges = graphs = 0
    if rkg_files:
        for r in load_records(rkg_files[0]):
            consensus = r.get("consensus_rkg") or r.get("consensus_dag") or {}
            if not consensus:
                continue
            graphs += 1
            nodes += len(consensus.get("nodes", []))
            edges += len(consensus.get("edges", []))

    return {
        "accuracy": round(100.0 * correct / scored, 2) if scored else None,
        "n_scored": scored,
        "n_missing": missing,
        "rkg_nodes": round(nodes / graphs, 2) if graphs else None,
        "rkg_edges": round(edges / graphs, 2) if graphs else None,
        "n_graphs": graphs,
        "run_dir": str(run_dir),
    }


def discover(pattern: str) -> List[Tuple[int, Path]]:
    """Runs whose directory name ends in the K they used, e.g. fld_k2 ... fld_k10."""
    base = Path(pattern)
    root = base.parent if base.parent != Path("") else Path(".")
    if not root.is_absolute() and not root.exists():
        root = RESULTS_ROOT / root
    found: List[Tuple[int, Path]] = []
    for path in sorted(root.glob(base.name)):
        if not path.is_dir():
            continue
        hit = re.search(r"(\d+)$", path.name)
        if hit:
            found.append((int(hit.group(1)), path))
    return sorted(found)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--runs", nargs="+", metavar="K=DIR",
                     help="One run directory per K, e.g. 2=craft_runs/fld_k2")
    src.add_argument("--glob", dest="pattern",
                     help="Directory pattern whose names end in K, e.g. 'craft_runs/fld_k*'")
    ap.add_argument("--label", default="FLD", help="Dataset name for the output and the plot")
    ap.add_argument("--domain", default="logical", choices=["logical", "math"])
    ap.add_argument("--output", default=None,
                    help="Write the measurements as JSON here. Default: "
                         "CRAFT_evaluation_results/hyperparameter_sensitivity/<model>/hyperparameter_sensitivity_<label>.json "
                         "results root, with the model read from the run's own metadata")
    args = ap.parse_args()

    if args.runs:
        pairs = []
        for spec in args.runs:
            if "=" not in spec:
                raise SystemExit(f"--runs takes K=DIR, got {spec!r}")
            k, raw = spec.split("=", 1)
            pairs.append((int(k), Path(resolve_input(raw.strip()))))
        pairs.sort()
    else:
        pairs = discover(args.pattern)
        if not pairs:
            raise SystemExit(f"No run directories matched {args.pattern!r}")

    print(f"\n  {args.label}   ({len(pairs)} values of K)")
    print(f"  {'K':>3} {'Acc(%)':>8} {'nodes':>8} {'edges':>8} {'n':>6} {'missing':>8}")
    print("  " + "─" * 46)
    series: Dict[int, Dict[str, Any]] = {}
    for k, run_dir in pairs:
        m = measure(run_dir, args.domain)
        series[k] = m
        acc = f"{m['accuracy']:.2f}" if m["accuracy"] is not None else "—"
        nod = f"{m['rkg_nodes']:.2f}" if m["rkg_nodes"] is not None else "—"
        edg = f"{m['rkg_edges']:.2f}" if m["rkg_edges"] is not None else "—"
        print(f"  {k:>3} {acc:>8} {nod:>8} {edg:>8} {m['n_scored']:>6} {m['n_missing']:>8}")
    print()

    model = run_model(*(d for _, d in pairs))
    out = Path(resolve_output(
        args.output or f"CRAFT_evaluation_results/hyperparameter_sensitivity/{model}/hyperparameter_sensitivity_{args.label}.json"))
    out.write_text(json.dumps({"label": args.label, "domain": args.domain,
                               "series": {str(k): v for k, v in series.items()}},
                              indent=2), encoding="utf-8")
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
