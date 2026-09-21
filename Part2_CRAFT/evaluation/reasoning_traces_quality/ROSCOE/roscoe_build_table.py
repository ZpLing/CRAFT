#!/usr/bin/env python3
"""
Build the raw-CoT vs CRAFT ROSCOE table from roscoe_score.py's summary.

Reads the evaluation_results.json that roscoe_score.py writes into an export
directory and reports the three metrics the paper's ROSCOE table carries —
Grammar (higher is better), Rep-Step and Rep-Word (lower is better) — as the
percentage-point change from raw CoT to CRAFT.

Output: a markdown table on stdout, and a LaTeX body with --latex_out.

A scoring run's evaluation_results.json is where the numbers arrive, not where
they are kept: --results_dir files them under one directory per model, as the
run's two sides,

    <results_dir>/<model>/ROSCOE_CRAFT.json
    <results_dir>/<model>/ROSCOE_Raw_CoT.json

which is also what this reads when --summaries is absent, so a table can be
rebuilt from the kept results without the scoring run's directories.

Example:
    python roscoe_build_table.py \\
        --summaries "GPT-5.4-nano:roscoe_craft/nano/evaluation_results.json" \\
                    "Gemini-3.1-flash-lite:roscoe_craft/gemini/evaluation_results.json" \\
        --results_dir CRAFT_results/reasoning_traces_quality/ROSCOE \\
        --latex_out roscoe_craft/roscoe_craft_table.tex

    python roscoe_build_table.py \\
        --results_dir CRAFT_results/reasoning_traces_quality/ROSCOE
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
try:
    from config import resolve_input, resolve_output
except ImportError:
    resolve_input = resolve_output = Path


# (column header, key in ROSCOE's score dict, higher-is-better)
METRICS: List[Tuple[str, str, bool]] = [
    ("Grammar",  "grammar_step",    True),
    ("Rep-Step", "repetition_step", False),
    ("Rep-Word", "repetition_word", False),
]

DATASET_LABEL = {"FLD": "FLD", "ProofWriter": "ProofWriter",
                 "OmniMATH": "Omni-MATH", "OlympiadBench": "OlympiadBench"}
DATASET_ORDER = ["FLD", "ProofWriter", "OmniMATH", "OlympiadBench"]

# The kept results: one file per side, under a directory named for the model.
SIDE_FILE = {"raw": "ROSCOE_Raw_CoT.json", "craft": "ROSCOE_CRAFT.json"}
SIDE_TRACE = {
    "raw": "the first of the K candidate traces",
    "craft": ("the trace Module III synthesizes from the consensus RKG, "
              "after deduplication"),
}
SCORES_NOTE = "ROSCOE's own, all thirteen higher-is-better"


def model_dir(label: str) -> str:
    """The directory a model's results live in.

    The rest of results/ names a model's directory by its API id in lower case
    (gpt-5.4-nano, gemini-3.1-flash-lite), so a table label of any casing has to
    land in the same place as the traces it scored.
    """
    return "-".join(label.strip().lower().split())


def write_results(results_dir: Path, label: str, summary: Dict[str, Any]) -> List[Path]:
    """File one model's scoring run as the two kept per-side results."""
    out_dir = results_dir / model_dir(label)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored = summary.get("datasets", {})
    order = DATASET_ORDER + sorted(set(scored) - set(DATASET_ORDER))

    written = []
    for side, filename in SIDE_FILE.items():
        datasets = {}
        for ds in order:
            block = scored.get(ds)
            if not block:
                continue
            metrics = (block.get("metrics", {}) or {}).get(side) or {}
            if not metrics:
                continue
            datasets[ds] = {
                "n_traces": (block.get("n_traces", {}) or {}).get(side),
                "metrics": {k: metrics[k] for k in sorted(metrics)},
            }
        if not datasets:
            continue
        path = out_dir / filename
        path.write_text(json.dumps({
            "model": model_dir(label),
            "side": side if side == "craft" else "raw_cot",
            "trace": SIDE_TRACE[side],
            "scores": SCORES_NOTE,
            "datasets": datasets,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(path)
    return written


def read_results(results_dir: Path) -> List[Tuple[str, Dict[str, Any]]]:
    """Rebuild the scoring-run shape from the kept results, one entry per model.

    Reading gives back what --summaries would have loaded, so everything below
    stays written against the one shape.
    """
    loaded: List[Tuple[str, Dict[str, Any]]] = []
    for model_path in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        per_dataset: Dict[str, Any] = {}
        for side, filename in SIDE_FILE.items():
            path = model_path / filename
            if not path.exists():
                continue
            doc = json.loads(path.read_text(encoding="utf-8"))
            for ds, block in doc.get("datasets", {}).items():
                cell = per_dataset.setdefault(ds, {"n_traces": {}, "metrics": {}})
                cell["metrics"][side] = block.get("metrics", {})
                cell["n_traces"][side] = block.get("n_traces")
        if per_dataset:
            loaded.append((model_path.name, {"datasets": per_dataset}))
    if not loaded:
        raise SystemExit(f"No <model>/{SIDE_FILE['craft']} under {results_dir}")
    return loaded


def load_summary(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def deltas(summary: Dict[str, Any], dataset: str) -> Dict[str, Optional[float]]:
    """craft - raw, in percentage points, for each reported metric."""
    block = summary.get("datasets", {}).get(dataset)
    if not block:
        return {key: None for _, key, _ in METRICS}
    raw = block.get("metrics", {}).get("raw", {}) or {}
    craft = block.get("metrics", {}).get("craft", {}) or {}
    out: Dict[str, Optional[float]] = {}
    for _, key, _ in METRICS:
        a, b = raw.get(key), craft.get(key)
        out[key] = None if a is None or b is None else round(100.0 * (b - a), 1)
    return out


def n_traces(summary: Dict[str, Any], dataset: str) -> Optional[int]:
    counts = summary.get("datasets", {}).get(dataset, {}).get("n_traces", {})
    vals = [v for v in counts.values() if v]
    return min(vals) if vals else None


def fmt(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{'+' if value > 0 else ''}{value:.1f}%"


def parse_spec(spec: str) -> Tuple[str, Path]:
    if ":" not in spec:
        raise ValueError(f"--summaries takes 'Label:path', got {spec!r}")
    label, raw_path = spec.split(":", 1)
    return label.strip(), Path(resolve_input(raw_path.strip()))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summaries", nargs="+", default=None,
                    help="One 'Model label:evaluation_results.json' per model, in table order")
    ap.add_argument("--results_dir", default=None,
                    help="Where the kept per-model results live. With --summaries, "
                         "the scoring runs are filed here first; without it, the "
                         "table is built from what is already here")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="Datasets to include (default: the benchmark's four, those present)")
    ap.add_argument("--latex_out", default=None,
                    help="Also write the LaTeX table body here")
    args = ap.parse_args()

    if not args.summaries and not args.results_dir:
        raise SystemExit("Give --summaries, or --results_dir to read kept results")

    if args.summaries:
        loaded = [(label, load_summary(path))
                  for label, path in map(parse_spec, args.summaries)]
        if args.results_dir:
            kept = Path(resolve_output(args.results_dir))
            for label, summary in loaded:
                for path in write_results(kept, label, summary):
                    print(f"kept → {path}")
    else:
        loaded = read_results(Path(resolve_input(args.results_dir)))

    # Datasets the summaries hold that DATASET_ORDER does not name still belong in
    # the table: a dataset added later should show up rather than disappear into a
    # list it was never added to.
    scored = {d for _, s in loaded for d in s.get("datasets", {})}
    wanted = args.datasets or (DATASET_ORDER + sorted(scored - set(DATASET_ORDER)))
    present = [d for d in wanted if d in scored]
    if not present:
        raise SystemExit("None of the requested datasets appear in these summaries")

    headers = [name for name, _, _ in METRICS]
    print()
    print(f"| {'Model':<14} | {'Dataset':<10} | " +
          " | ".join(f"{h:<8}" for h in headers) + " | n |")
    print(f"|{'-'*16}|{'-'*12}|" + "|".join("-" * 10 for _ in headers) + "|---|")
    for label, summary in loaded:
        for ds in present:
            d = deltas(summary, ds)
            n = n_traces(summary, ds)
            cells = " | ".join(f"{fmt(d[key]):<8}" for _, key, _ in METRICS)
            print(f"| {label:<14} | {DATASET_LABEL.get(ds, ds):<10} | {cells} | {n or '—'} |")
    print()
    print("Grammar higher is better; Rep-Step and Rep-Word lower is better. "
          "Values are CRAFT minus raw CoT, in percentage points.")

    if args.latex_out:
        lines: List[str] = []
        for label, summary in loaded:
            lines.append(f"  \\multirow{{{len(present)}}}{{*}}{{\\shortstack{{{label}}}}}")
            for i, ds in enumerate(present):
                d = deltas(summary, ds)
                cells = " & ".join(
                    "—" if d[key] is None
                    else f"${'+' if d[key] > 0 else '-'}${abs(d[key]):.1f}\\%"
                    for _, key, _ in METRICS)
                lead = "   " if i else "   "
                lines.append(f"{lead} & {DATASET_LABEL.get(ds, ds)} & {cells} \\\\")
            lines.append("  \\midrule")
        if lines and lines[-1] == "  \\midrule":
            lines[-1] = "  \\bottomrule"
        out = Path(resolve_output(args.latex_out))
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nLaTeX → {out}")


if __name__ == "__main__":
    main()
