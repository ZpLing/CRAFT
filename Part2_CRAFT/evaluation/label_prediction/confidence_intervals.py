#!/usr/bin/env python3
"""
confidence_intervals.py — 95% Wilson score intervals for the main table.

Accuracy is a proportion of correct samples, so a Wilson score interval is the
right one: unlike the normal approximation it stays inside [0, 1] and does not
collapse to zero width near the ceiling, which is where several of these
accuracies sit. Reported as Acc(%) +/- half-width, the format the appendix uses.

The half-width depends on n, so n has to be the real number of scored samples,
not the number selected for the run. Passing a run's own output is the safe way
to get that: the accuracy and the denominator then come from the same file, and
are computed by the same loader as the main table. A literal `label=acc/n` is
accepted for a setting whose output lives elsewhere, and states the n it used.

Usage:
    python confidence_intervals.py \\
        --run "CRAFT (Ours)=craft_runs/fld_gemini/synthesized.json" \\
        --value "DICE=72.6/500" \\
        --output ci/fld_gemini.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from config import resolve_input, resolve_output
except ImportError:
    resolve_input = resolve_output = Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_direct_accuracy import LOADERS, compute_metrics

Z_95 = 1.959963984540054


def wilson(correct: int, n: int, z: float = Z_95) -> Tuple[float, float]:
    """Return (centre, margin) of the Wilson score interval, both in percent.

    The centre is pulled towards 0.5 from the observed proportion, which is what
    keeps the interval inside [0, 1] and stops it collapsing near the ceiling.
    The margin is the interval's half-width about that centre, and is what the
    appendix reports beside the observed accuracy.
    """
    if n <= 0:
        return 0.0, 0.0
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return 100 * centre, 100 * margin


def half_width(correct: int, n: int) -> Tuple[float, float]:
    """(observed accuracy, Wilson half-width), both in percent."""
    if n <= 0:
        return 0.0, 0.0
    _, margin = wilson(correct, n)
    return 100.0 * correct / n, margin


def from_run(path: Path, source: str) -> Tuple[int, int]:
    """(correct, n) for a run, scored exactly as the main table scores it."""
    metrics = compute_metrics(LOADERS[source](path))
    n = metrics.get("n_samples") or metrics.get("valid_total") or 0
    return int(round(metrics["accuracy"] * n)), int(n)


def parse_spec(spec: str, what: str) -> Tuple[str, str]:
    if "=" not in spec:
        raise SystemExit(f"--{what} takes LABEL=..., got {spec!r}")
    label, rest = spec.split("=", 1)
    return label.strip(), rest.strip()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[], metavar="LABEL=PATH",
                    help="A setting's own output; accuracy and n both come from it")
    ap.add_argument("--source", default="synthesized",
                    choices=sorted(LOADERS), help="How --run files are read")
    ap.add_argument("--value", action="append", default=[], metavar="LABEL=ACC/N",
                    help="A setting reported as a literal accuracy and denominator, "
                         "e.g. 'DICE=72.6/500'")
    ap.add_argument("--output", default=None, help="Write the intervals as JSON here")
    args = ap.parse_args()

    if not args.run and not args.value:
        raise SystemExit("Nothing to report: pass --run and/or --value")

    rows: List[Dict[str, Any]] = []
    for spec in args.run:
        label, raw = parse_spec(spec, "run")
        correct, n = from_run(Path(resolve_input(raw)), args.source)
        acc, hw = half_width(correct, n)
        rows.append({"setting": label, "accuracy": round(acc, 1),
                     "half_width": round(hw, 1), "n": n, "source": raw})
    for spec in args.value:
        label, raw = parse_spec(spec, "value")
        if "/" not in raw:
            raise SystemExit(f"--value takes LABEL=ACC/N, got {spec!r}")
        acc_s, n_s = raw.split("/", 1)
        n = int(n_s)
        correct = int(round(float(acc_s) / 100.0 * n))
        acc, hw = half_width(correct, n)
        rows.append({"setting": label, "accuracy": round(acc, 1),
                     "half_width": round(hw, 1), "n": n, "source": "reported"})

    print()
    print(f"  {'Setting':<30} {'Acc (%)':>18} {'n':>6}")
    print("  " + "-" * 58)
    for r in rows:
        print(f"  {r['setting']:<30} {r['accuracy']:>9.1f} ± {r['half_width']:<6.1f} {r['n']:>6}")
    print()
    print("  95% Wilson score interval. The half-width scales with 1/sqrt(n), so a")
    print("  row's n is printed beside it: two rows are only comparable at the same n.")

    if args.output:
        out = Path(resolve_output(args.output))
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
