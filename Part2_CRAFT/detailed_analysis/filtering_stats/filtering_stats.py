#!/usr/bin/env python3
"""
filtering_stats.py — how much each filtering pass removes, and why.

Produces the two appendix tables together, because both describe the same two
passes over the same run:

  Step removal by filtering pass
      Original, then Del and % for Pass 1 (z-score) and Pass 2 (RKG).
      Original is every reasoning step across all K traces of the run.

  RKG structural filter breakdown
      Pass 2 split into Nodes Filtering (a step whose node is isolated, zero
      in- and out-degree) and Edges Filtering (a step whose edge weight fell
      below theta).

Everything is read from a finished run; no LLM is called. A run records the
split itself — anomaly_filter tags each removal with the pass that caught it —
so the breakdown is measured rather than reconstructed. Runs written before
that tagging report the split as unavailable rather than guessing at it.

Usage:
    python filtering_stats.py \\
        --run "FLD (nano)=craft_runs/fld_nano" \\
        --run "FLD (Gemini-3.1-flash-lite)=craft_runs/fld_gemini" \\
        --output filtering_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from config import resolve_input, resolve_output
except ImportError:
    resolve_input = resolve_output = Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "framework"
                       / "module1_generation_filtering"))
try:
    from anomaly_filter import parse_steps_from_trace
except ImportError:  # keep the script usable without the framework on the path
    import re as _re
    _STEP = _re.compile(r"(?m)^\s*Step\s*\d+\s*[:.\-]")

    def parse_steps_from_trace(trace):  # noqa: D103
        text = (trace.get("reasoning_text") or trace.get("raw_response") or ""
                if isinstance(trace, dict) else str(trace))
        return [s for s in _STEP.split(text) if s.strip()]


def load_records(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return raw.get("results", raw) if isinstance(raw, dict) else raw


def load_blob(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return raw if isinstance(raw, dict) else {"results": raw}


def find_one(run_dir: Path, *patterns: str) -> Optional[Path]:
    for pattern in patterns:
        hits = sorted(run_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def count_steps(records: List[Dict[str, Any]], key: str) -> int:
    """Total reasoning steps across every trace of every sample."""
    total = 0
    for r in records:
        for trace in r.get(key) or []:
            steps = trace.get("reasoning_steps")
            total += len(steps) if steps else len(parse_steps_from_trace(trace))
    return total


def measure(run_dir: Path) -> Dict[str, Any]:
    k_path = find_one(run_dir, "k_traces_*_samples.json")
    z_path = find_one(run_dir, "cleaned_traces_z*.json", "cleaned_z*.json")
    rkg_path = find_one(run_dir, "cleaned_traces_rkg.json", "cleaned.json")
    if k_path is None or z_path is None:
        raise FileNotFoundError(f"{run_dir} needs k_traces_*_samples.json and the "
                                f"z-score pass output")

    original = count_steps(load_records(k_path), "traces")
    after_z = count_steps(load_records(z_path), "cleaned_traces")
    pass1 = original - after_z

    pass2 = after_z_rkg = None
    by_filter: Dict[str, int] = {}
    if rkg_path is not None and rkg_path != z_path:
        rkg_records = load_records(rkg_path)
        after_z_rkg = count_steps(rkg_records, "cleaned_traces")
        pass2 = after_z - after_z_rkg
        for r in rkg_records:
            for name, n in (r.get("stats", {}).get("removed_by_filter") or {}).items():
                by_filter[name] = by_filter.get(name, 0) + n
            # Runs from before the tagging carry the reason on each removal instead.
            if not r.get("stats", {}).get("removed_by_filter"):
                for a in r.get("anomalous_steps") or []:
                    if a.get("filter"):
                        by_filter[a["filter"]] = by_filter.get(a["filter"], 0) + 1

    return {
        "run_dir": str(run_dir),
        "original_steps": original,
        "pass1_removed": pass1,
        "pass1_pct": round(100.0 * pass1 / original, 1) if original else None,
        "pass2_removed": pass2,
        "pass2_pct": round(100.0 * pass2 / after_z, 1) if pass2 is not None and after_z else None,
        "by_filter": by_filter or None,
    }


def pct(part: Optional[int], whole: Optional[int]) -> str:
    if not part or not whole:
        return "—"
    return f"{part:,} ({round(100.0 * part / whole):d}%)"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=DIR",
                    help="One finished run per table row, in table order. Repeatable")
    ap.add_argument("--output", default=None, help="Write both tables as JSON here")
    args = ap.parse_args()

    rows: List[Tuple[str, Dict[str, Any]]] = []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run takes LABEL=DIR, got {spec!r}")
        label, raw = (s.strip() for s in spec.split("=", 1))
        rows.append((label, measure(Path(resolve_input(raw)))))

    print()
    print("  Step removal by filtering pass")
    print(f"  {'Dataset':<28} {'Original':>9} {'P1 Del':>8} {'P1 %':>6} {'P2 Del':>8} {'P2 %':>6}")
    print("  " + "-" * 70)
    for label, m in rows:
        p2 = f"{m['pass2_removed']:,}" if m["pass2_removed"] is not None else "—"
        p2p = f"{m['pass2_pct']:.1f}" if m["pass2_pct"] is not None else "—"
        print(f"  {label:<28} {m['original_steps']:>9,} {m['pass1_removed']:>8,} "
              f"{m['pass1_pct']:>6.1f} {p2:>8} {p2p:>6}")

    print()
    print("  RKG structural filter breakdown")
    print(f"  {'Dataset':<28} {'Filtering':>10} {'Nodes Filtering':>18} {'Edges Filtering':>18}")
    print("  " + "-" * 78)
    for label, m in rows:
        bf = m["by_filter"]
        if not bf:
            print(f"  {label:<28} {'—':>10} {'not recorded by this run':>38}")
            continue
        total = m["pass2_removed"] or sum(bf.values())
        print(f"  {label:<28} {total:>10,} {pct(bf.get('nodes'), total):>18} "
              f"{pct(bf.get('edges'), total):>18}")
        if bf.get("math"):
            print(f"  {'':<28} {'':>10} {'(SymPy refuted: ' + format(bf['math'], ',') + ')':>38}")
    print()

    if args.output:
        out = Path(resolve_output(args.output))
        out.write_text(json.dumps({label: m for label, m in rows}, indent=2),
                       encoding="utf-8")
        print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
