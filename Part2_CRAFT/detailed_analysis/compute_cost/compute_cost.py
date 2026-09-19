#!/usr/bin/env python3
"""
compute_cost.py — LLM API calls per sample, measured rather than derived.

The appendix table gives the cost per module as an expression: K for generation,
K for the per-trace RKGs, n* + [0-2] for synthesis, where n* is the number of
non-fact nodes in the consensus RKG. What a run actually pays differs from that
expression — a trace that failed was retried, a malformed graph was re-requested
— so this reads the counter each module writes into its own output and reports
both: what was spent, and what the expression predicts for the same run.

Every LLM request in Modules I-III passes through one function per module, and
that function increments the counter, so a retry is counted as the call it is.
The filter stages make no calls at all, which is why they are listed at zero
rather than omitted.

Runs recorded before the counter existed report the measured column as absent;
the predicted column still works, since it needs only K and the graphs.

Usage:
    python compute_cost.py --run "FLD (Gemini)=craft_runs/fld_gemini" \\
                           --run "FLD (nano)=craft_runs/fld_nano" \\
                           --output compute_cost.json
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


def non_fact_nodes(rkg_blob: Dict[str, Any]) -> Tuple[Optional[float], int]:
    """Mean n*: the consensus-RKG nodes Module III has to generate a step for.

    Fact nodes are the problem's given premises, so they are walked but never
    generated; counting them would overstate what synthesis costs.
    """
    rows = rkg_blob.get("results", [])
    totals, graphs = 0, 0
    for r in rows:
        consensus = r.get("consensus_rkg") or r.get("consensus_dag") or {}
        nodes = consensus.get("nodes")
        if not nodes:
            continue
        graphs += 1
        totals += sum(1 for n in nodes if n.get("type") != "fact")
    return (totals / graphs if graphs else None), graphs


def measure(run_dir: Path) -> Dict[str, Any]:
    k_path = find_one(run_dir, "k_traces_*_samples.json")
    rkg_path = find_one(run_dir, "rkg*.json")
    synth_path = find_one(run_dir, "synthesized*.json")
    if k_path is None:
        raise FileNotFoundError(f"No k_traces_*_samples.json in {run_dir}")

    k_blob = load_blob(k_path)
    k_meta = k_blob.get("metadata", {})
    n_samples = len(k_blob.get("results", []))
    k = k_meta.get("k") or k_meta.get("statistics", {}).get("requested_traces_per_sample")

    rkg_blob = load_blob(rkg_path) if rkg_path else {}
    n_star, n_graphs = non_fact_nodes(rkg_blob)
    synth_blob = load_blob(synth_path) if synth_path else {}
    n_synth = len(synth_blob.get("results", []))

    def calls(blob: Dict[str, Any]) -> Optional[int]:
        return (blob.get("metadata") or {}).get("api_calls")

    modules = {
        "I  generate K traces":       {"measured": calls(k_blob),
                                       "predicted_per_sample": float(k) if k else None,
                                       "n_samples": n_samples},
        "I  TF-IRF terms":            {"measured": 0, "predicted_per_sample": 0.0},
        "I  z-score steps filtering": {"measured": 0, "predicted_per_sample": 0.0},
        "II build per-trace RKGs":    {"measured": calls(rkg_blob),
                                       "predicted_per_sample": float(k) if k else None,
                                       "n_samples": n_graphs or None},
        "II consensus RKG + filter":  {"measured": 0, "predicted_per_sample": 0.0},
        "III topology-guided synthesis": {"measured": calls(synth_blob),
                                          "predicted_per_sample": n_star,
                                          "n_samples": n_synth or None},
    }
    # Each module is divided by the samples it actually ran on: synthesis often
    # covers fewer than generation did, and dividing everything by one n would
    # quietly inflate the cheaper stages.
    for m in modules.values():
        n = m.get("n_samples") or n_samples
        m["per_sample"] = (round(m["measured"] / n, 2)
                           if m["measured"] is not None and n else None)

    measured_total = sum(m["measured"] for m in modules.values()
                         if m["measured"] is not None)
    have_all = all(m["measured"] is not None for m in modules.values())
    per_sample_total = sum(m["per_sample"] for m in modules.values()
                           if m["per_sample"] is not None)
    predicted_total = (2 * k + n_star) if (k and n_star) else None

    return {"run_dir": str(run_dir), "k": k, "n_star": round(n_star, 2) if n_star else None,
            "n_samples": n_samples, "modules": modules,
            "measured_total_calls": measured_total if have_all else None,
            "measured_per_sample": round(per_sample_total, 2) if have_all else None,
            "predicted_per_sample": round(predicted_total, 2) if predicted_total else None}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=DIR",
                    help="A finished run. Repeatable")
    ap.add_argument("--output", default=None, help="Write the breakdown as JSON here")
    args = ap.parse_args()

    results: Dict[str, Dict[str, Any]] = {}
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run takes LABEL=DIR, got {spec!r}")
        label, raw = (s.strip() for s in spec.split("=", 1))
        m = measure(Path(resolve_input(raw)))
        results[label] = m

        print()
        print(f"  {label}   (K={m['k']}, n*={m['n_star']}, {m['n_samples']} samples)")
        print(f"  {'Module / Operation':<34} {'Calls':>10} {'/sample':>9} {'predicted':>10}")
        print("  " + "-" * 66)
        for name, mod in m["modules"].items():
            calls = f"{mod['measured']:,}" if mod["measured"] is not None else "not recorded"
            per = f"{mod['per_sample']:.2f}" if mod["per_sample"] is not None else "—"
            pred = (f"{mod['predicted_per_sample']:.2f}"
                    if mod["predicted_per_sample"] is not None else "—")
            print(f"  {name:<34} {calls:>10} {per:>9} {pred:>10}")
        print("  " + "-" * 66)
        tot = f"{m['measured_total_calls']:,}" if m["measured_total_calls"] is not None else "—"
        mps = f"{m['measured_per_sample']:.2f}" if m["measured_per_sample"] is not None else "—"
        pps = f"{m['predicted_per_sample']:.2f}" if m["predicted_per_sample"] is not None else "—"
        print(f"  {'Total per sample':<34} {tot:>10} {mps:>9} {pps:>10}")
    print()
    print("  predicted = the appendix expression 2K + n* for the same run;")
    print("  /sample is what it actually cost, retries included.")

    if args.output:
        out = Path(resolve_output(args.output))
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
