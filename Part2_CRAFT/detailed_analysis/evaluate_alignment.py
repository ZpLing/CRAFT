#!/usr/bin/env python3
"""
evaluate_alignment.py
─────────────────────
Compare RKG construction with and without cross-trace step alignment.

For each sample, builds the consensus RKG in two modes:
  - baseline : positional Step IDs (current behavior)
  - aligned  : canonical IDs assigned by LLM alignment call first

Metrics reported per mode:
  consensus_edges     : edges passing the frequency threshold
  avg_edge_frequency  : mean frequency of consensus edges
  max_frequency       : highest edge frequency (shows how "tight" consensus is)
  shared_groups       : (aligned only) # canonical groups with ≥2 members
  shared_ratio        : (aligned only) shared_groups / total canonical IDs

Usage:
  python evaluate_alignment.py \
    --input craft_runs/craft_k5_full_fld_nano/k_traces.json \
    --n_samples 10 \
    --model gpt-4.1-mini \
    --threshold 0.3 \
    --output alignment_eval.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import require_pinned_endpoint, resolve_input, resolve_output
from framework.module2_rkg_construction.build_rkg import build_rkgs_for_sample, OPENAI_API_KEY, OPENAI_BASE_URL

# Pinned endpoint, read from the gitignored repo-root config.py.
# Explicit rather than via OPENAI_* so a stray env var cannot redirect these runs.
PINNED_KEY, PINNED_URL = require_pinned_endpoint()


def rkg_stats(consensus: Dict[str, Any]) -> Dict[str, float]:
    edges = consensus.get("edges", [])
    freqs = [e.get("frequency", 0) for e in edges]
    return {
        "consensus_edges":    len(edges),
        "avg_edge_frequency": round(sum(freqs) / len(freqs), 3) if freqs else 0.0,
        "max_frequency":      round(max(freqs), 3) if freqs else 0.0,
        "nodes_in_consensus": len(consensus.get("nodes", [])),
    }


async def run_sample(
    session: aiohttp.ClientSession,
    sample: Dict[str, Any],
    model: str,
    threshold: float,
) -> Dict[str, Any]:
    sample_id = sample.get("sample_id", "?")

    # Baseline (no alignment)
    base_result = await build_rkgs_for_sample(
        session, sample, model=model,
        consensus_threshold=threshold, use_alignment=False,
    )
    base_stats = rkg_stats(base_result["consensus_dag"])

    # Aligned
    aln_result = await build_rkgs_for_sample(
        session, sample, model=model,
        consensus_threshold=threshold, use_alignment=True,
    )
    aln_stats  = rkg_stats(aln_result["consensus_dag"])
    aln_info   = aln_result.get("alignment", {})

    return {
        "sample_id": sample_id,
        "baseline":  base_stats,
        "aligned":   {**aln_stats,
                      "shared_groups": aln_info.get("n_shared_groups", 0),
                      "shared_ratio":  aln_info.get("shared_ratio", 0.0),
                      "n_canonical":   aln_info.get("n_canonical_ids", 0),
                      "n_total_steps": aln_info.get("n_total_steps", 0)},
    }


async def main_async(args: argparse.Namespace) -> None:
    import framework.module2_rkg_construction.build_rkg as _rkg
    _rkg.OPENAI_API_KEY  = PINNED_KEY
    _rkg.OPENAI_BASE_URL = PINNED_URL
    _rkg.CHAT_URL        = PINNED_URL.rstrip("/") + "/chat/completions"
    _rkg.HEADERS         = {"Authorization": f"Bearer {PINNED_KEY}",
                            "Content-Type": "application/json"}

    with open(resolve_input(args.input)) as f:
        raw = json.load(f)
    all_samples = raw if isinstance(raw, list) else raw.get("results", [])
    samples = all_samples[: args.n_samples]
    print(f"Samples: {len(samples)} | Model: {args.model} | Threshold: {args.threshold}")

    results = []
    connector = aiohttp.TCPConnector(limit=6)
    async with aiohttp.ClientSession(connector=connector) as session:
        for i, sample in enumerate(samples):
            print(f"  [{i+1}/{len(samples)}] sample={sample.get('sample_id','?')}", flush=True)
            try:
                r = await run_sample(session, sample, args.model, args.threshold)
                results.append(r)
                b, a = r["baseline"], r["aligned"]
                print(f"    baseline  edges={b['consensus_edges']} avg_freq={b['avg_edge_frequency']}")
                print(f"    aligned   edges={a['consensus_edges']} avg_freq={a['avg_edge_frequency']} "
                      f"shared_groups={a['shared_groups']} ratio={a['shared_ratio']}")
            except Exception as e:
                print(f"    ERROR: {e}")
                results.append({"sample_id": sample.get("sample_id"), "error": str(e)})

    # Aggregate
    valid = [r for r in results if "error" not in r]
    if valid:
        def _avg(key, sub): return round(sum(r[sub][key] for r in valid) / len(valid), 3)
        print("\n" + "=" * 60)
        print(f"{'Metric':<28} {'Baseline':>10} {'Aligned':>10} {'Delta':>8}")
        print("-" * 60)
        for metric in ["consensus_edges", "avg_edge_frequency", "max_frequency", "nodes_in_consensus"]:
            b = _avg(metric, "baseline")
            a = _avg(metric, "aligned")
            delta = round(a - b, 3)
            sign = "+" if delta > 0 else ""
            print(f"{metric:<28} {b:>10} {a:>10} {sign+str(delta):>8}")
        print(f"{'shared_groups (aligned)':<28} {'—':>10} {_avg('shared_groups','aligned'):>10}")
        print(f"{'shared_ratio (aligned)':<28} {'—':>10} {_avg('shared_ratio','aligned'):>10}")
        print("=" * 60)

    out = resolve_output(args.output)
    with open(out, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nSaved: {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",     required=True)
    p.add_argument("--n_samples", type=int, default=10)
    p.add_argument("--model",     default="gemini-3.1-flash-lite")
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--output",    default="alignment_eval.json",
                   help="Relative paths resolve under the results root")
    asyncio.run(main_async(p.parse_args()))

if __name__ == "__main__":
    main()
