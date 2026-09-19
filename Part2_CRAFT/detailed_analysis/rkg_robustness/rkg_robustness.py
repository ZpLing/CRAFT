#!/usr/bin/env python3
"""
rkg_robustness.py — does the backbone change the graph Module II extracts?

Graph construction asks an LLM to name each step's dependencies, so the appendix
checks that the answer is a property of the traces rather than of the model. Two
measurements, from per-trace RKGs built over the same traces by different
backbones under the same prompt:

  Edge extraction F1   per model, against gold edge annotations, plus the
                       pair-wise Pearson r of the per-sample F1 between models.
                       Needs --gold.

  Pair-wise agreement  Jaccard overlap of the edge sets two models extract for
                       the same trace. Needs no annotations, and answers the
                       same question from the other side: models that agree with
                       each other cannot be adding model-specific noise.

Gold annotations are a JSON object mapping a sample id to its edge list,
{"FLD_0": [["Step1", "Step3"], ...]}. FLD as shipped here carries the premises
and the conclusion but not the proof tree, so the gold file comes from the
upstream annotations rather than from dataset/.

Usage:
    python rkg_robustness.py \\
        --model "GPT-5.4-nano=craft_runs/fld_nano/rkg.json" \\
        --model "Gemini-3.1-flash-lite=craft_runs/fld_gemini/rkg.json" \\
        --gold fld_gold_edges.json --output rkg_robustness.json
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from config import resolve_input, resolve_output
except ImportError:
    resolve_input = resolve_output = Path

Edge = Tuple[str, str]


def load_records(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return raw.get("results", raw) if isinstance(raw, dict) else raw


def edges_of(sample: Dict[str, Any]) -> Set[Edge]:
    """Every edge the model extracted for this sample, across its K per-trace RKGs.

    The union rather than the consensus: this asks what the extraction prompt
    produced, and consensus is a later stage that would hide a disagreement by
    voting it away.
    """
    out: Set[Edge] = set()
    for rkg in (sample.get("trace_rkgs") or sample.get("trace_dags") or []):
        if rkg.get("extraction_method") == "error":
            continue
        for e in rkg.get("edges", []):
            src, dst = e.get("src"), e.get("dst")
            if src and dst:
                out.add((str(src), str(dst)))
    return out


def prf(extracted: Set[Edge], gold: Set[Edge]) -> Dict[str, float]:
    tp = len(extracted & gold)
    precision = tp / len(extracted) if extracted else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "extracted": len(extracted), "gold": len(gold)}


def pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    """Pearson r, or None when a series has no spread for it to be defined on."""
    n = len(xs)
    if n < 2:
        return None
    mx, my = mean(xs), mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return cov / (vx ** 0.5 * vy ** 0.5)


def jaccard(a: Set[Edge], b: Set[Edge]) -> Optional[float]:
    union = a | b
    return len(a & b) / len(union) if union else None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=RKG_JSON",
                    help="One rkg.json per backbone, all built over the same traces")
    ap.add_argument("--gold", default=None,
                    help="Gold edge annotations {sample_id: [[src, dst], ...]}. "
                         "Without it only the pair-wise agreement is reported")
    ap.add_argument("--output", default=None, help="Write the measurements as JSON here")
    args = ap.parse_args()

    per_model: Dict[str, Dict[str, Set[Edge]]] = {}
    for spec in args.model:
        if "=" not in spec:
            raise SystemExit(f"--model takes LABEL=RKG_JSON, got {spec!r}")
        label, raw = (s.strip() for s in spec.split("=", 1))
        per_model[label] = {r["sample_id"]: edges_of(r)
                            for r in load_records(Path(resolve_input(raw)))
                            if "sample_id" in r}
    if len(per_model) < 2:
        raise SystemExit("Pass at least two --model entries; this compares backbones")

    # Only samples every model graphed: a sample one model failed on would
    # otherwise change which samples each model's mean is taken over.
    shared = set.intersection(*(set(m) for m in per_model.values()))
    shared_ids = sorted(shared)
    print(f"\n  {len(shared_ids)} samples graphed by all {len(per_model)} models")

    payload: Dict[str, Any] = {"n_shared_samples": len(shared_ids),
                               "models": list(per_model)}

    per_sample_f1: Dict[str, List[float]] = {}
    if args.gold:
        with open(resolve_input(args.gold), encoding="utf-8") as f:
            gold_raw = json.load(f)
        gold = {sid: {(str(a), str(b)) for a, b in edges}
                for sid, edges in gold_raw.items()}
        scored = [sid for sid in shared_ids if sid in gold]
        if not scored:
            raise SystemExit("No sample id in --gold matches the RKG files")
        print(f"  {len(scored)} of them have gold annotations\n")
        print(f"  {'Model':<28} {'P':>7} {'R':>7} {'F1':>7}")
        print("  " + "-" * 52)
        summary = {}
        for label, by_id in per_model.items():
            rows = [prf(by_id[sid], gold[sid]) for sid in scored]
            per_sample_f1[label] = [r["f1"] for r in rows]
            summary[label] = {k: round(mean(r[k] for r in rows), 4)
                              for k in ("precision", "recall", "f1")}
            summary[label]["n"] = len(rows)
            s = summary[label]
            print(f"  {label:<28} {s['precision']:>7.4f} {s['recall']:>7.4f} {s['f1']:>7.4f}")
        payload["edge_extraction"] = summary
        payload["n_gold_samples"] = len(scored)

        print(f"\n  {'Model pair':<44} {'Pearson r':>10}")
        print("  " + "-" * 56)
        correlations = {}
        for a, b in combinations(per_model, 2):
            r = pearson(per_sample_f1[a], per_sample_f1[b])
            correlations[f"{a} vs {b}"] = None if r is None else round(r, 4)
            print(f"  {a + ' vs ' + b:<44} {'—' if r is None else f'{r:.4f}':>10}")
        payload["pearson_r_per_sample_f1"] = correlations

    print(f"\n  {'Model pair':<44} {'edge Jaccard':>13}")
    print("  " + "-" * 59)
    agreement = {}
    for a, b in combinations(per_model, 2):
        vals = [j for sid in shared_ids
                if (j := jaccard(per_model[a][sid], per_model[b][sid])) is not None]
        m = round(mean(vals), 4) if vals else None
        agreement[f"{a} vs {b}"] = m
        print(f"  {a + ' vs ' + b:<44} {'—' if m is None else f'{m:.4f}':>13}")
    payload["edge_agreement_jaccard"] = agreement
    if not args.gold:
        print("\n  No --gold given, so edge extraction F1 and its Pearson r are not reported.")
    print()

    if args.output:
        out = Path(resolve_output(args.output))
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
