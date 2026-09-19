#!/usr/bin/env python3
"""
rkg_construct_robustness.py — does the backbone change the graph Module II extracts?

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

Gold annotations and the traces they describe both come from build_gold_edges.py,
which renders FLD's own proofs. They have to come from the same place: step ids
are positional, so scoring a generated trace against FLD's proof would compare
two different numberings and blame the extractor for the mismatch.

Usage:
    python build_gold_edges.py --dataset FLD_with_proofs.json
    # build an rkg.json from detailed_analysis/gold_traces.json with each backbone, then
    python rkg_construct_robustness.py \\
        --model "GPT-5.4-nano=detailed_analysis/gpt-5.4-nano/rkg.json" \\
        --model "Gemini-3.1-flash-lite=detailed_analysis/gemini-3.1-flash-lite/rkg.json" \\
        --gold detailed_analysis/gold_edges.json

The graphs it reads are each one model's, so they sit in that model's directory.
This measurement spans them, and the gold annotations belong to no model, so both
stay at the top of <results root>/detailed_analysis/ rather than under one of them.
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


def edges_by_trace(sample: Dict[str, Any]) -> Dict[int, Set[Edge]]:
    """{trace_idx: the edges the model extracted from that trace}.

    Per trace, not pooled across them. Step ids are positional within a trace, so
    "Step3" of trace 0 and "Step3" of trace 1 are different steps; pooling would
    compare a set of ids that mean several things at once. Comparing trace 0 of
    one model against trace 0 of another is well defined, because both graphed
    the same text.

    The per-trace graphs rather than the consensus: this asks what the extraction
    prompt produced, and consensus is a later stage that would hide a
    disagreement by voting it away.
    """
    out: Dict[int, Set[Edge]] = {}
    for rkg in (sample.get("trace_rkgs") or sample.get("trace_dags") or []):
        if rkg.get("extraction_method") == "error":
            continue
        idx = rkg.get("trace_idx")
        if idx is None:
            continue
        edges = {(str(e["src"]), str(e["dst"])) for e in rkg.get("edges", [])
                 if e.get("src") and e.get("dst")}
        out[int(idx)] = edges
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
    ap.add_argument("--output", default="detailed_analysis/rkg_construct_robustness.json",
                    help="Write the measurements as JSON here. It compares the models "
                         "rather than reporting one, so it is not filed under a model "
                         "directory (relative paths land under the results root)")
    args = ap.parse_args()

    per_model: Dict[str, Dict[str, Dict[int, Set[Edge]]]] = {}
    for spec in args.model:
        if "=" not in spec:
            raise SystemExit(f"--model takes LABEL=RKG_JSON, got {spec!r}")
        label, raw = (s.strip() for s in spec.split("=", 1))
        per_model[label] = {r["sample_id"]: edges_by_trace(r)
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
        # Every model must have graphed at least one trace of a scored sample.
        # Otherwise one model's score list is shorter than another's, and the
        # pair-wise correlation below would pair up different samples.
        scored = [sid for sid in shared_ids
                  if sid in gold and all(per_model[m][sid] for m in per_model)]
        if not scored:
            raise SystemExit("No sample id in --gold was graphed by every model")
        print(f"  {len(scored)} of them have gold annotations\n")
        print(f"  {'Model':<28} {'P':>7} {'R':>7} {'F1':>7}")
        print("  " + "-" * 52)
        summary = {}
        for label, by_id in per_model.items():
            rows = []
            for sid in scored:
                # A sample's score is the mean over its traces, so a sample with
                # more traces does not weigh more than one with fewer.
                per_trace = [prf(edges, gold[sid]) for edges in by_id[sid].values()]
                rows.append({k: mean(r[k] for r in per_trace)
                             for k in ("precision", "recall", "f1")})
            per_sample_f1[label] = [r["f1"] for r in rows]
            summary[label] = {k: round(mean(r[k] for r in rows), 4)
                              for k in ("precision", "recall", "f1")}
            summary[label]["n"] = len(rows)
            s = summary[label]
            print(f"  {label:<28} {s['precision']:>7.4f} "
                  f"{s['recall']:>7.4f} {s['f1']:>7.4f}")
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
        vals = []
        for sid in shared_ids:
            ta, tb = per_model[a][sid], per_model[b][sid]
            # Only traces both models graphed: a trace one of them failed on has
            # no counterpart to agree or disagree with.
            per_trace = [j for idx in sorted(set(ta) & set(tb))
                         if (j := jaccard(ta[idx], tb[idx])) is not None]
            if per_trace:
                vals.append(mean(per_trace))
        m = round(mean(vals), 4) if vals else None
        agreement[f"{a} vs {b}"] = m
        print(f"  {a + ' vs ' + b:<44} {'—' if m is None else f'{m:.4f}':>13}")
    payload["edge_agreement_jaccard"] = agreement
    if not args.gold:
        print("\n  No --gold given, so edge extraction F1 and its Pearson r are not reported.")
    print()

    out = Path(resolve_output(args.output))
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
