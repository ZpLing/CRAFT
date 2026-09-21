#!/usr/bin/env python3
"""Fold an adjudication pass back into the synthesized traces it re-answered.

The pass runs over k_traces and records which answer a fresh derivation landed
on. This writes that back into the synthesis output so the result is scored by
the same loader and the same metric as every other cell, rather than by a
count written for the occasion — two numbers computed different ways are not
comparable even when both are right.

An answer is taken only where the pass ran, landed on one of the candidates,
and the consensus behind the original answer was no stronger than --max_votes.
That threshold is chosen on a validation split, not here.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as _cfg
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation" / "label_prediction"))
from answer_match import answers_match  # noqa: E402


def load(p):
    raw = json.loads(Path(p).read_text(encoding="utf-8"))
    return raw.get("results", raw) if isinstance(raw, dict) else raw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth", required=True)
    ap.add_argument("--adjudicated", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max_votes", type=int, default=3,
                    help="Only override where the consensus had at most this "
                         "many of the k votes behind it")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    syn = load(_cfg.resolve_input(args.synth))
    adj = {r["sample_id"]: r for r in load(_cfg.resolve_input(args.adjudicated))}

    changed = 0
    out = []
    for r in syn:
        r = dict(r)
        a = (adj.get(r["sample_id"]) or {}).get("adjudication") or {}
        src = adj.get(r["sample_id"]) or {}
        if a.get("ran") and a.get("picked"):
            labs = [t.get("label") for t in (src.get("traces") or []) if t.get("label")]
            top = sum(1 for l in labs if answers_match(l, a["options"][0], args.dataset))
            if top <= args.max_votes and not answers_match(
                    a["picked"], a["options"][0], args.dataset):
                r["synthesized_trace"] = ((r.get("synthesized_trace") or "").rstrip()
                                          + "\n\n[Re-derivation]\n" + a.get("text", "").strip())
                changed += 1
        out.append(r)

    p = Path(_cfg.resolve_output(args.output))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out), encoding="utf-8")
    print(f"  {len(out)} samples, {changed} answers replaced -> {p}")


if __name__ == "__main__":
    main()
