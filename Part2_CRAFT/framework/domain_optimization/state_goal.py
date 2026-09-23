#!/usr/bin/env python3
"""Open each mathematics trace with the problem's own statement of the goal.

A solver's write-up begins with what is given and what is asked; the
synthesizer's first step tends to begin with a construction. This puts the
problem's qualifying sentences (see math_text.goal_sentences) verbatim on a
line before Step 1, for every trace in a synthesis output. The rule is
deterministic and touches nothing after that line, so the answer a trace
reaches is unchanged.

    python state_goal.py --synth synth.json --problems cleaned.json --output out.json

--problems is any file whose records carry sample_id and problem_text (the
cleaned Module I output, k_traces, or an adjudication file).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as _cfg  # noqa: E402
from framework.domain_optimization.math_text import prepend_goal  # noqa: E402


def records(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("results", raw.get("samples", raw))
    return raw if isinstance(raw, list) else list(raw.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    problems = {r["sample_id"]: r.get("problem_text") or "" for r in records(_cfg.resolve_input(args.problems))}
    out, changed, missing = [], 0, 0
    for r in records(_cfg.resolve_input(args.synth)):
        r = dict(r)
        trace = r.get("synthesized_trace") or ""
        if trace.strip():
            if r["sample_id"] not in problems:
                missing += 1
            else:
                new = prepend_goal(trace, problems[r["sample_id"]])
                changed += new != trace
                r["synthesized_trace"] = new
        out.append(r)
    if missing:
        raise SystemExit(f"{missing} traced samples have no problem text in {args.problems}")
    p = Path(_cfg.resolve_output(args.output))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out), encoding="utf-8")
    print(f"  {len(out)} samples, goal stated on {changed} -> {p}")


if __name__ == "__main__":
    main()
