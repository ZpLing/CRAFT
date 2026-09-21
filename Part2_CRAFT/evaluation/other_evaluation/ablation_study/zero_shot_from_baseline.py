#!/usr/bin/env python3
"""The ablation's 'w/o CRAFT' row, taken from the baseline run that already exists.

Removing the whole pipeline leaves one call to the backbone, which is what the
baselines' zero-shot CoT setting already is, over the same samples and the same
model. Generating it again would spend the calls a second time to get the same
thing, so this reads that run instead and rewrites it into the schema the
ablation's scorer loads.

What it does not do is carry the baseline's own prediction across. The row is
written with the trace text alone, so the scorer re-derives the label from it
with the same extractor every other row is scored with. A baseline and an
ablation row that disagree would then be disagreeing about the run, not about
how two scorers read a `\\boxed{}`.

Only the samples the CRAFT run covers are kept, matched by sample_id, so the row
and the row above it are over one set of problems.

    python zero_shot_from_baseline.py \
        --baseline results/baseline_results/gpt-5.4-nano/cot/traces.jsonl \
        --like craft_runs/full500/FLD_gpt-5.4-nano/synthesized.json \
        --output craft_runs/full500/FLD_gpt-5.4-nano/zero_shot.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
try:
    from config import resolve_input, resolve_output
except ImportError:  # running outside the part
    resolve_input = resolve_output = Path


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True,
                    help="traces.jsonl from the baseline's single-call setting (cot)")
    ap.add_argument("--like", required=True,
                    help="The CRAFT run whose samples this row must cover")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    like = json.loads(Path(resolve_input(args.like)).read_text(encoding="utf-8"))
    like = like.get("results", like) if isinstance(like, dict) else like
    wanted = {r.get("sample_id") for r in like}

    rows, seen = [], set()
    for rec in read_jsonl(Path(args.baseline)):
        sid = rec.get("sample_id")
        if sid not in wanted or sid in seen:
            continue
        seen.add(sid)
        traces = rec.get("traces") or []
        rows.append({
            "sample_id": sid,
            "source_dataset": rec.get("source_dataset"),
            "domain": rec.get("domain"),
            "ground_truth": rec.get("ground_truth"),
            "synthesized_trace": traces[0] if traces else "",
        })

    missing = wanted - seen
    out = Path(resolve_output(args.output))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"  {len(rows)} samples written to {out}")
    if missing:
        print(f"  {len(missing)} of the run's samples are absent from the baseline "
              f"and are left out of the row: {sorted(missing)[:5]}")


if __name__ == "__main__":
    main()
