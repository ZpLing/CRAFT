#!/usr/bin/env python3
"""Export the trace CRAFT actually reports, one JSONL per benchmark and backbone.

A cell's reported number does not always come from `synthesized.json`. Three
settings differ per configuration and each writes its own file — prior_mode,
the depth-weighted consensus, and the closed-world recheck — so reading the
directory and taking the obvious name gives the wrong trace for five of the
eight cells. This module holds which file each cell reports from, and the
export fails rather than falling back if that file is missing.

Each line is one sample:

    sample_id, source_dataset, domain, ground_truth, predicted, n_steps,
    trace (the post-processed text), setting (how the cell was run)

`predicted` is re-derived from the trace text with the same extractor the
scorer uses, so a line cannot disagree with the reported table. A sample
Module III produced no trace for is left out, as the scorer leaves it out, so
this file's line count is the n its cell reports.

The trace is written out with its restatements removed. Walking the consensus
graph carries each step's conclusion forward, so a late step repeats what the
earlier ones established: across the eight cells that is 8% to 20% of the
sentences, and the final answer itself is written out up to four times. The
answer is read from the last commitment a trace makes, and dedup_trace leaves
every sentence that states one alone and checks the rewrite against this
cell's own reader, so `predicted` cannot move -- over the 3969 traces here it
does not, on any of them.

    python export_craft_traces.py --out_dir CRAFT_results/Output

The run directory holds one folder per cell (see CELLS) with the file that
cell reports from; the 2026-09-23 run lives under craft_runs/full500_v4/traces.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as _cfg  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "label_prediction"))
from evaluate_accuracy import LOADERS, compute_metrics  # noqa: E402
from extract_label import extract_label, extract_math_answer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "framework" / "module3_topology_guided_synthesis"))
from dedup_trace import dedup_trace  # noqa: E402

# Which reader names the answer, so the rewrite can be checked against it.
_READER = {"FLD": extract_label, "ProofWriter": extract_label,
           "OmniMATH": extract_math_answer, "OlympiadBench": extract_math_answer}

_STEP_HEAD = re.compile(r"(?m)^\s*Step\s*\d+\s*[:.\-]\s*")

# (dataset, model) -> (cell directory under --runs, file, how it was run).
# The 2026-09-23 run: the atomic synthesizer for every cell; ProofWriter goes
# through the closed-world recheck with its proof search written back as
# steps; gpt's ProofWriter trace is then restated with its answer pinned;
# gemini's mathematics has the adjudicator's derivation written back as steps
# and opens with the problem's own statement of the goal.
CELLS = {
    ("FLD", "gemini-3.1-flash-lite"):
        ("FLDgem", "synth", "prior_mode=verify"),
    ("FLD", "gpt-5.4-nano"):
        ("FLDgpt", "synth", "prior_mode=follow"),
    ("ProofWriter", "gemini-3.1-flash-lite"):
        ("PWgem", "cwa_resolve",
         "prior_mode=verify, weight_by=gold_depth, cwa_recheck (integrated), cwa_resolve"),
    ("ProofWriter", "gpt-5.4-nano"):
        ("PWgpt", "polish",
         "prior_mode=verify, weight_by=gold_depth, cwa_recheck (integrated), cwa_resolve, polish two3"),
    ("OmniMATH", "gemini-3.1-flash-lite"):
        ("OMgem", "adj_goal", "prior_mode=verify, adjudication (integrated), goal stated"),
    ("OmniMATH", "gpt-5.4-nano"):
        ("OMgpt", "synth", "prior_mode=follow"),
    ("OlympiadBench", "gemini-3.1-flash-lite"):
        ("OBgem", "adj_goal", "prior_mode=verify, adjudication (integrated), goal stated"),
    ("OlympiadBench", "gpt-5.4-nano"):
        ("OBgpt", "synth", "prior_mode=follow"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="craft_runs/full500_v4/traces")
    ap.add_argument("--out_dir", default="CRAFT_results/Output")
    args = ap.parse_args()

    runs = Path(_cfg.resolve_input(args.runs))
    out_dir = Path(_cfg.resolve_output(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for (ds, model), (run, stem, setting) in sorted(CELLS.items()):
        src = runs / run / f"{stem}.json"
        if not src.exists():
            missing.append(str(src))
            continue
        rows = LOADERS["synthesized"](src)
        metrics = compute_metrics(rows)
        shrunk = total = dropped = 0
        out = out_dir / model / f"{ds}_Output.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            reader = _READER[ds]
            for r in rows:
                t = r["traces"][0]
                original = t["text"] or ""
                # Module III produced nothing for this sample. The scorer
                # already leaves those out -- a cell's n is 492 to 500, not 500
                # -- so writing them here as an empty trace put a row in this
                # file that no reported number counts, and handed anything
                # reading it an empty string to score.
                if not original.strip():
                    dropped += 1
                    continue
                text = dedup_trace(original, extractor=reader)
                n_steps = len(_STEP_HEAD.findall(text)) or t["n_steps"]
                shrunk += len(original) - len(text)
                total += len(original)
                fh.write(json.dumps({
                    "sample_id": r["sample_id"],
                    "source_dataset": r["source_dataset"],
                    "domain": r["domain"],
                    "model": model,
                    "setting": setting,
                    "ground_truth": r["ground_truth"],
                    "predicted": r["predicted"],
                    "n_steps": n_steps,
                    "trace": text,
                }, ensure_ascii=False) + "\n")
        print(f"  {model:<22} {ds:<14} {len(rows) - dropped:>4} traces  "
              f"acc {100*metrics['accuracy']:5.1f}  steps {metrics['avg_steps']:4.1f}  "
              f"trimmed {100*shrunk/max(total,1):4.1f}%"
              + (f"  dropped {dropped}" if dropped else "") + f"  -> {out}")

    if missing:
        raise SystemExit("These cells have no reported trace file:\n  "
                         + "\n  ".join(missing))


if __name__ == "__main__":
    main()
