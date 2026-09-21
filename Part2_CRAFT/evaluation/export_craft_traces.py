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
scorer uses, so a line cannot disagree with the reported table.

    python export_craft_traces.py --out_dir CRAFT_results/traces
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as _cfg  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "label_prediction"))
from evaluate_accuracy import LOADERS, compute_metrics  # noqa: E402

# (dataset, model) -> (run directory, file, how it was run)
CELLS = {
    ("FLD", "gemini-3.1-flash-lite"):
        ("FLD_gemini-3.1-flash-lite", "synth_prior", "prior_mode=verify"),
    ("FLD", "gpt-5.4-nano"):
        ("FLD_gpt-5.4-nano", "synthesized", "prior_mode=follow"),
    ("ProofWriter", "gemini-3.1-flash-lite"):
        ("ProofWriter_gd", "synth_cwa",
         "prior_mode=verify, weight_by=gold_depth, cwa_recheck"),
    ("ProofWriter", "gpt-5.4-nano"):
        ("ProofWriter_gpt-5.4-nano", "synth_cwa",
         "prior_mode=verify, weight_by=gold_depth, cwa_recheck"),
    ("OmniMATH", "gemini-3.1-flash-lite"):
        ("OmniMATH_gemini-3.1-flash-lite", "synthesized", "prior_mode=verify"),
    ("OmniMATH", "gpt-5.4-nano"):
        ("OmniMATH_gpt-5.4-nano", "synth_follow2", "prior_mode=follow"),
    ("OlympiadBench", "gemini-3.1-flash-lite"):
        ("OlympiadBench_gemini-3.1-flash-lite", "synthesized", "prior_mode=verify"),
    ("OlympiadBench", "gpt-5.4-nano"):
        ("OlympiadBench_gpt-5.4-nano", "synth_follow2", "prior_mode=follow"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="craft_runs/full500")
    ap.add_argument("--out_dir", default="CRAFT_results/traces")
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
        out = out_dir / model / f"{ds}_Traces.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for r in rows:
                t = r["traces"][0]
                fh.write(json.dumps({
                    "sample_id": r["sample_id"],
                    "source_dataset": r["source_dataset"],
                    "domain": r["domain"],
                    "model": model,
                    "setting": setting,
                    "ground_truth": r["ground_truth"],
                    "predicted": r["predicted"],
                    "n_steps": t["n_steps"],
                    "trace": t["text"],
                }, ensure_ascii=False) + "\n")
        print(f"  {model:<22} {ds:<14} {len(rows):>4} samples  "
              f"acc {100*metrics['accuracy']:5.1f}  steps {metrics['avg_steps']:4.1f}  -> {out}")

    if missing:
        raise SystemExit("These cells have no reported trace file:\n  "
                         + "\n  ".join(missing))


if __name__ == "__main__":
    main()
