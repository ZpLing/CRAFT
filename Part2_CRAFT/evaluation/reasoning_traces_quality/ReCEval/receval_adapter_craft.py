"""
Adapter: CRAFT pipeline outputs -> ReCEval flat-text schema.

Pairs raw k_traces[0] (first generated trace = raw CoT baseline) with
CRAFT's synthesized_trace (post-processed), emitting a JSON file in the
exact schema that receval_evaluate_traces.py expects:

    [{
       "id": ..., "hypothesis": ..., "question": ...,
       "with_answer": {"steps": [...]},   # CRAFT post-processed
       "blind":       {"steps": [...]},   # raw CoT
    }]

Container names (with_answer/blind) are kept so the existing
receval_evaluate_traces.py runs without modification. The comparison
script relabels them in the final table.

Usage:
    python receval_adapter_craft.py \\
        --craft_dir results/craft_runs/craft_fld_o4mini_100 \\
        --source    dataset/logical/FLD.json \\
        --output    receval_inputs/fld_o4mini.json \\
        [--raw_mode first|majority]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Source dataset field extraction (FLD / FOLIO / math)
# ---------------------------------------------------------------------------

def extract_hypothesis_and_premises(sample: dict) -> tuple[str, str]:
    """Return (hypothesis, premises_text) from an FLD/FOLIO/math sample."""
    # FOLIO: has ori_conclusion + ori_premises list
    if "ori_conclusion" in sample and "ori_premises" in sample:
        hyp = sample["ori_conclusion"]
        prem = " ".join(sample["ori_premises"])
        return hyp, prem
    # FLD: has Conclusion + Facts
    if "Conclusion" in sample and "Facts" in sample:
        return sample["Conclusion"], sample["Facts"]
    # GSM8K / Olympiad style
    if "question" in sample and "answer" in sample:
        return str(sample["answer"]), sample["question"]
    # Fallback — use input as premises, proof_label as hypothesis marker
    return sample.get("proof_label", ""), sample.get("input", "")


# ---------------------------------------------------------------------------
# CRAFT trace helpers
# ---------------------------------------------------------------------------

_STEP_RE = re.compile(r"(?m)^\s*Step\s*\d+\s*[:.\-]\s*")


def split_synthesized_text(text: str) -> list[str]:
    """Synthesized trace is one string 'Step 1: ... Step 2: ...' — split into steps."""
    if not text:
        return []
    parts = _STEP_RE.split(text)
    # First element before first 'Step 1:' is preamble (usually empty / whitespace)
    steps = [p.strip() for p in parts if p.strip()]
    return steps


def majority_label(traces: list[dict]) -> str | None:
    labels = [t.get("label") for t in traces if t.get("label")]
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]


def pick_raw_trace(traces: list[dict], mode: str) -> list[str]:
    """Return reasoning_steps list for the chosen raw trace. Falls through empties."""
    if not traces:
        return []
    if mode == "first":
        for t in traces:
            steps = list(t.get("reasoning_steps") or [])
            if steps:
                return steps
        return []
    if mode == "majority":
        mlabel = majority_label(traces)
        if mlabel:
            for t in traces:
                if t.get("label") == mlabel:
                    steps = list(t.get("reasoning_steps") or [])
                    if steps:
                        return steps
        for t in traces:
            steps = list(t.get("reasoning_steps") or [])
            if steps:
                return steps
        return []
    raise ValueError(f"unknown raw_mode: {mode}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--craft_dir", required=True,
                    help="CRAFT pipeline output directory (contains k_traces_*.json and synthesized_traces.json)")
    ap.add_argument("--source", required=True,
                    help="Source dataset JSON (FLD.json / FOLIO.json / ...) for hypothesis+premises lookup")
    ap.add_argument("--output", required=True, help="Output JSON in ReCEval schema")
    ap.add_argument("--raw_mode", choices=["first", "majority"], default="first",
                    help="How to pick the 'raw CoT' baseline from k_traces (default: first)")
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()

    craft_dir = Path(args.craft_dir)
    # Find k_traces file (k=5 or k=10)
    k_trace_files = sorted(craft_dir.glob("k_traces_*_samples.json"))
    if not k_trace_files:
        raise FileNotFoundError(f"No k_traces_*_samples.json in {craft_dir}")
    k_path = k_trace_files[0]
    synth_path = craft_dir / "synthesized_traces.json"
    if not synth_path.exists():
        raise FileNotFoundError(f"{synth_path} missing")

    print(f"[adapter] k_traces  : {k_path.name}")
    print(f"[adapter] synth     : {synth_path.name}")
    print(f"[adapter] source    : {args.source}")
    print(f"[adapter] raw_mode  : {args.raw_mode}")

    with open(k_path) as f:
        k_data = json.load(f)
    with open(synth_path) as f:
        synth_data = json.load(f)
    with open(args.source) as f:
        src_data = json.load(f)

    # Index source by original_index / metadata for FOLIO, by source_index for FLD
    src_by_idx: dict[int, dict] = {}
    for i, s in enumerate(src_data):
        src_by_idx[i] = s

    synth_by_id = {r["sample_id"]: r for r in synth_data["results"]}

    out_items: list[dict] = []
    n_raw_empty = n_synth_empty = n_src_missing = 0

    for rec in k_data["results"]:
        sid = rec["sample_id"]
        src_idx = rec.get("source_index")
        source = src_by_idx.get(src_idx)
        if source is None:
            n_src_missing += 1
            continue
        hyp, prem = extract_hypothesis_and_premises(source)

        raw_steps = pick_raw_trace(rec.get("traces") or [], args.raw_mode)
        if not raw_steps:
            n_raw_empty += 1

        synth_rec = synth_by_id.get(sid)
        synth_steps: list[str] = []
        if synth_rec and synth_rec.get("synthesized_trace"):
            synth_steps = split_synthesized_text(synth_rec["synthesized_trace"])
        if not synth_steps:
            n_synth_empty += 1

        # Skip samples where either side is empty (can't compare)
        if not raw_steps or not synth_steps:
            continue

        out_items.append({
            "id": sid,
            "dataset": rec.get("source_dataset", ""),
            "hypothesis": hyp,
            "question": prem,
            "proof_label": rec.get("target_answer", ""),
            "with_answer": {"steps": synth_steps},  # CRAFT post
            "blind":       {"steps": raw_steps},    # raw CoT
        })
        if args.max_samples and len(out_items) >= args.max_samples:
            break

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_items, f, indent=2, ensure_ascii=False)

    print(f"[adapter] wrote {len(out_items)} paired items → {out_path}")
    print(f"[adapter] skipped: raw_empty={n_raw_empty}  synth_empty={n_synth_empty}  src_missing={n_src_missing}")


if __name__ == "__main__":
    main()
