#!/usr/bin/env python3
"""
Adapter: CRAFT pipeline outputs -> FineLogic's step-evaluation schema.

Writes one file per side of the comparison:

    <output_dir>/{dataset}_raw.json      the first of the K candidate traces
    <output_dir>/{dataset}_craft.json    the synthesized trace

which is what finelogic_eval_steps.py reads. Each record is

    {"problem": {"input", "proof_label", "original_data"},
     "responses": [{"model", "prompt_style", "response"}]}

`original_data.steps` is the dataset's own step count, which the table builder
uses to restrict to the 10–20 step band FineLogic reports on.

Usage:
    python finelogic_adapter_craft.py \\
        --craft_dir  craft_runs/fld_nano \\
        --dataset    dataset/reasoning_traces_quality/finelogic/FLD.json \\
        --output_dir finelogic/fld_nano
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
try:
    from config import resolve_input, resolve_output
except ImportError:
    resolve_input = resolve_output = Path


def load_records(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return raw.get("results", raw) if isinstance(raw, dict) else raw


def original_data_for(source: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """FineLogic reads the reference step count from here; FLD stores it as a string."""
    od = (source or {}).get("original_data")
    if isinstance(od, dict):
        if "steps" not in od:
            od = {**od, "steps": "unknown"}
        return od
    steps = (source or {}).get("steps")
    return {"steps": steps if steps is not None else "unknown"}


def record(problem_input: str, label: Any, source: Optional[Dict[str, Any]],
           response: str, model: str, prompt_style: str, sample_id: Any) -> Dict[str, Any]:
    return {
        "problem": {
            "input": problem_input,
            "proof_label": label,
            "original_data": original_data_for(source),
        },
        "responses": [{"model": model, "prompt_style": prompt_style, "response": response}],
        "_sample_id": sample_id,
    }


def raw_response(trace: Dict[str, Any]) -> str:
    """The trace as generated — FineLogic parses the Step headers out of it itself."""
    text = trace.get("raw_response") or trace.get("reasoning_text") or ""
    if text:
        return text
    steps = trace.get("reasoning_steps") or []
    return "\n".join(f"Step {i}: {s}" for i, s in enumerate(steps, 1))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--craft_dir", required=True,
                    help="CRAFT run directory (k_traces_*_samples.json + synthesized*.json)")
    ap.add_argument("--dataset", required=True,
                    help="The FLD.json / FOLIO.json the run was generated from")
    ap.add_argument("--output_dir", default="CRAFT_evaluation_results/reasoning_traces_quality/FineLogic",
                    help="Where the {dataset}_{raw,craft}.json pair is written; "
                         "a relative path resolves under the results root")
    ap.add_argument("--synth_file", default=None,
                    help="Synthesis output (default: the one synthesized*.json in --craft_dir)")
    ap.add_argument("--trace_idx", type=int, default=0,
                    help="Which of the K traces is the raw baseline (default: the first)")
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Cap the pairs written — these evaluations run on a sample")
    args = ap.parse_args()

    craft_dir = Path(resolve_input(args.craft_dir))
    k_files = sorted(craft_dir.glob("k_traces_*_samples.json"))
    if not k_files:
        raise FileNotFoundError(f"No k_traces_*_samples.json in {craft_dir}")
    if args.synth_file:
        synth_path = Path(resolve_input(args.synth_file))
    else:
        synth_files = sorted(craft_dir.glob("synthesized*.json"))
        if not synth_files:
            raise FileNotFoundError(f"No synthesized*.json in {craft_dir}")
        synth_path = synth_files[0]

    dataset_path = Path(resolve_input(args.dataset))
    source_rows = load_records(dataset_path)
    by_input = {s["input"]: s for s in source_rows if s.get("input")}

    with open(k_files[0], encoding="utf-8") as f:
        k_blob = json.load(f)
    k_records = k_blob.get("results", k_blob) if isinstance(k_blob, dict) else k_blob
    gen_model = (k_blob.get("metadata", {}) or {}).get("model", "unknown") \
        if isinstance(k_blob, dict) else "unknown"
    synth_by_id = {r["sample_id"]: r for r in load_records(synth_path) if "sample_id" in r}

    raw_out: List[Dict[str, Any]] = []
    craft_out: List[Dict[str, Any]] = []
    n_no_raw = n_no_craft = 0

    for rec in k_records:
        sid = rec.get("sample_id")
        problem_input = rec.get("problem_text") or ""
        source = by_input.get(problem_input)

        traces = rec.get("traces") or []
        raw_text = raw_response(traces[min(args.trace_idx, len(traces) - 1)]) if traces else ""
        synth = synth_by_id.get(sid) or {}
        craft_text = synth.get("synthesized_trace") or ""

        if not raw_text:
            n_no_raw += 1
        if not craft_text:
            n_no_craft += 1
        # Both sides must exist: a sample scored on one side only would shift the
        # before/after comparison by changing which samples each side averages over.
        if not raw_text or not craft_text:
            continue
        if args.max_samples and len(raw_out) >= args.max_samples:
            break

        label = rec.get("target_answer")
        raw_out.append(record(problem_input, label, source, raw_text,
                              gen_model, "raw_cot", sid))
        craft_out.append(record(problem_input, synth.get("ground_truth", label), source,
                                craft_text, gen_model, "rkg_synthesis", sid))

    stem = dataset_path.stem
    out_dir = Path(resolve_output(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    for setting, rows in (("raw", raw_out), ("craft", craft_out)):
        path = out_dir / f"{stem}_{setting}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"[adapter] {len(rows)} samples → {path}")
    print(f"[adapter] skipped: no_raw={n_no_raw} no_craft={n_no_craft}")


if __name__ == "__main__":
    main()
