#!/usr/bin/env python3
"""
Adapter: CRAFT pipeline outputs -> ROSCOE export schema.

Pairs the raw CoT baseline (the first of the K candidate traces) with CRAFT's
synthesized trace and writes one file per dataset per setting:

    <output_dir>/{dataset}_raw.jsonl      raw CoT
    <output_dir>/{dataset}_craft.jsonl    CRAFT post-processed

which is what roscoe_score.py reads. Each line carries ROSCOE's own fields —
`premise`, `hypothesis`, `gpt-3` (the trace it scores) — plus the step
boundaries the generator used, so step-level analysis need not guess them back.

ROSCOE's sets are built for verifying a shipped trace, so `hypothesis` is the
correct answer for CosmosQA and DROP and the reference solution for GSM8K. It is
copied through for the scorer's reference-based metrics, never into the prompt;
Module I's loader is what keeps it out of generation.

Usage:
    python roscoe_adapter_craft.py \\
        --craft_dir  results/roscoe_craft/nano \\
        --dataset    dataset/reasoning_traces_quality/roscoe \\
        --output_dir roscoe_craft/nano
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
try:
    from config import resolve_input, resolve_output
except ImportError:  # running outside the part
    resolve_input = resolve_output = Path


_STEP_RE = re.compile(r"(?m)^\s*Step\s*\d+\s*[:.\-]\s*")


def split_synthesized_text(text: str) -> List[str]:
    """A synthesized trace is one string, 'Step 1: ... Step 2: ...' — split it."""
    if not text:
        return []
    return [p.strip() for p in _STEP_RE.split(text) if p.strip()]


def first_nonempty_trace(traces: List[Dict[str, Any]]) -> List[str]:
    """Steps of the first candidate trace that has any — the raw CoT baseline."""
    for t in traces or []:
        steps = list(t.get("reasoning_steps") or [])
        if steps:
            return steps
    return []


def load_records(path: Path) -> List[Dict[str, Any]]:
    """Read a .json list or a .jsonl stream, both of which appear in this pipeline."""
    with open(path, encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        raw = json.load(f)
    return raw.get("results", raw) if isinstance(raw, dict) else raw


def load_source(source_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """{dataset name -> its records}, indexed the way Module I enumerated them."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for path in sorted(source_dir.glob("*.jsonl")):
        out[path.stem] = load_records(path)
    if not out:
        raise FileNotFoundError(f"No ROSCOE .jsonl files in {source_dir}")
    return out


def build_entry(src: Dict[str, Any], dataset: str, steps: List[str],
                setting: str, sample_id: str) -> Dict[str, Any]:
    """One ROSCOE line. `gpt-3` is the field the scorer reads the trace from."""
    entry: Dict[str, Any] = {
        "key":        src.get("key", sample_id),
        "premise":    src.get("premise", ""),
        "hypothesis": src.get("hypothesis", ""),
        "answer":     src.get("answer", ""),
        "gpt-3":      " ".join(steps),
        "steps":      steps,
        "dataset":    dataset,
        "setting":    setting,
    }
    if dataset == "esnli":
        for i in (1, 2, 3):
            key = f"explanation_{i}"
            if src.get(key):
                entry[key] = src[key]
    return entry


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--craft_dir", required=True,
                    help="CRAFT run directory (k_traces_*_samples.json + synthesized*.json)")
    ap.add_argument("--dataset", default="dataset/reasoning_traces_quality/roscoe",
                    help="Directory of the ROSCOE .jsonl sets the run was generated from")
    ap.add_argument("--output_dir", default="CRAFT_results/reasoning_traces_quality/ROSCOE",
                    help="Where the {dataset}_{raw,craft}.jsonl pairs are written; "
                         "a relative path resolves under the results root")
    ap.add_argument("--synth_file", default=None,
                    help="Synthesis output to read (default: the one synthesized*.json in --craft_dir)")
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Cap the pairs written per dataset")
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

    source = load_source(Path(resolve_input(args.dataset)))
    k_records = load_records(k_files[0])
    synth_by_id = {r["sample_id"]: r for r in load_records(synth_path) if "sample_id" in r}

    export_dir = Path(resolve_output(args.output_dir))
    export_dir.mkdir(parents=True, exist_ok=True)

    pairs: Dict[str, List[tuple]] = defaultdict(list)
    n_raw_empty = n_craft_empty = n_src_missing = 0

    for rec in k_records:
        dataset = Path(str(rec.get("source_dataset", ""))).stem
        rows = source.get(dataset)
        idx = rec.get("source_index")
        if rows is None or idx is None or idx >= len(rows):
            n_src_missing += 1
            continue

        raw_steps = first_nonempty_trace(rec.get("traces"))
        if not raw_steps:
            n_raw_empty += 1
        synth = synth_by_id.get(rec.get("sample_id"))
        craft_steps = split_synthesized_text((synth or {}).get("synthesized_trace") or "")
        if not craft_steps:
            n_craft_empty += 1
        # Both sides have to exist, or the delta for this sample compares a trace
        # against nothing and still lands in the mean.
        if not raw_steps or not craft_steps:
            continue
        if args.max_samples and len(pairs[dataset]) >= args.max_samples:
            continue
        pairs[dataset].append((rows[idx], raw_steps, craft_steps, rec["sample_id"]))

    for dataset, items in sorted(pairs.items()):
        for setting, which in (("raw", 1), ("craft", 2)):
            path = export_dir / f"{dataset}_{setting}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for src, raw_steps, craft_steps, sid in items:
                    steps = raw_steps if which == 1 else craft_steps
                    f.write(json.dumps(build_entry(src, dataset, steps, setting, sid),
                                       ensure_ascii=False) + "\n")
        print(f"[adapter] {dataset}: {len(items)} paired traces → "
              f"{dataset}_{{raw,craft}}.jsonl")

    total = sum(len(v) for v in pairs.values())
    print(f"[adapter] wrote {total} pairs across {len(pairs)} datasets → {export_dir}")
    print(f"[adapter] skipped: raw_empty={n_raw_empty} craft_empty={n_craft_empty} "
          f"src_missing={n_src_missing}")


if __name__ == "__main__":
    main()
