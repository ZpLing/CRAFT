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

`premise` and `hypothesis` are read by the dataset's own adapter in
dataset_adapters.py, since none of the benchmark's four sets stores that pair the
way ROSCOE's own sets did. They are the scorer's reference fields and never enter
a prompt; Module I's loader is what keeps them out of generation.

The three metrics the paper reports — Grammar, Rep-Step, Rep-Word — are
reference-free and defined on any trace, and the scorer selects them on its own
when a set carries no reference chain, which none of these four does.

Usage:
    python roscoe_adapter_craft.py \\
        --craft_dir  results/craft_runs/fld_nano \\
        --dataset    dataset/FLD.json \\
        --output_dir roscoe_craft/fld_nano
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_adapters import adapt


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


def index_by_text(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """{problem text -> its record}, over whichever field the set states it in."""
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        for key in ("input", "premise", "question"):
            text = row.get(key)
            if isinstance(text, str) and text:
                out.setdefault(text, row)
                break
    return out


def load_source(source: Path) -> Dict[str, List[Dict[str, Any]]]:
    """{dataset name -> its records}, indexed the way Module I enumerated them.

    Takes ROSCOE's directory of .jsonl sets, the benchmark's .json ones, or a
    single file of either kind, so one run directory can be exported whichever
    data it was generated from.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    if source.is_dir():
        paths = sorted(source.glob("*.jsonl")) + sorted(source.glob("*.json"))
    else:
        paths = [source]
    for path in paths:
        out[path.stem] = load_records(path)
    if not out:
        raise FileNotFoundError(f"No .jsonl or .json dataset files in {source}")
    return out


def build_entry(src: Dict[str, Any], dataset: str, steps: List[str],
                setting: str, sample_id: str) -> Dict[str, Any]:
    """One ROSCOE line. `gpt-3` is the field the scorer reads the trace from.

    ROSCOE's own sets already carry `premise` / `hypothesis`; a benchmark set
    does not, and its adapter is what says which of its fields play those parts.
    """
    if "premise" in src or "hypothesis" in src:
        premise, hypothesis, answer = (src.get("premise", ""),
                                       src.get("hypothesis", ""),
                                       src.get("answer", ""))
    else:
        problem = adapt(src, dataset)
        premise, hypothesis, answer = (problem.premises, problem.hypothesis,
                                       problem.answer)
    entry: Dict[str, Any] = {
        "key":        src.get("key", sample_id),
        "premise":    premise,
        "hypothesis": hypothesis,
        "answer":     answer,
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
    ap.add_argument("--dataset", default="dataset",
                    help="The data the run was generated from: the dataset directory "
                         "or a single file in it")
    ap.add_argument("--output_dir", default="CRAFT_results/reasoning_traces_quality/ROSCOE",
                    help="Where the {dataset}_{raw,craft}.jsonl pairs are written; "
                         "a relative path resolves under the results root")
    ap.add_argument("--synth_file", default=None,
                    help="Synthesis output to read (default: the one synthesized*.json in --craft_dir)")
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Cap the pairs written per dataset")
    ap.add_argument("--raw_traces", default=None,
                    help="traces.jsonl of the baseline the CRAFT trace is compared against "
                         "(matched by sample_id). Without it the raw side is the run's own "
                         "first candidate trace, which compares CRAFT to the traces it was "
                         "built from rather than to a baseline anyone else would run")
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

    # The raw side, when it is a baseline's own generations rather than ours.
    baseline_raw: Dict[str, List[str]] = {}
    if args.raw_traces:
        for row in load_records(Path(resolve_input(args.raw_traces))):
            sid = row.get("sample_id")
            traces = row.get("traces") or []
            if sid and traces:
                baseline_raw[sid] = split_synthesized_text(traces[0]) or [
                    s.strip() for s in str(traces[0]).split("\n") if s.strip()]
        print(f"[adapter] raw side from baseline: {len(baseline_raw)} traces")
    synth_by_id = {r["sample_id"]: r for r in load_records(synth_path) if "sample_id" in r}

    export_dir = Path(resolve_output(args.output_dir))
    export_dir.mkdir(parents=True, exist_ok=True)

    # Two files can hold the same samples in a different order, so a sample is
    # found by its problem text first and only then by the position the run
    # recorded — matching by position alone pairs a trace with someone else's
    # problem and reports nothing missing.
    by_text = {name: index_by_text(rows) for name, rows in source.items()}

    pairs: Dict[str, List[tuple]] = defaultdict(list)
    n_raw_empty = n_craft_empty = n_src_missing = n_by_position = 0

    for rec in k_records:
        dataset = Path(str(rec.get("source_dataset", ""))).stem
        rows = source.get(dataset)
        if rows is None:
            n_src_missing += 1
            continue
        src_row = by_text.get(dataset, {}).get(rec.get("problem_text") or "")
        if src_row is None:
            idx = rec.get("source_index")
            if idx is None or idx >= len(rows):
                n_src_missing += 1
                continue
            src_row = rows[idx]
            n_by_position += 1

        raw_steps = (baseline_raw.get(rec.get("sample_id"))
                     if baseline_raw else first_nonempty_trace(rec.get("traces")))
        raw_steps = raw_steps or []
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
        pairs[dataset].append((src_row, raw_steps, craft_steps, rec["sample_id"]))

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
    if n_by_position:
        print(f"[adapter] WARNING: {n_by_position} samples matched by position, not by "
              f"problem text — check that --dataset is the data the run was generated from")


if __name__ == "__main__":
    main()
