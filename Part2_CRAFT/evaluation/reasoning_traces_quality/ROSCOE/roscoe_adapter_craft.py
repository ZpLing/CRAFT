#!/usr/bin/env python3
"""
Adapter: CRAFT pipeline outputs -> ROSCOE export schema.

Pairs the raw CoT baseline -- the model's own full response from the CoT run
the main table reports (baseline_results/<model>/cot/results.json,
`raw_response`), never one of CRAFT's K sampled traces -- with CRAFT's
synthesized trace, deduplicated the way the reported traces are, and writes
one file per dataset per setting:

    <output_dir>/{dataset}_raw.jsonl      raw CoT
    <output_dir>/{dataset}_craft.jsonl    CRAFT post-processed

which is what roscoe_score.py reads. Each line carries ROSCOE's own fields —
`premise`, `hypothesis`, `gpt-3` (the trace it scores) — plus the step
boundaries the generator used, so step-level analysis need not guess them back.

`premise` and `hypothesis` are read by the dataset's own adapter in
dataset_adapters.py, since none of the benchmark's four sets stores that pair the
way ROSCOE's own sets did. They are the scorer's reference fields and never enter
a prompt; Module I's loader is what keeps them out of generation.

The thirteen metrics the paper reports are
reference-free and defined on any trace, and the scorer selects them on its own
when a set carries no reference chain, which none of these four does.

Usage:
    python roscoe_adapter_craft.py \\
        --craft_dir  results/craft_runs/fld_nano \\
        --dataset    dataset/FLD.json \\
        --raw_model  gpt-5.4-nano \\
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


from framework.domain_optimization.math_text import (  # noqa: E402
    normalize_math, protect_factorials, split_steps)
# The CRAFT side is scored as it is reported: deduplicated by dedup_trace,
# checked against the answer reader the cell is scored with.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "framework" / "module3_topology_guided_synthesis"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "label_prediction"))
from dedup_trace import dedup_trace  # noqa: E402
from extract_label import extract_label, extract_math_answer  # noqa: E402

_READER = {"FLD": extract_label, "ProofWriter": extract_label,
           "OmniMATH": extract_math_answer, "OlympiadBench": extract_math_answer}


def split_synthesized_text(text: str) -> List[str]:
    """Split a trace into its steps.

    'Step 1: ... Step 2: ...' is what this pipeline's generator writes, but a
    baseline passed through --raw_traces writes prose, and re.split hands back
    the whole string when its pattern never matches. That one-element list is
    truthy, so the caller's fallback never ran and such a trace reached ROSCOE
    as a single step -- where the step-level metrics, all of them defined over
    pairs of steps, award a one-step chain a free 1.0 and the comparison
    against a nine-step trace stops meaning anything. So when no marker is
    found the trace is split on its own line breaks, and failing those, on
    sentence ends.
    """
    if not text:
        return []
    # The pipeline's own cut, so a display block on the lines after a "Step
    # N:" header is scored with its step. Every trailer stays: a scorer sees
    # the text as written, "Final Answer: \\boxed{7}" included.
    parts = split_steps(text, keep_conclusion_lines=True, keep_trailers=True)
    if len(parts) > 1:
        return parts
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) > 1:
        return lines
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return sentences if len(sentences) > 1 else parts


def load_records(path: Path) -> List[Dict[str, Any]]:
    """Read a .json list or a .jsonl stream, both of which appear in this pipeline.

    A run file keeps its rows under `results`; a baseline's results.json keeps
    them under `predictions`.
    """
    with open(path, encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        raw = json.load(f)
    if isinstance(raw, dict):
        return raw.get("results") or raw.get("predictions") or raw
    return raw


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


# The problem's own notation. Every OmniMATH and OlympiadBench statement writes
# its mathematics between dollar signs and none uses markdown, while a trace
# from gpt-5.4-nano writes \( ... \) -- 136 of them in a typical OmniMATH trace
# against 11 in its raw CoT -- and both models bold a rule or a result with
# asterisks. ROSCOE scores the text as written: its word alignment measures
# each token of the trace against the problem's tokens, and its grammar model
# reads "**if someone is blue then they eat the cow**" as a sentence. Neither
# the delimiter nor the asterisks is part of the reasoning, so both sides of a
# comparison are rendered in the notation the problem uses before scoring.
# Markdown bold: the opening marker is not glued to a letter or digit on its
# left and the closing one not on its right, so "x**2 + 2**n" -- two powers
# written the Python way -- is not read as bold "2 + 2".
_BOLD = re.compile(r"(?<![A-Za-z0-9])\*\*(?=\S)(.+?)(?<=\S)\*\*(?![A-Za-z0-9])", re.DOTALL)
# Spacing tells a power from a marker: x**2 has none on either side, 2 ** 3
# has it on both; "every** box" and ": ** Using" have it on one side only.
_POWER = re.compile(r"(?<=[A-Za-z0-9)\]}])\*\*(?=[A-Za-z0-9(\[{\\-])|(?<=[A-Za-z0-9)\]}]) \*\* (?=[A-Za-z0-9(\[{\\-])")


def normalize_markup(text: str) -> str:
    """Render a step in the problem's notation: no markdown, $-delimited maths.

    A factorial's "!" is also marked as not ending a sentence, because the
    scorer splits sentences with punkt, which ends one at every "!": a trace
    about $n!!\\mid 2012!!$ otherwise scores as dozens of one-word sentences.
    The same is done to the problem text, so both sides are read alike.
    """
    # A "**" between two operands -- x**2, 2 ** 3, a**-1 -- is a power written
    # the Python way and is arithmetic, not markup; it is set aside before
    # the bold rules run and put back after (the spacing rule is on _POWER).
    powers: list = []

    def _keep(m: re.Match) -> str:
        powers.append(m.group(0))
        return f"\x00{len(powers) - 1}\x00"
    text = _POWER.sub(_keep, text)
    text = _BOLD.sub(r"\1", text)
    # A bold marker that lost its partner -- "Step 2: ** Using the ..." -- is
    # markup with nothing to mark; it goes too.
    text = re.sub(r"\*\*(?=\s)|(?<=\s)\*\*", "", text)   # beside a space: the space stays
    text = re.sub(r"\*\*", " ", text)                    # glued on both sides: becomes a space
    text = re.sub(r"(?<=\S) {2,}(?=\S)", " ", text)
    text = re.sub(r"\x00(\d+)\x00", lambda m: powers[int(m.group(1))], text)
    return protect_factorials(normalize_math(text))


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
    steps = [normalize_markup(s) for s in steps]
    entry: Dict[str, Any] = {
        "key":        src.get("key", sample_id),
        "premise":    protect_factorials(premise),
        "hypothesis": protect_factorials(hypothesis),
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
    ap.add_argument("--raw_model", default=None,
                    help="Model whose CoT run is the raw side: reads "
                         "baseline_results/<model>/cot/results.json (predictions[].raw_response, "
                         "matched by sample_id)")
    ap.add_argument("--raw_traces", default=None,
                    help="Instead of --raw_model, a baseline file to read the raw side from: "
                         "a cot results.json (predictions[].raw_response) or a traces.jsonl "
                         "(traces[0]), matched by sample_id")
    args = ap.parse_args()
    if not args.raw_model and not args.raw_traces:
        # The raw side used to default to the run's own first sampled trace,
        # cut to its "Step N:" lines. That compared CRAFT to an outline of the
        # traces it was built from -- for gpt-5.4-nano on mathematics, 19% of
        # the characters -- and not to the CoT anyone reports. There is no
        # sensible default to fall back to, so the choice is made explicit.
        ap.error("one of --raw_model or --raw_traces is required: the raw side is the "
                 "model's own CoT response, not a trace of this run")

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

    # The raw side: the baseline's own generations, matched by sample_id. A cot
    # results.json keeps them under predictions[].raw_response; a traces.jsonl
    # under traces[0]. The full response is cut with the pipeline's own splitter
    # (prose falls back to lines, then sentences), so a display block is never
    # dropped from the raw side either.
    raw_path = Path(resolve_input(args.raw_traces)) if args.raw_traces else Path(
        resolve_input(f"baseline_results/{args.raw_model}/cot/results.json"))
    baseline_raw: Dict[str, List[str]] = {}
    for row in load_records(raw_path):
        sid = row.get("sample_id")
        text = row.get("raw_response") or ((row.get("traces") or [""])[0])
        if isinstance(text, dict):
            text = text.get("raw_response") or text.get("text") or ""
        if sid and str(text).strip():
            baseline_raw[sid] = split_synthesized_text(str(text))
    print(f"[adapter] raw side from {raw_path}: {len(baseline_raw)} traces")
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

        raw_steps = baseline_raw.get(rec.get("sample_id")) or []
        if not raw_steps:
            n_raw_empty += 1
        synth = synth_by_id.get(rec.get("sample_id"))
        craft_text = (synth or {}).get("synthesized_trace") or ""
        if craft_text.strip() and dataset in _READER:
            craft_text = dedup_trace(craft_text, extractor=_READER[dataset])
        craft_steps = split_synthesized_text(craft_text)
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
