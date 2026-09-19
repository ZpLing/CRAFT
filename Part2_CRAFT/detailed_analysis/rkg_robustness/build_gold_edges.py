#!/usr/bin/env python3
"""
build_gold_edges.py — FLD's annotated proofs, as traces plus their true edges.

The robustness check needs graphs a model extracted and the graph that is
actually right, over the same steps. FLD ships the proof for each problem, so
both come from it: this renders each proof as a numbered reasoning trace and
records the dependencies that proof states.

Rendering the trace here rather than scoring a generated one is what makes the
comparison well defined. Node ids are assigned while the trace is written, so
`Step3` means the same step in the gold edges and in whatever a model extracts
from that text. Scoring a generated trace against FLD's proof would compare two
different step numberings and report the mismatch as extraction error.

FLD writes a proof as
    Step1: fact8 -> int1: <text>   Step2: int1 & fact10 -> int2: <text>   ...
so an antecedent is either a fact of the problem or something an earlier step
derived, and the second is resolved to that step rather than assumed to be the
step of the same number. Proofs by assumption open with
    Step1: void -> assump1: Let's assume ...
whose antecedent list is empty by design — that step introduces an assumption
instead of deriving anything, so it has no incoming edges. `int` and `assump`
are separate numbering schemes, and int1 is not assump1.

Two files come out:
    <out>/gold_traces.json   k_traces schema, one trace per sample, for build_rkg
    <out>/gold_edges.json    {sample_id: [[src, dst], ...]}

Usage:
    python build_gold_edges.py --dataset FLD_with_proofs.json --output_dir rkg_robustness/gold
    python ../../framework/module2_rkg_construction/build_rkg.py \\
        --input rkg_robustness/gold/gold_traces.json --model <backbone> \\
        --output rkg_robustness/<backbone>.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from config import resolve_input, resolve_output
except ImportError:
    resolve_input = resolve_output = Path

# "Step1:" / "Step 1:" — the marker that starts each proof step.
_STEP_SPLIT = re.compile(r"\bStep\s*(\d+)\s*:", re.IGNORECASE)
# "<antecedents> -> <conclusion id>: <text>", or "-> hypothesis" on the last step.
_BODY = re.compile(
    r"^(?P<ante>.*?)->\s*(?P<concl>hypothesis|(?:int|assump)\s*\d+)\s*:?\s*(?P<text>.*)$",
    re.IGNORECASE | re.DOTALL)
_REF = re.compile(r"\b(fact|int|assump)\s*(\d+)\b", re.IGNORECASE)
# "void" is FLD's way of saying a step rests on nothing, which an assumption does.
_VOID = re.compile(r"^\s*void\s*$", re.IGNORECASE)


def parse_proof(nl_solution: str) -> Optional[List[Dict[str, Any]]]:
    """Parse FLD's proof into steps, or None when it does not parse cleanly.

    Returning None rather than a partial parse matters: a proof that half-parses
    would contribute a truncated gold graph, and every model would then be
    penalised for edges the gold simply lost.
    """
    if not nl_solution or not nl_solution.strip():
        return None
    parts = _STEP_SPLIT.split(nl_solution)
    if len(parts) < 3:
        return None
    steps: List[Dict[str, Any]] = []
    # parts is [preamble, num, body, num, body, ...]
    for num_s, body in zip(parts[1::2], parts[2::2]):
        # a body runs to the next Step marker; strip the trailing separator
        body = body.strip().rstrip(";").strip()
        m = _BODY.match(body)
        if not m:
            return None
        ante_text = m.group("ante")
        antecedents = [(kind.lower(), int(idx)) for kind, idx in _REF.findall(ante_text)]
        if not antecedents and not _VOID.match(ante_text):
            return None
        concl = m.group("concl").lower().replace(" ", "")
        steps.append({
            "n": int(num_s),
            "antecedents": antecedents,
            "conclusion": concl,          # "intK" or "hypothesis"
            "text": " ".join(m.group("text").split()),
        })
    if not steps:
        return None
    # The numbering has to be 1..N with no gaps, or "Step3" in the rendered trace
    # would not be the third step of it.
    if [s["n"] for s in steps] != list(range(1, len(steps) + 1)):
        return None
    return steps


def facts_of(sample: Dict[str, Any]) -> Dict[int, str]:
    """{1: 'fact1 text', ...} from the problem statement."""
    raw = (sample.get("original_data") or {}).get("facts") or sample.get("Facts") or ""
    out: Dict[int, str] = {}
    hits = list(re.finditer(r"\bfact\s*(\d+)\s*:", str(raw), re.IGNORECASE))
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(str(raw))
        out[int(m.group(1))] = " ".join(str(raw)[m.end():end].split())
    return out


def producers(steps: List[Dict[str, Any]]) -> Dict[Tuple[str, int], int]:
    """(kind, index) -> the step number that derived it.

    int1 is whatever step concluded int1, which need not be step 1; and int1 and
    assump1 are different things, so the key carries the kind. The rendered
    citations and the gold edges both resolve through this, or the text a model
    reads would name different steps than the gold it is scored against.
    """
    out: Dict[Tuple[str, int], int] = {}
    for s in steps:
        concl = s["conclusion"]
        for kind in ("int", "assump"):
            if concl.startswith(kind):
                out[(kind, int(concl[len(kind):]))] = s["n"]
                break
    return out


def cite(kind: str, idx: int, produced_by: Dict[Tuple[str, int], int]) -> Optional[str]:
    if kind == "fact":
        return f"Fact{idx}"
    step = produced_by.get((kind, idx))
    return f"Step{step}" if step is not None else None


def render(steps: List[Dict[str, Any]], facts: Dict[int, str],
           hypothesis: str, label: Optional[str]) -> Tuple[str, List[str]]:
    """Write the proof as a trace, citing each step's antecedents the way traces do."""
    produced_by = producers(steps)
    lines: List[str] = []
    for s in steps:
        cites = ", ".join(c for kind, i in s["antecedents"]
                          if (c := cite(kind, i, produced_by)) is not None)
        suffix = f" ({cites})" if cites else ""
        if s["conclusion"] == "hypothesis":
            body = f"Therefore the hypothesis holds: {hypothesis}"
            if label:
                body += f" Final conclusion: {label}"
        else:
            body = s["text"]
        lines.append(f"Step {s['n']}: {body}{suffix}")
    fact_block = "\n".join(f"Fact{i}: {t}" for i, t in sorted(facts.items()))
    return fact_block + ("\n\n" if fact_block else "") + "\n".join(lines), lines


def gold_edges(steps: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """The dependencies the proof states, as (src, dst) over the rendered ids."""
    produced_by = producers(steps)
    edges: List[Tuple[str, str]] = []
    for s in steps:
        dst = f"Step{s['n']}"
        for kind, idx in s["antecedents"]:
            src = cite(kind, idx, produced_by)
            # None means the proof cites something it never established: the edge
            # has no source, and the rendered trace leaves that citation out too,
            # so the two stay in step.
            if src is not None:
                edges.append((src, dst))
    return edges


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    help="An FLD file carrying nl_solution (the annotated proof)")
    ap.add_argument("--output_dir", required=True,
                    help="Where gold_traces.json and gold_edges.json are written")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--id_prefix", default="FLD", help="Prefix for the generated sample ids")
    args = ap.parse_args()

    with open(resolve_input(args.dataset), encoding="utf-8") as f:
        raw = json.load(f)
    rows = raw.get("results", raw) if isinstance(raw, dict) else raw

    traces: List[Dict[str, Any]] = []
    edges_by_id: Dict[str, List[List[str]]] = {}
    n_no_proof = n_unparsed = 0

    for idx, sample in enumerate(rows):
        nl = sample.get("nl_solution")
        if not nl:
            n_no_proof += 1
            continue
        steps = parse_proof(nl)
        if steps is None:
            n_unparsed += 1
            continue
        facts = facts_of(sample)
        hypothesis = ((sample.get("original_data") or {}).get("hypothesis")
                      or sample.get("Conclusion") or "")
        label = sample.get("proof_label")
        text, step_lines = render(steps, facts, hypothesis, label)
        sample_id = f"{args.id_prefix}_{idx}"

        traces.append({
            "sample_id": sample_id,
            "source_dataset": Path(args.dataset).name,
            "source_index": idx,
            "target_answer": label,
            "problem_text": sample.get("input") or sample.get("Facts") or "",
            "domain": "logical",
            "traces": [{
                "trace_idx": 0,
                "label": label,
                "reasoning_text": text,
                "raw_response": text,
                "reasoning_steps": step_lines,
                "temperature": 0.0,
            }],
            "num_traces": 1,
            "requested_traces": 1,
            "errors": None,
        })
        edges_by_id[sample_id] = [list(e) for e in gold_edges(steps)]
        if args.max_samples and len(traces) >= args.max_samples:
            break

    out_dir = Path(resolve_output(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gold_traces.json").write_text(json.dumps(
        {"metadata": {"model": "FLD annotation", "k": 1,
                      "source": str(args.dataset), "n_samples": len(traces)},
         "results": traces}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "gold_edges.json").write_text(
        json.dumps(edges_by_id, indent=2), encoding="utf-8")

    n_edges = sum(len(v) for v in edges_by_id.values())
    n_fact = sum(1 for v in edges_by_id.values() for s, _ in v if s.startswith("Fact"))
    print(f"\n  {len(traces)} proofs rendered → {out_dir}/gold_traces.json")
    print(f"  {n_edges:,} gold edges ({n_fact:,} from facts, {n_edges - n_fact:,} between steps)"
          f" → {out_dir}/gold_edges.json")
    print(f"  mean {n_edges / len(traces):.1f} edges per proof" if traces else "")
    if n_no_proof or n_unparsed:
        print(f"  skipped: {n_no_proof} without nl_solution, {n_unparsed} that did not parse")


if __name__ == "__main__":
    main()
