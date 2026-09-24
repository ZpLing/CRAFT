#!/usr/bin/env python3
"""
build_rkg.py  (Module II — Consensus RKG Construction)
------------------------------------------------------------------------------
Build a Reasoning Knowledge Graph (RKG) for each reasoning trace, then construct
a label-weighted consensus RKG via edge-frequency voting across k traces.

Node types:
  - fact       : given premises from the problem (no incoming edges)
  - step       : intermediate reasoning step
  - conclusion : final conclusion step (last step in trace)

Edge type:
  - uses       : step B directly depends on the conclusion of step/fact A

Core pipeline:
  1. Extract Fact nodes from problem text
  2. Single LLM call per trace to extract all dependency edges
  3. Regex fallback (when LLM fails)
  4. Validate RKG acyclicity (detect cycles, dangling references)
  5. Build Consensus RKG across k traces via edge-frequency voting (BuildRKG)

Output format:
  {
    "sample_id": "...",
    "trace_rkgs": [
      {
        "trace_idx": 0,
        "nodes": [{"id": "Fact1", "type": "fact", "text": "...", "step_number": null}, ...],
        "edges": [{"src": "Fact1", "dst": "Step2", "type": "uses", "confidence": 0.95}, ...],
        "is_acyclic": true,
        "extraction_method": "llm" | "regex_fallback"
      }
    ],
    "consensus_rkg": {
      "nodes": [...],
      "edges": [...],
      "edge_frequencies": {"Fact1->Step2": 0.9, ...}
    }
  }
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp
import backoff
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    class _DummyTqdm:
        def __init__(self, *a, **k): pass
        def update(self, *a, **k): pass
        def close(self): pass
    def tqdm(*a, **k): return _DummyTqdm()

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from framework.module1_generation_filtering.steps_filter import parse_steps_from_trace

#########################
# Configuration — loaded from root config.py; change models there
#########################
import importlib.util as _ilu, pathlib as _pl
_cfg_path = _pl.Path(__file__).resolve().parents[2] / "config.py"
_spec = _ilu.spec_from_file_location("_root_config", _cfg_path)
_cfg  = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_cfg)

OPENAI_API_KEY  = _cfg.OPENAI_API_KEY
OPENAI_BASE_URL = _cfg.OPENAI_BASE_URL
DEFAULT_MODEL   = _cfg.MODEL_RKG_BUILD   # Step 3.2: Build Consensus RKG
REQUEST_TIMEOUT = getattr(_cfg, "REQUEST_TIMEOUT", 120)
RESPONSE_TOKENS = 2048

HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json",
}
CHAT_URL = OPENAI_BASE_URL.rstrip("/") + "/chat/completions"

# Regex: explicit step/fact references
_REF_STEP = re.compile(r'\bstep\s*(\d+)\b', re.IGNORECASE)
_REF_FACT = re.compile(r'\bfact\s*(\d+)\b', re.IGNORECASE)
_FACT_LINE = re.compile(r'^(?:fact\s*(\d+)\s*[:\.\-])\s*(.+)$', re.IGNORECASE | re.MULTILINE)
_GIVEN_LINE = re.compile(r'^(?:given|let|assume|suppose)[:\s]+(.+)$', re.IGNORECASE | re.MULTILINE)


#########################
# Trace selection for the consensus
#########################

# The scorer's matcher, so "the traces that agree" means agreement about the
# answer rather than about how it is spelled. \frac{1}{2}, \dfrac{1}{2} and 0.5
# are one answer here and three answers to a string count.
import sys as _sys
_EVAL_DIR = str(_pl.Path(__file__).resolve().parents[2] / "evaluation" / "label_prediction")
if _EVAL_DIR not in _sys.path:
    _sys.path.insert(0, _EVAL_DIR)
from answer_match import answers_match as _answers_match      # noqa: E402
from extract_label import extract_pred as _extract_pred       # noqa: E402


def select_majority_traces(
    traces: List[Dict[str, Any]],
    domain: str = "logical",
    dataset: Optional[str] = None,
    answer_type: Optional[str] = None,
) -> List[int]:
    """Indices of the traces that reach the majority answer.

    The consensus used to be built over all k traces whatever they concluded, and
    on 69.5% of Omni-MATH samples that meant merging traces that had reached
    different answers. Node identity in the consensus is the step's position, so
    such a merge takes step 1 from the chain that ended at 1995 and step 3 from
    the chain that ended at 1996 and calls the result one derivation. Module III
    then re-derives along a path no trace ever took, which is why its answer was
    worse than a plain vote over the same traces in every agreement bucket.

    Restricting the consensus to the traces that agree costs nothing — they are
    already generated — and leaves the pipeline's shape untouched: Module I still
    filters, Module II still votes on edges, Module III still walks the graph.
    Ties keep the first group, matching the vote's own tie-break.

    Returns every index when no answer can be read, so a sample is never left
    without a consensus.
    """
    answers: List[Tuple[int, str]] = []
    for i, t in enumerate(traces):
        text = t.get("raw_response") or t.get("reasoning_text") or ""
        if not text and t.get("reasoning_steps"):
            text = "\n".join(str(s) for s in t["reasoning_steps"])
        pred = t.get("label") if domain == "logical" else None
        pred = pred or _extract_pred(text, domain)
        if pred:
            answers.append((i, pred))
    if len(answers) < 2:
        return list(range(len(traces)))

    groups: List[List[int]] = []
    reps: List[str] = []
    for i, a in answers:
        for g, rep in zip(groups, reps):
            try:
                same = _answers_match(a, rep, dataset=dataset, answer_type=answer_type)
            except Exception:
                same = (a == rep)
            if same:
                g.append(i)
                break
        else:
            groups.append([i])
            reps.append(a)

    best = max(groups, key=len)
    return sorted(best)


#########################
# Fact Node Extraction
#########################

def extract_facts_from_problem(problem_text: str, domain: str = "logical") -> List[Dict[str, Any]]:
    """Extract Fact nodes from problem text.

    Returns:
        List of {"id": "Fact1", "type": "fact", "text": "...", "step_number": None}
    """
    facts = []

    if domain == "logical":
        for m in _FACT_LINE.finditer(problem_text):
            idx = int(m.group(1))
            text = m.group(2).strip()
            facts.append({
                "id": f"Fact{idx}",
                "type": "fact",
                "text": text,
                "step_number": None,
            })

    else:  # math domain
        # "Given: ..." / "Let x = ..." etc.
        for m in _GIVEN_LINE.finditer(problem_text):
            text = m.group(1).strip()
            if text:
                n = len(facts) + 1
                facts.append({
                    "id": f"Given{n}",
                    "type": "fact",
                    "text": text,
                    "step_number": None,
                })
        # If no given statements found, treat the entire problem as a single implicit Fact node
        if not facts and problem_text.strip():
            facts.append({
                "id": "Problem",
                "type": "fact",
                "text": problem_text.strip()[:300],
                "step_number": None,
            })

    return facts


#########################
# Regex Fallback Edge Extraction
#########################

def extract_rkg_edges_regex_fallback(
    steps: List[Dict[str, Any]],
    facts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Use regex to match explicit step/fact references and build dependency edges.

    Returns:
        List of {"src": "Fact1", "dst": "Step3", "type": "uses", "confidence": 0.6}
    """
    fact_ids = {f["id"] for f in facts}
    step_id_map = {s["step_number"]: s["id"] for s in steps}
    edges = []
    seen = set()

    for step in steps:
        dst_id = step["id"]
        text = step["text"]

        # Referenced step numbers
        for m in _REF_STEP.finditer(text):
            n = int(m.group(1))
            src_id = step_id_map.get(n)
            if src_id and src_id != dst_id:
                key = (src_id, dst_id)
                if key not in seen:
                    seen.add(key)
                    edges.append({"src": src_id, "dst": dst_id, "type": "uses", "confidence": 0.6})

        # Referenced fact numbers
        for m in _REF_FACT.finditer(text):
            n = int(m.group(1))
            src_id = f"Fact{n}"
            if src_id in fact_ids:
                key = (src_id, dst_id)
                if key not in seen:
                    seen.add(key)
                    edges.append({"src": src_id, "dst": dst_id, "type": "uses", "confidence": 0.6})

    return edges


#########################
# LLM Edge Extraction
#########################

_DEP_PROMPT_TEMPLATE = """\
You are analyzing a step-by-step reasoning trace. Your task is to identify the DIRECT dependencies between steps.

For each reasoning step, identify which facts or EARLIER steps it DIRECTLY depends on to reach its conclusion.
- Only list DIRECT dependencies (not transitive ones).
- A step depends on another step if it uses that step's conclusion as a premise.
- Do NOT list a step as depending on itself.
- If a step uses no earlier steps or facts, set "uses" to [].
{confidence_rule}
Output ONLY valid JSON with this exact structure:
{{
  "dependencies": [
    {{"step_id": "Step1", "uses": []{confidence_example_empty}}},
    {{"step_id": "Step2", "uses": ["Fact1", "Step1"]{confidence_example_two}}},
    {{"step_id": "Step3", "uses": ["Step2"]{confidence_example_one}}}
  ]
}}

Available facts (given premises):
{facts_block}

Reasoning steps:
{steps_block}

Respond with ONLY the JSON, no other text."""


_REASONING_MODELS = ("o4-mini", "o4-mini-2025-04-16", "deepseek-r1")


def _apply_reasoning_budget(payload: dict, model: str) -> dict:
    """Reasoning models bill hidden reasoning against max_tokens, so a budget sized
    for the visible answer comes back empty. Raise the total and reserve a visible slice."""
    if model in _REASONING_MODELS:
        payload["max_tokens"] = max(payload.get("max_tokens") or 0, 16000)
        payload["max_output_tokens"] = min(payload["max_tokens"], 4096)
    return payload


# Every LLM request this module makes passes through one function, so counting
# there is the number of calls actually paid for — retries included — rather than
# the number a run was expected to need. Each stage writes it into its own
# metadata as api_calls, which is where the cost of a run is read from.
API_CALLS = {"count": 0}


@backoff.on_exception(
    backoff.expo,
    (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError),
    max_tries=5,
    factor=2,
)


async def _call_llm_json(session: aiohttp.ClientSession, prompt: str, model: str) -> Optional[Dict]:
    """Call the LLM and parse the JSON response. Returns None on failure."""
    API_CALLS["count"] += 1
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": RESPONSE_TOKENS,
    }
    _apply_reasoning_budget(payload, model)
    async with session.post(
        CHAT_URL, json=payload, headers=HEADERS,
        timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
    ) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {await resp.text()[:200]}")
        data = await resp.json()
        msg = data["choices"][0]["message"]
        # o4-mini / deepseek-r1 put reasoning in reasoning_content, not content
        if model in ("o4-mini", "o4-mini-2025-04-16", "deepseek-r1"):
            content = (msg.get("reasoning_content") or msg.get("content") or "").strip()
        else:
            content = (msg.get("content") or "").strip()

    # Extract JSON — LLM sometimes wraps it in ```json ... ```
    json_match = re.search(r'\{[\s\S]+\}', content)
    if not json_match:
        return None
    try:
        return json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return None


async def extract_rkg_edges_with_llm(
    session: aiohttp.ClientSession,
    steps: List[Dict[str, Any]],
    facts: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
) -> Tuple[List[Dict[str, Any]], str]:
    """Extract all dependency edges for a trace in a single LLM call.

    Returns:
        (edges, method)  method = "llm" | "regex_fallback"
    """
    if not steps:
        return [], "regex_fallback"

    facts_block = "\n".join(
        f"  {f['id']}: {f['text'][:120]}" for f in facts
    ) or "  (no explicit facts given)"

    steps_block = "\n".join(
        f"  {s['id']} (Step {s['step_number']}): {s['text'][:STEP_TEXT_CHARS]}"
        for s in steps
    )

    # --edge_confidence llm: the model rates each dependency it lists, so W(e)
    # carries a judgement rather than the constant 0.9 every extracted edge got.
    fmt = dict(
        confidence_rule=('- For each entry also give "confidence": a list with one number in [0, 1] '
                         'per item of "uses", how sure you are that the step really depends on it. '
                         'Use the whole scale: 1.0 only when the step names or quotes that fact or '
                         'step and its conclusion could not be reached without it; about 0.7 when the '
                         'step clearly uses its content without naming it; about 0.4 when the link is '
                         'plausible but the step would stand without it; below 0.3 when you are '
                         'guessing.\n'),
        confidence_example_empty=', "confidence": []',
        confidence_example_two=', "confidence": [0.95, 0.7]',
        confidence_example_one=', "confidence": [0.9]',
    )
    # legacy (pre-2026-09-23, EDGE_CONFIDENCE == "constant"): no confidence was asked for and
    # every listed edge got 0.9 below. Kept for reference:
    # fmt = dict(confidence_rule="", confidence_example_empty="",
    #            confidence_example_two="", confidence_example_one="")
    prompt = _DEP_PROMPT_TEMPLATE.format(
        facts_block=facts_block,
        steps_block=steps_block,
        **fmt,
    )

    try:
        result = await _call_llm_json(session, prompt, model)
    except Exception:
        result = None

    if result is None or "dependencies" not in result:
        # fallback
        return extract_rkg_edges_regex_fallback(steps, facts), "regex_fallback"

    # Parse LLM output
    step_ids = {s["id"] for s in steps}
    fact_ids = {f["id"] for f in facts}
    valid_src_ids = step_ids | fact_ids

    edges = []
    seen = set()
    for dep in result.get("dependencies", []):
        dst_id = dep.get("step_id", "")
        if dst_id not in step_ids:
            continue
        uses = dep.get("uses", []) or []
        confs = dep.get("confidence")   # legacy: `None` when EDGE_CONFIDENCE == "constant"
        if not (isinstance(confs, list) and len(confs) == len(uses)):
            confs = None
        for i, src_id in enumerate(uses):
            if src_id not in valid_src_ids or src_id == dst_id:
                continue
            # Prevent forward references (src step_number > dst step_number)
            dst_step = next((s for s in steps if s["id"] == dst_id), None)
            src_step = next((s for s in steps if s["id"] == src_id), None)
            if src_step and dst_step and src_step["step_number"] > dst_step["step_number"]:
                continue  # skip forward references
            key = (src_id, dst_id)
            if key not in seen:
                seen.add(key)
                conf = 0.9   # only when the model omitted the number (legacy constant)
                if confs is not None:
                    try:
                        conf = min(1.0, max(0.0, float(confs[i])))
                    except (TypeError, ValueError):
                        conf = 0.9
                edges.append({"src": src_id, "dst": dst_id, "type": "uses", "confidence": conf})

    if not edges:
        return extract_rkg_edges_regex_fallback(steps, facts), "regex_fallback"

    return edges, "llm"


#########################
# RKG Validation
#########################

def check_rkg_acyclic(nodes: List[Dict], edges: List[Dict]) -> bool:
    """Kahn's algorithm: detect whether the directed graph is acyclic."""
    node_ids = {n["id"] for n in nodes}
    in_degree = {nid: 0 for nid in node_ids}
    adj = defaultdict(list)

    for e in edges:
        src, dst = e["src"], e["dst"]
        if src in node_ids and dst in node_ids:
            adj[src].append(dst)
            in_degree[dst] += 1

    queue = deque(nid for nid in node_ids if in_degree[nid] == 0)
    visited = 0
    while queue:
        nid = queue.popleft()
        visited += 1
        for neighbor in adj[nid]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return visited == len(node_ids)


def remove_weakest_edges_to_break_cycles(
    nodes: List[Dict], edges: List[Dict]
) -> List[Dict]:
    """If the RKG contains cycles, remove lowest-confidence edges until acyclic."""
    sorted_edges = sorted(edges, key=lambda e: e.get("confidence", 0.5))
    remaining = list(edges)
    for e in sorted_edges:
        if check_rkg_acyclic(nodes, remaining):
            break
        remaining.remove(e)
    return remaining


#########################
# Per-Trace RKG Construction
#########################

async def build_rkg_for_trace(
    session: aiohttp.ClientSession,
    trace: Dict[str, Any],
    problem_text: str,
    model: str = DEFAULT_MODEL,
    domain: str = "logical",
) -> Dict[str, Any]:
    """Build complete RKG for a single trace."""
    facts = extract_facts_from_problem(problem_text, domain=domain)
    parsed_steps = parse_steps_from_trace(trace)

    if not parsed_steps:
        return {
            "nodes": facts,
            "edges": [],
            "is_acyclic": True,
            "extraction_method": "empty",
        }

    # Identify the last step (conclusion)
    max_step_num = max(s["step_number"] for s in parsed_steps)

    # Build step nodes
    step_nodes = []
    for s in parsed_steps:
        node_type = "conclusion" if s["step_number"] == max_step_num else "step"
        step_nodes.append({
            "id": f"Step{s['step_number']}",
            "type": node_type,
            "text": s["step_text"],
            "step_number": s["step_number"],
        })

    all_nodes = facts + step_nodes

    # Extract edges
    edges, method = await extract_rkg_edges_with_llm(session, step_nodes, facts, model=model)

    # Validate and fix cycles
    is_acyclic = check_rkg_acyclic(all_nodes, edges)
    if not is_acyclic:
        edges = remove_weakest_edges_to_break_cycles(all_nodes, edges)
        is_acyclic = True  # guaranteed acyclic after repair

    return {
        "nodes": all_nodes,
        "edges": edges,
        "is_acyclic": is_acyclic,
        "extraction_method": method,
    }


#########################
# Cross-Trace Step Alignment
#########################

_ALIGN_PROMPT_TEMPLATE = """\
You are given {k} reasoning traces for the same problem. Your task is to identify \
which steps across different traces are SEMANTICALLY EQUIVALENT — meaning they reach \
the same intermediate conclusion through the same logical operation, even if worded differently.

Rules:
- Only group steps that genuinely express the same reasoning sub-goal.
- A step that has no equivalent in ANY other trace gets its own singleton group.
- Do NOT force alignment — it is fine if most steps are singletons.
- Each step must appear in exactly one group.

Output ONLY valid JSON:
{{
  "groups": [
    {{"canonical_id": "C1", "members": [{{"trace": 0, "step": "Step2"}}, {{"trace": 1, "step": "Step3"}}]}},
    {{"canonical_id": "C2", "members": [{{"trace": 0, "step": "Step4"}}]}},
    ...
  ]
}}

{traces_block}

Respond with ONLY the JSON."""


async def align_steps_across_traces(
    session: aiohttp.ClientSession,
    all_trace_steps: List[List[Dict[str, Any]]],
    model: str = DEFAULT_MODEL,
) -> Dict[Tuple[int, str], str]:
    """Ask the LLM to align semantically equivalent steps across k traces.

    Returns a mapping (trace_idx, original_step_id) -> canonical_id.
    Steps with no cross-trace match get a unique canonical ID T{t}_S{n}.
    Falls back to identity mapping (no alignment) on LLM failure.
    """
    k = len(all_trace_steps)
    if k == 0:
        return {}

    # Build the traces block for the prompt
    trace_lines = []
    for t_idx, steps in enumerate(all_trace_steps):
        trace_lines.append(f"=== Trace {t_idx} ===")
        for s in steps:
            text_preview = s["text"][:150].replace("\n", " ")
            trace_lines.append(f"  {s['id']}: {text_preview}")
    traces_block = "\n".join(trace_lines)

    prompt = _ALIGN_PROMPT_TEMPLATE.format(k=k, traces_block=traces_block)

    try:
        result = await _call_llm_json(session, prompt, model)
    except Exception:
        result = None

    # Build identity fallback mapping
    identity: Dict[Tuple[int, str], str] = {}
    for t_idx, steps in enumerate(all_trace_steps):
        for s in steps:
            identity[(t_idx, s["id"])] = f"T{t_idx}_{s['id']}"

    if result is None or "groups" not in result:
        return identity

    # Parse groups → mapping
    mapping: Dict[Tuple[int, str], str] = {}
    seen_members: set = set()

    for group in result.get("groups", []):
        cid = group.get("canonical_id", "")
        if not cid:
            continue
        for member in group.get("members", []):
            t = member.get("trace")
            sid = member.get("step", "")
            if t is None or not sid:
                continue
            t = int(t)
            if 0 <= t < k and (t, sid) not in seen_members:
                mapping[(t, sid)] = cid
                seen_members.add((t, sid))

    # Fill in any missing steps with unique IDs
    for t_idx, steps in enumerate(all_trace_steps):
        for s in steps:
            key = (t_idx, s["id"])
            if key not in mapping:
                mapping[key] = f"T{t_idx}_{s['id']}"

    return mapping


def remap_rkg_with_alignment(
    rkg: Dict[str, Any],
    trace_idx: int,
    alignment: Dict[Tuple[int, str], str],
) -> Dict[str, Any]:
    """Remap step node IDs in a per-trace RKG using the alignment mapping.
    Fact nodes are never remapped (they're shared across all traces already).
    """
    def _remap(node_id: str) -> str:
        if node_id.startswith("Fact") or node_id.startswith("Given") or node_id == "Problem":
            return node_id  # Facts are already shared
        return alignment.get((trace_idx, node_id), f"T{trace_idx}_{node_id}")

    new_nodes = []
    for node in rkg.get("nodes", []):
        new_node = dict(node)
        new_node["id"] = _remap(node["id"])
        new_nodes.append(new_node)

    new_edges = []
    for edge in rkg.get("edges", []):
        new_edge = dict(edge)
        new_edge["src"] = _remap(edge["src"])
        new_edge["dst"] = _remap(edge["dst"])
        if new_edge["src"] != new_edge["dst"]:
            new_edges.append(new_edge)

    return {**rkg, "nodes": new_nodes, "edges": new_edges}


#########################
# Consensus RKG Construction (BuildRKG)
#########################

def _term_overlap_score(text_a: str, text_b: str) -> float:
    """Compute term overlap (Jaccard) between two step texts.
    Used to assist in validating LLM-extracted edges: if B's key terms
    include A's conclusion terms, the edge is more credible.
    """
    if not text_a or not text_b:
        return 0.0
    # Extract lowercase words of length > 3, filter stopwords
    _STOP = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
             "have", "has", "had", "this", "that", "and", "or", "but",
             "so", "then", "if", "we", "it", "its", "from", "to", "of",
             "in", "on", "at", "by", "for", "with", "can", "not", "no"}
    def _tok(text: str) -> set:
        return {w for w in re.findall(r'\b[a-z]{3,}\b', text.lower()) if w not in _STOP}
    set_a, set_b = _tok(text_a), _tok(text_b)
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# The ablation's "Embedding Cosine Similarity" row
# ---------------------------------------------------------------------------
# W(e) fuses the extractor's confidence with how much the two steps overlap, and
# the paper measures that overlap as Jaccard over content terms. The alternative
# the ablation reports is the cosine of two sentence embeddings, which scores two
# steps as related when they say the same thing in different words — and scores
# two steps as related when they merely talk about the same objects, which on a
# deduction is most pairs of steps in the problem. Both are computed here so the
# row is the same pipeline with one function swapped.
_EMBEDDER = None
_EMBED_CACHE: Dict[str, Any] = {}


# How an edge is admitted to the consensus RKG (--edge_rule):
#   "weighted" (default; Algorithm 1 of the paper): each trace's vote for an edge is
#       its fused weight W(e) = (1-lambda)*conf + lambda*Jaccard, and the edge is kept
#       when support(e) = sum W_S(e)*w_S / total weight >= theta. lambda and the
#       confidence therefore decide which edges survive.
#   "frequency" (legacy; the runs before 2026-09-23): the share of traces containing
#       the edge must be >= theta, and W(e) is stored but never thresholded. Kept only
#       to reproduce those runs and as the ablation's "w/o Weighted Edge Fusion".
# Characters of each step shown to the dependency extractor. 200 (legacy) cut most
# mathematics steps mid-equation; 500 covers a step of a few sentences.
STEP_TEXT_CHARS = 500
EDGE_RULE = "weighted"      # legacy value "frequency" is commented out below, not selectable
# Where an edge's confidence comes from (--edge_confidence): "llm" (default; the paper's
# "LLM-reported confidence") asks the model for a number in [0, 1] per dependency it
# lists. "constant" (legacy) gave 0.9 to every listed edge and 0.6 to a regex-fallback
# edge, which left W(e) with almost no spread (0.56-0.78 on the reported graphs).
EDGE_CONFIDENCE = "llm"     # legacy value "constant" is commented out below, not selectable


def _embedding_cosine_score(text_a: str, text_b: str) -> float:
    """Cosine of two sentence embeddings, in place of the term-overlap score.

    all-mpnet-base-v2 is the encoder, which is the one ROSCOE scores traces with,
    so the ablation is not also introducing a second embedding space. Vectors are
    cached by text: a trace's steps are compared repeatedly and the encoder is the
    slow part.
    """
    global _EMBEDDER
    if not text_a or not text_b:
        return 0.0
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    missing = [t for t in (text_a, text_b) if t not in _EMBED_CACHE]
    if missing:
        vecs = _EMBEDDER.encode(missing, normalize_embeddings=True,
                                show_progress_bar=False)
        for t, v in zip(missing, vecs):
            _EMBED_CACHE[t] = v
    import numpy as _np
    return float(_np.clip(_np.dot(_EMBED_CACHE[text_a], _EMBED_CACHE[text_b]), 0.0, 1.0))


# Which of the two the edge weight uses. build_rkg sets it from --similarity;
# everything downstream calls _similarity_score and does not know the difference.
_SIMILARITY = "jaccard"


def set_similarity(kind: str) -> None:
    """Choose the overlap measure for W(e): "jaccard" (paper) or "embedding"."""
    global _SIMILARITY
    if kind not in ("jaccard", "embedding"):
        raise ValueError(f"unknown similarity {kind!r}")
    _SIMILARITY = kind


def _similarity_score(text_a: str, text_b: str) -> float:
    if _SIMILARITY == "embedding":
        return _embedding_cosine_score(text_a, text_b)
    return _term_overlap_score(text_a, text_b)


def build_consensus_rkg(
    trace_rkgs: List[Dict[str, Any]],
    consensus_threshold: float = 0.3,
    node_threshold: Optional[float] = None,
    term_overlap_weight: float = 0.3,   # lambda, the edge-weight balance
    proved_threshold: Optional[float] = None,
    weight_by: str = "uniform",   # "uniform" | "step_count" | "gold_depth" | "support"
    expected_depth: Optional[int] = None,
    trace_weights_override: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    """Equal-weight edge-frequency voting across k trace RKGs to build the Consensus RKG.

    P1-A — Term overlap edge confidence fusion:
        For each LLM-extracted edge (src->dst), calibrate confidence using src.text/dst.text term overlap:
        Auxiliary confidence calibration:
        final_confidence = (1 - term_overlap_weight) * llm_confidence
                         + term_overlap_weight * overlap_score

    Args:
        trace_rkgs:            k trace RKG list (one per trace)
        consensus_threshold:   edge frequency threshold (edges >= this value are included in consensus)
        term_overlap_weight:   lambda, the edge-weight balance between the LLM's confidence
                               and the term-overlap score (paper: 0.3)

    Returns:
        {
          "nodes": [...],
          "edges": [...],           # high-frequency edges with frequency / confidence fields
          "edge_frequencies": {...},
          "node_texts": {...}
        }
    """
    if not trace_rkgs:
        return {"nodes": [], "edges": [], "edge_frequencies": {}, "node_texts": {}}

    # ── Trace weights ──────────────────────────────────────────────────────
    # "uniform"   : every trace counts as 1.0 (default — plain MV)
    # "step_count": weight ∝ #step+conclusion nodes; longer traces are more thorough
    #               and empirically more reliable on logical-deduction tasks.
    # "gold_depth": weight by how near a trace lands to the depth the dataset is
    #               drawn at, and only where that depth is a property of the
    #               selection rather than of the sample. On the depth-5
    #               ProofWriter slice a trace within two steps of it is right 90%
    #               of the time against 50% for one six steps over, and an equal
    #               vote spends the two the same.
    # "support"   : weight by how well the consensus supports the trace -- the mean
    #               W(e)-weighted support of the trace's edges once the graph has
    #               been built with equal weights. This is where the edge weight
    #               W(e) = (1-lambda)*conf + lambda*Jaccard reaches the vote:
    #               Module III weights each trace's answer by these weights, so a
    #               trace whose dependencies the other traces confirm counts more.
    if weight_by == "support" and trace_weights_override is None:
        first = build_consensus_rkg(trace_rkgs, consensus_threshold, node_threshold,
                                    term_overlap_weight, proved_threshold, "uniform",
                                    expected_depth)
        support = first.get("edge_support") or first.get("edge_frequencies") or {}
        derived: Dict[int, float] = {}
        for rkg_trace in trace_rkgs:
            keys = {f"{e['src']}->{e['dst']}" for e in rkg_trace.get("edges", [])}
            vals = [support.get(k, 0.0) for k in keys]
            derived[rkg_trace.get("trace_idx", 0)] = max(0.05, sum(vals) / len(vals)) if vals else 0.05
        mean_w = sum(derived.values()) / max(len(derived), 1)
        derived = {k: v / mean_w for k, v in derived.items()} if mean_w > 0 else derived
        return build_consensus_rkg(trace_rkgs, consensus_threshold, node_threshold,
                                   term_overlap_weight, proved_threshold, "uniform",
                                   expected_depth, trace_weights_override=derived)
    trace_weights: Dict[int, float] = {}
    for rkg_trace in trace_rkgs:
        tidx = rkg_trace.get("trace_idx", 0)
        n_steps = sum(1 for n in rkg_trace.get("nodes", [])
                      if n.get("type") in ("step", "conclusion"))
        # gold_depth compares the trace's own length, not the graph's
        n_trace_steps = rkg_trace.get("n_trace_steps") or n_steps
        if trace_weights_override is not None:
            trace_weights[tidx] = float(trace_weights_override.get(tidx, 1.0))
        elif weight_by == "step_count":
            # Use a small floor to avoid zero-weight on degenerate traces
            trace_weights[tidx] = float(max(n_steps, 1))
        elif weight_by == "gold_depth" and expected_depth:
            # Exponent chosen on the first 250 ProofWriter samples and checked on
            # the other 250, where it is worth 5.2 points over an equal vote
            # (0.584 -> 0.636). Sharper exponents score higher on that second
            # half, but picking one by that number is choosing on the set the
            # number is read from.
            trace_weights[tidx] = 1.0 / (1.0 + abs(n_trace_steps - expected_depth)) ** 4
        else:
            trace_weights[tidx] = 1.0

    total_weight = sum(trace_weights.get(rkg_trace.get("trace_idx", 0), 1.0) for rkg_trace in trace_rkgs)

    # ── Normalize conclusion node ids to a single canonical id ─────────────
    # Each trace has exactly one final conclusion (text contains __PROVED__/__DISPROVED__/__UNKNOWN__),
    # but its node id may be Step9 in trace A, Step10 in trace B, C9 in trace C. Without this rewrite,
    # the conclusion vote splits across distinct ids → consensus picks an arbitrary trace.
    _CONC_ID = "ConcShared"
    _LBL_PAT = re.compile(r"__(?:PROVED|DISPROVED|UNKNOWN)__")
    for rkg_trace in trace_rkgs:
        # Find all conclusion-like nodes in this trace
        concl_node_ids = {
            n["id"] for n in rkg_trace.get("nodes", [])
            if _LBL_PAT.search(n.get("text", ""))
        }
        if not concl_node_ids:
            continue
        # Pick the highest-step-number one as "the" conclusion
        nodes_with_step = [
            (n.get("step_number") or 0, n["id"])
            for n in rkg_trace.get("nodes", [])
            if n["id"] in concl_node_ids
        ]
        primary_id = max(nodes_with_step)[1]
        # Rewrite primary → ConcShared; drop other concl-text nodes (avoid duplicates)
        new_nodes = []
        for n in rkg_trace.get("nodes", []):
            if n["id"] == primary_id:
                nn = dict(n); nn["id"] = _CONC_ID; nn["type"] = "conclusion"
                new_nodes.append(nn)
            elif n["id"] in concl_node_ids:
                continue  # drop secondary conclusion nodes from this trace
            else:
                new_nodes.append(n)
        rkg_trace["nodes"] = new_nodes
        # Rewrite edges
        new_edges = []
        for e in rkg_trace.get("edges", []):
            src = _CONC_ID if e["src"] in concl_node_ids else e["src"]
            dst = _CONC_ID if e["dst"] in concl_node_ids else e["dst"]
            if src != dst:
                ee = dict(e); ee["src"] = src; ee["dst"] = dst
                new_edges.append(ee)
        rkg_trace["edges"] = new_edges

    # ── Build node text index (for P1-A term overlap) ──────────────────────
    all_node_texts: Dict[str, str] = {}
    node_trace_count: Dict[str, int] = defaultdict(int)
    for rkg_trace in trace_rkgs:
        for nid in {n["id"] for n in rkg_trace.get("nodes", [])}:
            node_trace_count[nid] += 1      # how many traces contain this node at all
    for rkg_trace in trace_rkgs:
        for node in rkg_trace.get("nodes", []):
            nid  = node["id"]
            text = node.get("text", "")
            if nid not in all_node_texts and text:
                all_node_texts[nid] = text  # first occurrence; later overridden by majority vote

    # ── Accumulate weighted edge frequencies ──────────────────────────────
    edge_weight_sum:      Dict[str, float] = defaultdict(float)
    edge_confidence_sum:  Dict[str, float] = defaultdict(float)
    edge_count:           Dict[str, int]   = defaultdict(int)

    for rkg_trace in trace_rkgs:
        tidx   = rkg_trace.get("trace_idx", 0)
        weight = trace_weights.get(tidx, 1.0)
        seen   = set()
        for e in rkg_trace.get("edges", []):
            src, dst = e["src"], e["dst"]
            key = f"{src}->{dst}"
            if key in seen:
                continue
            seen.add(key)

            # P1-A: fuse LLM confidence + term overlap
            llm_conf     = e.get("confidence", 0.7)
            overlap      = _similarity_score(
                all_node_texts.get(src, ""),
                all_node_texts.get(dst, ""),
            )
            fused_conf   = (1 - term_overlap_weight) * llm_conf + term_overlap_weight * overlap

            edge_weight_sum[key]     += weight
            edge_confidence_sum[key] += fused_conf * weight
            edge_count[key]          += 1

    # ── Accumulate node texts (majority-weighted version) ─────────────────
    node_text_wsum:  Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    node_type_votes: Dict[str, Counter]           = defaultdict(Counter)

    for rkg_trace in trace_rkgs:
        tidx   = rkg_trace.get("trace_idx", 0)
        weight = trace_weights.get(tidx, 1.0)
        for node in rkg_trace.get("nodes", []):
            nid  = node["id"]
            text = node.get("text", "")
            node_text_wsum[nid][text]          += weight
            node_type_votes[nid][node.get("type", "step")] += 1

    # consensus text = the version with the highest accumulated weight
    node_texts = {
        nid: max(wmap, key=lambda t: wmap[t])
        for nid, wmap in node_text_wsum.items()
    }
    node_types = {
        nid: counter.most_common(1)[0][0]
        for nid, counter in node_type_votes.items()
    }

    # ── Conclusion label aggregation (FIX: vote on label, not full text) ────
    # When a node is type "conclusion", its text variants may differ in wording
    # ("I conclude __DISPROVED__" vs "Final Conclusion: __DISPROVED__") and
    # split the vote. Re-tally __PROVED__/__DISPROVED__ across all traces and
    # rewrite the consensus text to use the majority label.
    #
    # Asymmetric voting: many models exhibit a class-bias (e.g. gpt-5.4-nano
    # over-predicts __DISPROVED__). We use a calibrated threshold τ on the
    # PROVED ratio: predict PROVED if (PROVED_weight / total) ≥ τ. With τ=0.5
    # this is plain majority vote; with τ<0.5 we counteract DISPROVED bias.
    for nid, counter in node_type_votes.items():
        if counter.most_common(1)[0][0] != "conclusion":
            continue
        label_w: Dict[str, float] = defaultdict(float)
        for text, w in node_text_wsum[nid].items():
            if "__PROVED__" in text:
                label_w["__PROVED__"] += w
            elif "__DISPROVED__" in text:
                label_w["__DISPROVED__"] += w
            elif "__UNKNOWN__" in text:
                label_w["__UNKNOWN__"] += w
        if not label_w:
            continue
        total_w = sum(label_w.values())
        p_ratio = label_w.get("__PROVED__", 0) / total_w if total_w else 0
        if proved_threshold is not None and "__UNKNOWN__" not in label_w:
            best = "__PROVED__" if p_ratio >= proved_threshold else "__DISPROVED__"
        else:
            best = max(label_w, key=lambda l: label_w[l])
        node_texts[nid] = f"Final Conclusion: {best}"

    # ── Filter high-weight edges ───────────────────────────────────────────
    edge_frequencies = {
        key: round(wsum / total_weight, 4)
        for key, wsum in edge_weight_sum.items()
    }
    # Weighted support: each trace's vote for an edge counts W_S(e) instead of 1.
    edge_support = {
        key: round(edge_confidence_sum[key] / total_weight, 4)
        for key in edge_weight_sum
    }
    admission = edge_support
    # legacy (pre-2026-09-23, EDGE_RULE == "frequency"): admission by the share of traces
    # containing the edge, W(e) stored but never thresholded. Kept for reference:
    # admission = edge_frequencies

    consensus_edges    = []
    consensus_node_ids: Set[str] = set()

    for key, freq in edge_frequencies.items():
        if admission[key] >= consensus_threshold:
            src, dst   = key.split("->", 1)
            avg_conf   = edge_confidence_sum[key] / edge_weight_sum[key] if edge_weight_sum[key] else 0.7
            consensus_edges.append({
                "src":       src,
                "dst":       dst,
                "type":      "uses",
                "confidence": round(avg_conf, 3),
                "frequency":  round(freq, 3),
                "support":    edge_support[key],
            })
            consensus_node_ids.add(src)
            consensus_node_ids.add(dst)

    # Node-frequency vote (beta). Collecting nodes only from edges that clear theta
    # loses every node of a sample whose traces disagree on structure: the graph comes
    # back empty even when a step is present in every single trace. A node carried by
    # at least beta of the traces belongs in the consensus on its own merit.
    _beta = consensus_threshold if node_threshold is None else node_threshold
    n_traces = max(len(trace_rkgs), 1)
    consensus_node_ids |= {
        nid for nid, cnt in node_trace_count.items()
        if cnt / n_traces >= _beta
    }

    # Fact nodes are always retained
    fact_ids = {nid for nid, ntype in node_types.items() if ntype == "fact"}
    consensus_node_ids |= fact_ids

    # Conclusion nodes are always retained (regardless of edge frequency).
    # When alignment is on, all conclusions share canonical id "ConcFinal".
    # When alignment is off, conclusion ids are positional (e.g. "Step10").
    conclusion_ids = {nid for nid, ntype in node_types.items() if ntype == "conclusion"}
    consensus_node_ids |= conclusion_ids
    # Also retain any incoming edges to conclusion nodes that meet a relaxed threshold,
    # to give the synthesizer something to anchor on.
    if conclusion_ids:
        relax = max(1.0 / max(len(trace_rkgs), 1), 0.05)  # at least 1/k of traces
        for key, freq in edge_frequencies.items():
            src, dst = key.split("->", 1)
            if dst in conclusion_ids and freq >= relax:
                if not any(e["src"] == src and e["dst"] == dst for e in consensus_edges):
                    avg_conf = (edge_confidence_sum[key] / edge_weight_sum[key]
                                if edge_weight_sum[key] else 0.7)
                    consensus_edges.append({
                        "src": src, "dst": dst, "type": "uses",
                        "confidence": round(avg_conf, 3),
                        "frequency":  round(freq, 3),
                    })
                    consensus_node_ids.add(src)

    consensus_nodes = []
    for nid in sorted(consensus_node_ids):
        if nid in node_texts:
            step_num = int(nid[4:]) if nid.startswith("Step") and nid[4:].isdigit() else None
            consensus_nodes.append({
                "id":          nid,
                "type":        node_types.get(nid, "step"),
                "text":        node_texts[nid],
                "step_number": step_num,
            })

    return {
        "nodes":           consensus_nodes,
        "edges":           consensus_edges,
        "edge_frequencies": edge_frequencies,
        # Only a weighted-rule graph publishes its support map; the second-pass filter
        # (steps_filter --method rkg) reads it in place of the frequencies when present.
        "edge_support": edge_support,
        "node_frequencies": {nid: round(c / n_traces, 4) for nid, c in node_trace_count.items()},
        "node_texts":      node_texts,
        "trace_weights":   trace_weights,   # uniform (all 1.0)
    }


#########################
# Sample-Level Batch Processing
#########################

async def build_rkgs_for_sample(
    session: aiohttp.ClientSession,
    sample: Dict[str, Any],
    model: str = DEFAULT_MODEL,
    domain: str = "logical",
    consensus_threshold: float = 0.3,
    node_threshold: Optional[float] = None,
    use_alignment: bool = False,
    edge_lambda: float = 0.3,
    weight_by: str = "uniform",
    expected_depth: Optional[int] = None,
    consensus_scope: str = "all",
    dataset_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Concurrently build per-trace RKGs for a sample, then compute the consensus RKG.

    When use_alignment=True, first runs a cross-trace step alignment LLM call to
    assign canonical node IDs to semantically equivalent steps, eliminating positional
    mismatch before edge-frequency voting.
    """
    traces = sample.get("cleaned_traces") or sample.get("traces", [])
    problem_text = (
        sample.get("problem_text")
        or sample.get("problem_input")
        or sample.get("input", "")
    )

    # Concurrently build per-trace RKGs
    tasks = [
        build_rkg_for_trace(session, trace, problem_text, model=model, domain=domain)
        for trace in traces
    ]
    rkg_results = await asyncio.gather(*tasks, return_exceptions=True)

    trace_rkgs = []
    for i, result in enumerate(rkg_results):
        if isinstance(result, Exception):
            trace_rkgs.append({
                "trace_idx":          i,
                "nodes":              [],
                "edges":              [],
                "is_acyclic":         True,
                "extraction_method":  "error",
                "error":              str(result),
            })
        else:
            result["trace_idx"] = i
            # The trace's own length, before the step filter shortened it, which
            # is what a depth weighting has to compare against. The graph's node
            # count is not it: extraction normalises the graphs to a similar size
            # (mean 8.15, sd 1.30 on ProofWriter) while the traces vary (mean
            # 5.99, sd 1.66), so weighting on nodes gave 2500 traces two distinct
            # weights and the weighting did nothing.
            result["n_trace_steps"] = (traces[i].get("original_num_steps")
                                       or len(traces[i].get("reasoning_steps") or []))
            trace_rkgs.append(result)

    valid_rkgs = [d for d in trace_rkgs if d.get("extraction_method") != "error"]

    # Every trace keeps its own graph in the output — the ablation's "consensus
    # over all traces" row is then a free offline rebuild — but only the traces
    # that agree on the answer are voted into the consensus. Merging chains that
    # ended at different answers, which is what "all" does on most samples,
    # produces a graph that describes no derivation any trace performed.
    consensus_rkgs = valid_rkgs
    scope_info: Dict[str, Any] = {"scope": consensus_scope}
    if consensus_scope == "majority" and len(valid_rkgs) > 1:
        keep = set(select_majority_traces(
            traces, domain=domain, dataset=dataset_name,
            answer_type=(traces[0] or {}).get("answer_type") if traces else None,
        ))
        scoped = [d for d in valid_rkgs if d.get("trace_idx") in keep]
        if scoped:
            consensus_rkgs = scoped
            scope_info["kept"] = sorted(keep)
            scope_info["dropped"] = len(valid_rkgs) - len(scoped)

    # Optional: cross-trace alignment before consensus voting
    alignment_info: Dict[str, Any] = {"used": False}
    if use_alignment and len(valid_rkgs) > 1:
        # Collect ONLY step nodes for LLM alignment (conclusion nodes share a canonical ID)
        all_trace_steps = [
            [n for n in rkg.get("nodes", []) if n.get("type") == "step"]
            for rkg in valid_rkgs
        ]
        alignment = await align_steps_across_traces(session, all_trace_steps, model=model)

        # Force all conclusion nodes across traces to share a single canonical ID.
        # Each trace has exactly one final conclusion → they should always be grouped.
        for rkg in valid_rkgs:
            t_idx = rkg["trace_idx"]
            for n in rkg.get("nodes", []):
                if n.get("type") == "conclusion":
                    alignment[(t_idx, n["id"])] = "ConcFinal"

        # Count how many steps got grouped with at least one other step
        from collections import Counter
        cid_counts = Counter(alignment.values())
        n_shared = sum(1 for v in cid_counts.values() if v > 1)
        n_total_steps = sum(len(s) for s in all_trace_steps)

        alignment_info = {
            "used":            True,
            "n_canonical_ids": len(cid_counts),
            "n_shared_groups": n_shared,
            "n_total_steps":   n_total_steps,
            "shared_ratio":    round(n_shared / max(len(cid_counts), 1), 3),
        }

        # Remap step IDs in each valid RKG
        for rkg in valid_rkgs:
            t_idx = rkg["trace_idx"]
            remapped = remap_rkg_with_alignment(rkg, t_idx, alignment)
            rkg.update(remapped)

    consensus = build_consensus_rkg(
        consensus_rkgs,
        consensus_threshold=consensus_threshold,
        node_threshold=node_threshold,
        term_overlap_weight=edge_lambda,
        weight_by=weight_by,
        expected_depth=expected_depth,
    )

    return {
        "sample_id":       sample.get("sample_id", "unknown"),
        "trace_rkgs":      trace_rkgs,
        "consensus_rkg":   consensus,
        "alignment":       alignment_info,
        "consensus_scope": scope_info,
    }


async def build_rkgs_for_dataset(
    input_file: Path,
    output_file: Path,
    model: str = DEFAULT_MODEL,
    concurrency: int = 10,
    domain: str = "logical",
    consensus_threshold: float = 0.3,
    node_threshold: Optional[float] = None,
    max_samples: Optional[int] = None,
    edge_lambda: float = 0.3,
    weight_by: str = "uniform",
    expected_depth: Optional[int] = None,
    consensus_scope: str = "all",
    dataset_name: Optional[str] = None,
    use_alignment: bool = False,
) -> None:
    """Build RKGs for all samples in a dataset and write output to rkg.json."""
    print(f"Reading file: {input_file}")
    with open(input_file, encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "results" in raw:
        samples = raw["results"]
    elif isinstance(raw, list):
        samples = raw
    else:
        samples = []

    if max_samples:
        samples = samples[:max_samples]

    print(f"Processing {len(samples)} samples | model: {model} | domain: {domain}")

    semaphore = asyncio.Semaphore(concurrency)
    results = [None] * len(samples)

    connector = aiohttp.TCPConnector(limit=concurrency * 3)
    async with aiohttp.ClientSession(connector=connector) as session:
        pbar = tqdm(total=len(samples), desc="Build RKG", unit="sample")

        async def worker(idx: int, sample: Dict) -> None:
            async with semaphore:
                try:
                    results[idx] = await build_rkgs_for_sample(
                        session, sample,
                        model=model, domain=domain,
                        consensus_threshold=consensus_threshold,
                        node_threshold=node_threshold,
                        edge_lambda=edge_lambda,
                        consensus_scope=consensus_scope,
                        dataset_name=dataset_name,
                        use_alignment=use_alignment,
                    )
                except Exception as e:
                    results[idx] = {
                        "sample_id": sample.get("sample_id", f"unknown_{idx}"),
                        "error": str(e),
                        "trace_rkgs": [],
                        "consensus_rkg": {},
                    }
                finally:
                    pbar.update(1)

        await asyncio.gather(*[
            asyncio.create_task(worker(i, s))
            for i, s in enumerate(samples)
        ])
        pbar.close()

    output_data = {
        "metadata": {
            "model": model,
            "domain": domain,
            "consensus_threshold": consensus_threshold,
            "edge_lambda": edge_lambda,
            "edge_rule": EDGE_RULE,
            "edge_confidence": EDGE_CONFIDENCE,
            "node_threshold": consensus_threshold if node_threshold is None else node_threshold,
            "total_samples": len(samples),
            "successful": sum(1 for r in results if r and "error" not in r),
            "api_calls": API_CALLS["count"],
        },
        "results": [r for r in results if r is not None],
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Results saved to: {output_file}")
    print(f"  Successful: {output_data['metadata']['successful']} / {len(samples)}")


#########################
# Offline consensus rebuild
#########################

def rebuild_consensus(
    input_file: Path,
    consensus_threshold: float = 0.3,
    node_threshold: Optional[float] = None,
    term_overlap_weight: float = 0.3,
    proved_threshold: Optional[float] = None,
    weight_by: str = "uniform",
    gt_file: Optional[Path] = None,
    expected_depth: Optional[int] = None,
    consensus_scope: str = "all",
    traces_file: Optional[Path] = None,
    domain: str = "logical",
    dataset: Optional[str] = None,
    use_alignment: bool = False,
    model: str = DEFAULT_MODEL,
    concurrency: int = 10,
) -> None:
    """Recompute consensus_rkg in an existing RKG file from its cached trace_rkgs.

    No LLM calls: the per-trace graphs are already stored, so a changed voting rule
    can be applied to a finished run without paying for extraction again. Rewrites
    the file in place, and with --gt_file reports how often the consensus conclusion
    node carries the right label.
    """
    import copy

    with open(input_file) as f:
        raw = json.load(f)
    results = raw.get("results", raw) if isinstance(raw, dict) else raw

    gt_map: Dict[str, Any] = {}
    if gt_file and Path(gt_file).exists():
        with open(gt_file) as f:
            gt_raw = json.load(f)
        gt_rows = gt_raw.get("results", gt_raw) if isinstance(gt_raw, dict) else gt_raw
        gt_map = {s["sample_id"]: s.get("target_answer") for s in gt_rows}

    correct = total = no_conclusion = 0
    matrix: Counter = Counter()

    # consensus_scope="majority" needs each trace's answer, which the cached
    # graphs do not carry; it comes from the trace file the run was built from.
    traces_map: Dict[str, List[Dict[str, Any]]] = {}
    if consensus_scope == "majority":
        if not traces_file or not Path(traces_file).exists():
            raise SystemExit("--consensus_scope majority needs --traces_file")
        with open(traces_file) as f:
            t_raw = json.load(f)
        t_rows = t_raw.get("results", t_raw) if isinstance(t_raw, dict) else t_raw
        traces_map = {
            s["sample_id"]: (s.get("cleaned_traces") or s.get("traces") or [])
            for s in t_rows
        }

    n_scoped = n_dropped = 0

    # Cross-trace alignment on the cached per-trace graphs: one LLM call per
    # sample, so a rebuilt consensus can vote over steps rather than positions
    # without extracting the graphs again.
    alignments: Dict[str, Dict[Tuple[int, str], str]] = {}
    if use_alignment:
        async def _align_all() -> None:
            sem = asyncio.Semaphore(concurrency)
            connector = aiohttp.TCPConnector(limit=concurrency)
            async with aiohttp.ClientSession(connector=connector) as session:
                async def one(r: Dict[str, Any]) -> None:
                    rkgs = [t for t in (r.get("trace_rkgs") or r.get("trace_dags") or [])
                            if t.get("extraction_method") != "error"]
                    if len(rkgs) < 2:
                        return
                    steps = [[n for n in t.get("nodes", []) if n.get("type") == "step"] for t in rkgs]
                    async with sem:
                        mapping = await align_steps_across_traces(session, steps, model=model)
                    for t in rkgs:
                        for n in t.get("nodes", []):
                            if n.get("type") == "conclusion":
                                mapping[(t["trace_idx"], n["id"])] = "ConcFinal"
                    alignments[r["sample_id"]] = mapping
                await asyncio.gather(*(one(r) for r in results))
        asyncio.run(_align_all())
        print(f"aligned {len(alignments)}/{len(results)} samples")

    for r in results:
        valid = [copy.deepcopy(t) for t in (r.get("trace_rkgs") or r.get("trace_dags") or [])
                 if t.get("extraction_method") != "error"]
        if not valid:
            r["consensus_rkg"] = {"nodes": [], "edges": []}
            continue
        mapping = alignments.get(r.get("sample_id"))
        if mapping:
            for t in valid:
                t.update(remap_rkg_with_alignment(t, t["trace_idx"], mapping))
            cid_counts = Counter(mapping.values())
            r["alignment"] = {
                "used": True,
                "n_canonical_ids": len(cid_counts),
                "n_shared_groups": sum(1 for v in cid_counts.values() if v > 1),
                "n_total_steps": sum(1 for k in mapping if k[1] != "ConcFinal"),
            }

        if consensus_scope == "majority":
            src_traces = traces_map.get(r.get("sample_id")) or []
            if len(src_traces) >= 2:
                keep = set(select_majority_traces(
                    src_traces, domain=domain, dataset=dataset,
                    answer_type=(src_traces[0] or {}).get("answer_type"),
                ))
                scoped = [t for t in valid if t.get("trace_idx") in keep]
                if scoped:
                    n_dropped += len(valid) - len(scoped)
                    n_scoped += 1
                    valid = scoped
            r["consensus_scope"] = consensus_scope
        # lambda and the node threshold travel with the rest. They used not to:
        # a rebuild always used build_consensus_rkg's defaults for both, so
        # --edge_lambda 0 rebuilt a graph byte-identical to --edge_lambda 0.3 —
        # node sets, edge sets and edge confidences all unchanged on 100 of 100
        # graphs. The ablation's "w/o Weighted Edges Fusion" row was therefore
        # scoring the full model against itself and reporting no difference,
        # which is exactly what a component with no effect would look like.
        consensus = build_consensus_rkg(
            valid,
            consensus_threshold=consensus_threshold,
            node_threshold=node_threshold,
            term_overlap_weight=term_overlap_weight,
            proved_threshold=proved_threshold,
            weight_by=weight_by,
            expected_depth=expected_depth,
        )
        r["consensus_rkg"] = consensus

        if not gt_map:
            continue
        label = None
        for n in consensus.get("nodes", []):
            if n.get("type") != "conclusion":
                continue
            text = n.get("text", "")
            if "__PROVED__" in text:
                label = "__PROVED__"; break
            if "__DISPROVED__" in text:
                label = "__DISPROVED__"; break
        gt = gt_map.get(r.get("sample_id"))
        if not label:
            no_conclusion += 1
        elif gt:
            total += 1
            correct += label == gt
            matrix[(label, gt)] += 1

    with open(input_file, "w") as f:
        json.dump({"results": results} if isinstance(raw, dict) and "results" in raw else results, f)
    print(f"Wrote: {input_file}")
    if total:
        print(f"Conclusion-node accuracy: {correct}/{total} = {correct / total:.3f}")
        print(f"No conclusion: {no_conclusion}")
        print(f"Matrix: {dict(matrix)}")


#########################
# CLI Entry Point
#########################

def main() -> None:
    parser = argparse.ArgumentParser(description="Build RKGs for reasoning traces")
    parser.add_argument("--input", required=True, help="Path to cleaned_traces.json")
    parser.add_argument("--output", default="rkg.json",
                        help="Output RKG JSON path (relative paths resolve under the results root)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"LLM model (default: {DEFAULT_MODEL})")
    parser.add_argument("--api_key", default=None, help="API Key")
    parser.add_argument("--base_url", default=None, help="API Base URL")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrency level (default 10)")
    parser.add_argument("--domain", default="logical", choices=["logical", "math"])
    parser.add_argument("--dataset_name", default=None,
                        choices=[None, "FLD", "ProofWriter", "OlympiadBench", "OmniMATH"],
                        help="Which dataset adapter compares two answers for --consensus_scope "
                             "majority; without it the comparison falls back to string equality")
    parser.add_argument("--consensus_threshold", type=float, default=0.3,
                        help="Edge frequency threshold theta (default 0.3)")
    parser.add_argument("--node_threshold", type=float, default=None,
                        help="Node frequency threshold beta (default: same as --consensus_threshold)")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--align", action="store_true",
                        help="Align steps across the K traces (one LLM call per sample) before "
                             "voting, so a node is a step and not a position: without it Step 2 "
                             "of every trace is one node whatever each trace did second, and the "
                             "synthesized trace re-derives what a collided node hides")
    parser.add_argument("--similarity", choices=["jaccard", "embedding"],
                        default="jaccard",
                        help="The overlap measure fused into W(e): term Jaccard "
                             "(the paper) or the cosine of all-mpnet-base-v2 "
                             "embeddings (the ablation's row)")
    parser.add_argument("--edge_rule", choices=["weighted"], default="weighted",
                        help="How an edge enters the consensus RKG (Algorithm 1): each trace's vote is "
                             "weighted by W(e) = (1-lambda)*conf + lambda*Jaccard and the edge is kept when "
                             "the weighted support is at least theta. The pre-2026-09-23 'frequency' rule "
                             "is commented out in build_consensus_rkg and no longer selectable")
    parser.add_argument("--edge_confidence", choices=["llm"], default="llm",
                        help="The model states a confidence in [0, 1] for each dependency it lists (the "
                             "paper's LLM-reported confidence). The pre-2026-09-23 constant 0.9 is "
                             "commented out in extract_rkg_edges and no longer selectable")
    parser.add_argument("--edge_lambda", type=float, default=0.3,
                        help="Edge weight balance lambda: W(e) = (1-lambda)*LLM confidence "
                             "+ lambda*term overlap (paper: 0.3). 0 drops the fusion, which "
                             "is the ablation's 'w/o Weighted Edges Fusion' setting")

    # Offline mode: re-vote an existing RKG file's cached trace_rkgs, no LLM calls.
    parser.add_argument("--rebuild_consensus", action="store_true",
                        help="Recompute consensus_rkg in --input in place from its cached "
                             "trace_rkgs (no LLM calls); --output is ignored")
    parser.add_argument("--proved_threshold", type=float, default=None,
                        help="--rebuild_consensus: asymmetric voting tau; predict PROVED when the "
                             "PROVED weight ratio >= tau (default: plain majority vote)")
    parser.add_argument("--expected_depth", type=int, default=None,
                        help="The proof depth this dataset is drawn at, for --weight_by "
                             "gold_depth. Only meaningful where the depth is a property of "
                             "the selection and not gold annotation about the sample")
    parser.add_argument("--weight_by", choices=["uniform", "step_count", "gold_depth", "support"],
                        default="uniform",
                        help="--rebuild_consensus: trace weighting; 'step_count' favors longer traces")
    parser.add_argument("--gt_file", default=None,
                        help="--rebuild_consensus: cleaned_with_problem.json, for conclusion-label diagnostics")
    parser.add_argument("--consensus_scope", choices=["all", "majority"], default="all",
                        help="Which traces the consensus is built over. 'all' merges every "
                             "trace whatever it concluded, which on 69.5%% of Omni-MATH "
                             "samples means merging traces that reached different answers "
                             "into one graph whose nodes are matched by step position. "
                             "'majority' keeps the traces that agree on the answer, so the "
                             "graph describes one derivation. Needs --traces_file")
    parser.add_argument("--traces_file", default=None,
                        help="--consensus_scope majority: the cleaned/k_traces file the run was "
                             "built from, read for each trace's answer")
    args = parser.parse_args()
    set_similarity(args.similarity)

    global OPENAI_API_KEY, OPENAI_BASE_URL, CHAT_URL, HEADERS, EDGE_RULE, EDGE_CONFIDENCE
    EDGE_RULE = args.edge_rule
    EDGE_CONFIDENCE = args.edge_confidence
    if args.api_key:
        OPENAI_API_KEY = args.api_key
        HEADERS["Authorization"] = f"Bearer {OPENAI_API_KEY}"
    if args.base_url:
        OPENAI_BASE_URL = args.base_url
        CHAT_URL = OPENAI_BASE_URL.rstrip("/") + "/chat/completions"

    if args.rebuild_consensus:
        rebuild_consensus(
            input_file=_cfg.resolve_input(args.input),
            consensus_threshold=args.consensus_threshold,
            node_threshold=args.node_threshold,
            term_overlap_weight=args.edge_lambda,
            proved_threshold=args.proved_threshold,
            weight_by=args.weight_by,
            expected_depth=args.expected_depth,
            gt_file=_cfg.resolve_input(args.gt_file) if args.gt_file else None,
            consensus_scope=args.consensus_scope,
            traces_file=_cfg.resolve_input(args.traces_file) if args.traces_file else None,
            domain=args.domain,
            dataset=args.dataset_name,
            use_alignment=args.align,
            model=args.model,
            concurrency=args.concurrency,
        )
        return

    asyncio.run(build_rkgs_for_dataset(
        input_file=_cfg.resolve_input(args.input),
        output_file=_cfg.resolve_output(args.output),
        model=args.model,
        concurrency=args.concurrency,
        domain=args.domain,
        consensus_threshold=args.consensus_threshold,
        node_threshold=args.node_threshold,
        max_samples=args.max_samples,
        edge_lambda=args.edge_lambda,
        weight_by=args.weight_by,
        expected_depth=args.expected_depth,
        consensus_scope=args.consensus_scope,
        dataset_name=args.dataset_name,
        use_alignment=args.align,
    ))


if __name__ == "__main__":
    main()
