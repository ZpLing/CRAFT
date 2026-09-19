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

from framework.module1_generation_filtering.anomaly_filter import parse_steps_from_trace, STEP_PATTERN

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

Output ONLY valid JSON with this exact structure:
{{
  "dependencies": [
    {{"step_id": "Step1", "uses": []}},
    {{"step_id": "Step2", "uses": ["Fact1", "Step1"]}},
    {{"step_id": "Step3", "uses": ["Step2"]}}
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
# the number a run was expected to need. Stages write it into their metadata, and
# detailed_analysis/compute_cost reads it back.
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
        f"  {s['id']} (Step {s['step_number']}): {s['text'][:200]}"
        for s in steps
    )

    prompt = _DEP_PROMPT_TEMPLATE.format(
        facts_block=facts_block,
        steps_block=steps_block,
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
        for src_id in dep.get("uses", []):
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
                edges.append({"src": src_id, "dst": dst_id, "type": "uses", "confidence": 0.9})

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


def build_consensus_rkg(
    trace_rkgs: List[Dict[str, Any]],
    consensus_threshold: float = 0.3,
    node_threshold: Optional[float] = None,
    term_overlap_weight: float = 0.3,   # lambda, the edge-weight balance
    proved_threshold: Optional[float] = None,
    weight_by: str = "uniform",   # "uniform" | "step_count"
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
    trace_weights: Dict[int, float] = {}
    for rkg_trace in trace_rkgs:
        tidx = rkg_trace.get("trace_idx", 0)
        if weight_by == "step_count":
            n_steps = sum(1 for n in rkg_trace.get("nodes", [])
                          if n.get("type") in ("step", "conclusion"))
            # Use a small floor to avoid zero-weight on degenerate traces
            trace_weights[tidx] = float(max(n_steps, 1))
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
            overlap      = _term_overlap_score(
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

    consensus_edges    = []
    consensus_node_ids: Set[str] = set()

    for key, freq in edge_frequencies.items():
        if freq >= consensus_threshold:
            src, dst   = key.split("->", 1)
            avg_conf   = edge_confidence_sum[key] / edge_weight_sum[key] if edge_weight_sum[key] else 0.7
            consensus_edges.append({
                "src":       src,
                "dst":       dst,
                "type":      "uses",
                "confidence": round(avg_conf, 3),
                "frequency":  round(freq, 3),
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
            trace_rkgs.append(result)

    valid_rkgs = [d for d in trace_rkgs if d.get("extraction_method") != "error"]

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
        valid_rkgs,
        consensus_threshold=consensus_threshold,
        node_threshold=node_threshold,
        term_overlap_weight=edge_lambda,
    )

    return {
        "sample_id":     sample.get("sample_id", "unknown"),
        "trace_rkgs":    trace_rkgs,
        "consensus_rkg": consensus,
        "alignment":     alignment_info,
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
    proved_threshold: Optional[float] = None,
    weight_by: str = "uniform",
    gt_file: Optional[Path] = None,
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

    for r in results:
        valid = [copy.deepcopy(t) for t in (r.get("trace_rkgs") or r.get("trace_dags") or [])
                 if t.get("extraction_method") != "error"]
        if not valid:
            r["consensus_rkg"] = {"nodes": [], "edges": []}
            continue
        consensus = build_consensus_rkg(
            valid,
            consensus_threshold=consensus_threshold,
            proved_threshold=proved_threshold,
            weight_by=weight_by,
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
    parser.add_argument("--consensus_threshold", type=float, default=0.3,
                        help="Edge frequency threshold theta (default 0.3)")
    parser.add_argument("--node_threshold", type=float, default=None,
                        help="Node frequency threshold beta (default: same as --consensus_threshold)")
    parser.add_argument("--max_samples", type=int, default=None)
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
    parser.add_argument("--weight_by", choices=["uniform", "step_count"], default="uniform",
                        help="--rebuild_consensus: trace weighting; 'step_count' favors longer traces")
    parser.add_argument("--gt_file", default=None,
                        help="--rebuild_consensus: cleaned_with_problem.json, for conclusion-label diagnostics")
    args = parser.parse_args()

    if args.rebuild_consensus:
        rebuild_consensus(
            input_file=_cfg.resolve_input(args.input),
            consensus_threshold=args.consensus_threshold,
            proved_threshold=args.proved_threshold,
            weight_by=args.weight_by,
            gt_file=_cfg.resolve_input(args.gt_file) if args.gt_file else None,
        )
        return

    global OPENAI_API_KEY, OPENAI_BASE_URL, CHAT_URL, HEADERS
    if args.api_key:
        OPENAI_API_KEY = args.api_key
        HEADERS["Authorization"] = f"Bearer {OPENAI_API_KEY}"
    if args.base_url:
        OPENAI_BASE_URL = args.base_url
        CHAT_URL = OPENAI_BASE_URL.rstrip("/") + "/chat/completions"

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
    ))


if __name__ == "__main__":
    main()
