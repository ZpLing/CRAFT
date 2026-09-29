#!/usr/bin/env python3
"""
build_rkg.py  (Module II — Consensus RKG Construction)
------------------------------------------------------------------------------
Build a Reasoning Knowledge Graph (RKG) G_S for each reasoning trace, weight its
edges by W_S(e) = (1-lambda)*conf(e) + lambda*Jaccard(u,v), take the union G*,
and keep the edges whose consensus weight W(e) = 1/K * sum_S W_S(e) reaches theta
(Algorithm 1, Module II).

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
  5. Build the consensus RKG G*: W(e) edge filtering, isolated-node filtering

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
      "edge_weights": {"Fact1->Step2": 0.62, ...},      W(e)
      "edge_frequencies": {"Fact1->Step2": 0.8, ...}    fraction of the K traces
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
# Table 1 hyperparameters, set once in config.py
THETA, LAMBDA = _cfg.THETA, _cfg.LAMBDA


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
_FACT_MARK = re.compile(r'\bfact\s*(\d+)\s*:', re.IGNORECASE)
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
        # The datasets write every fact on one line ("Fact1: ... Fact2: ..."), so
        # the facts are cut at each inline "FactN:" marker rather than read one
        # per line, which returned only Fact1 with the whole problem as its text.
        marks = list(_FACT_MARK.finditer(problem_text))
        for i, m in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(problem_text)
            text = problem_text[m.end():end].strip()
            # The last fact runs into whatever follows the facts (the hypothesis).
            text = re.split(r"\b(?:Hypothesis|Question|Conclusion)\s*:", text, maxsplit=1)[0].strip()
            if text:
                facts.append({
                    "id": f"Fact{int(m.group(1))}",
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
- For every dependency, give your confidence in [0, 1] that the step really uses it:
  1.0 when the step states or plainly applies that premise, lower when the link is
  only implied, and do not list a dependency you would put below 0.3.

Output ONLY valid JSON with this exact structure:
{{
  "dependencies": [
    {{"step_id": "Step1", "uses": []}},
    {{"step_id": "Step2", "uses": [{{"id": "Fact1", "confidence": 1.0}}, {{"id": "Step1", "confidence": 0.8}}]}},
    {{"step_id": "Step3", "uses": [{{"id": "Step2", "confidence": 0.9}}]}}
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
        for use in dep.get("uses", []):
            # conf(e) of Section 3.2 is the extractor's own confidence. A bare id
            # (the format before confidences were asked for) keeps the old 0.9.
            if isinstance(use, dict):
                src_id = use.get("id") or use.get("step_id") or ""
                try:
                    conf = min(1.0, max(0.0, float(use.get("confidence", 0.9))))
                except (TypeError, ValueError):
                    conf = 0.9
            else:
                src_id, conf = use, 0.9
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


_ALIGN_TOKEN = re.compile(r"[a-z]{3,}|\d+(?:\.\d+)?")
_ALIGN_STOP = {"the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
               "have", "has", "had", "then", "therefore", "thus", "hence", "since",
               "because", "step", "fact", "facts", "given", "which", "into", "its"}

# Two steps of different traces are the same node when their step similarity
# reaches this value. Not a paper hyperparameter: it only decides which ids
# name the same step, which the paper takes as given.
ALIGN_THRESHOLD = 0.4


_STEP_REF = re.compile(r"\bstep\s*\d+\s*:?", re.IGNORECASE)
_FACT_REF = re.compile(r"\bfact\s*(\d+)", re.IGNORECASE)


def _align_tokens(text: str) -> Set[str]:
    """Content tokens of a step, for matching it to other traces' steps.

    Step numbers are positions within one trace ("Step 3", "from Step 2") and
    say nothing about which inference a step is, so they are dropped. A cited
    premise is kept whole ("fact8") because it is the most specific thing a
    step says about itself.
    """
    text = _STEP_REF.sub(" ", text or "").lower()
    facts = {f"fact{m}" for m in _FACT_REF.findall(text)}
    text = _FACT_REF.sub(" ", text)
    return facts | {t for t in _ALIGN_TOKEN.findall(text) if t not in _ALIGN_STOP}


def _cited_facts(text: str) -> Set[str]:
    return {f"Fact{m}" for m in _FACT_REF.findall(text or "")}


def align_step_nodes(trace_rkgs: List[Dict[str, Any]]) -> Dict[int, Dict[str, str]]:
    """Give the same step the same node id in every trace, in place.

    Section 3.2 aggregates the per-trace graphs by set union on the premise that
    "nodes and edges produced by LLMs are identical when they correspond to
    identical steps". Extraction names steps by position, so Step3 of one trace
    and Step3 of another are in general different inferences, and one inference
    drawn by all K traces arrives under K different ids. This makes the premise
    hold before the union.

    Fact nodes are the problem's own premises and already share their ids, and
    the conclusion has been merged into one node before this runs. Every other
    step is matched greedily, trace by trace, to the closest step group no
    earlier step of the same trace has joined; the similarity is the Jaccard of
    the two texts, averaged with the Jaccard of the facts each step uses when
    both use some. Below ALIGN_THRESHOLD a step opens a group of its own.
    Groups are named Step1, Step2, ... by their mean relative position, so the
    ids still read in reasoning order.

    Returns {trace_idx: {original id: aligned id}} for every renamed node.
    """
    groups: List[Dict[str, Any]] = []   # {"members": [(tokens, fact_parents)], "pos": [..]}
    assign: Dict[Tuple[int, str], int] = {}

    for rkg in trace_rkgs:
        tidx = rkg.get("trace_idx", 0)
        steps = [n for n in rkg.get("nodes", []) if n.get("type") == "step"]
        steps.sort(key=lambda n: (n.get("step_number") or 0))
        fact_parents: Dict[str, Set[str]] = defaultdict(set)
        for e in rkg.get("edges", []):
            if e["src"].startswith("Fact"):
                fact_parents[e["dst"]].add(e["src"])
        used: Set[int] = set()
        n_steps = max(len(steps), 1)
        for i, node in enumerate(steps):
            toks = _align_tokens(node.get("text", ""))
            facts = fact_parents.get(node["id"], set()) | _cited_facts(node.get("text", ""))
            best, best_sim = None, 0.0
            for gi, g in enumerate(groups):
                if gi in used:
                    continue
                sim = 0.0
                for mtoks, mfacts in g["members"]:
                    u = toks | mtoks
                    st = len(toks & mtoks) / len(u) if u else 0.0
                    if facts and mfacts:
                        st = 0.5 * st + 0.5 * len(facts & mfacts) / len(facts | mfacts)
                    sim = max(sim, st)
                if sim > best_sim:
                    best, best_sim = gi, sim
            if best is None or best_sim < ALIGN_THRESHOLD:
                groups.append({"members": [], "pos": []})
                best = len(groups) - 1
            groups[best]["members"].append((toks, facts))
            groups[best]["pos"].append(i / n_steps)
            used.add(best)
            assign[(tidx, node["id"])] = best

    order = sorted(range(len(groups)), key=lambda gi: sum(groups[gi]["pos"]) / len(groups[gi]["pos"]))
    name = {gi: f"Step{rank + 1}" for rank, gi in enumerate(order)}

    mapping: Dict[int, Dict[str, str]] = defaultdict(dict)
    for rkg in trace_rkgs:
        tidx = rkg.get("trace_idx", 0)
        ren = {nid: name[gi] for (t, nid), gi in assign.items() if t == tidx}
        for n in rkg.get("nodes", []):
            if n["id"] in ren:
                n.setdefault("orig_id", n["id"])
                mapping[tidx][n["orig_id"]] = ren[n["id"]]
                n["id"] = ren[n["id"]]
        new_edges, seen = [], set()
        for e in rkg.get("edges", []):
            src, dst = ren.get(e["src"], e["src"]), ren.get(e["dst"], e["dst"])
            if src == dst or (src, dst) in seen:
                continue
            seen.add((src, dst))
            ee = dict(e); ee["src"] = src; ee["dst"] = dst
            new_edges.append(ee)
        rkg["edges"] = new_edges
    return dict(mapping)


def restore_original_ids(rkg_trace: Dict[str, Any]) -> None:
    """Undo a previous consensus pass's renaming of one trace graph, in place.

    build_consensus_rkg rewrites a trace graph's ids (the conclusion to
    ConcShared, aligned steps to their group names) and a saved run stores the
    graphs that way. A later rebuild from those graphs would align the aligned
    ids again and could merge two nodes; putting every node back under the id
    extraction gave it first makes a rebuild start from the same graph a fresh
    run does.
    """
    back: Dict[str, str] = {}
    nodes = []
    for n in rkg_trace.get("nodes", []):
        orig = n.get("orig_conc_id") or n.get("orig_id")
        if orig and orig != n["id"]:
            back[n["id"]] = orig
            n = {k: v for k, v in n.items() if k not in ("orig_id", "orig_conc_id")}
            n["id"] = orig
        nodes.append(n)
    if not back:
        return
    rkg_trace["nodes"] = nodes
    rkg_trace["edges"] = [
        dict(e, src=back.get(e["src"], e["src"]), dst=back.get(e["dst"], e["dst"]))
        for e in rkg_trace.get("edges", [])
    ]


def build_consensus_rkg(
    trace_rkgs: List[Dict[str, Any]],
    theta: float = THETA,
    lam: float = LAMBDA,
    weight_by: str = "uniform",   # "uniform" | "gold_depth"
    expected_depth: Optional[int] = None,
    node_threshold: Optional[float] = None,
    consensus_threshold: Optional[float] = None,
    term_overlap_weight: Optional[float] = None,
    align_steps: bool = True,
    anchor_conclusion: bool = False,
    support: str = "direct",
) -> Dict[str, Any]:
    """Module II (2): Edges & Nodes Filtering, as Algorithm 1 lines 10-15 write it.

        W_S(e) = (1 - lambda) * conf(e) + lambda * Jaccard(u, v)     per trace S, e in E_S
        G* = (V*, E*) = union of the G_S
        W(e)   = 1/K * sum_{S : e in E_S} W_S(e)                    consensus edge weight
        G*    <- G* minus {e : W(e) < theta} minus isolated nodes

    conf(e) is the extractor's confidence for the edge in trace S, and
    Jaccard(u, v) compares the two steps as trace S wrote them. A trace without
    the edge contributes zero, so W(e) rewards an edge both for how many traces
    draw it and for how strongly each of them does. K is the number of traces
    that produced an RKG. Under weight_by="gold_depth" the sum and K are both
    trace-weighted, which is the uniform case when every weight is 1.

    A node is isolated when no edge that survives theta touches it, and it is
    removed whatever its type. Module III writes the verdict itself when the
    conclusion node does not survive, so nothing downstream needs it kept.

    consensus_threshold / term_overlap_weight are the old names of theta /
    lambda and are still accepted. node_threshold no longer has an effect: the
    paper filters nodes by isolation only.

    Returns:
        {
          "nodes": [...],
          "edges": [...],            # E*, each with weight W(e), support and mean W_S(e)
          "edge_weights": {...},     # W(e) for every edge of the union, kept or not
          "edge_frequencies": {...}, # fraction of the K traces that contain each edge
          "node_frequencies": {...},
          "node_texts": {...}
        }
    """
    if consensus_threshold is not None:
        theta = consensus_threshold
    if term_overlap_weight is not None:
        lam = term_overlap_weight
    if not trace_rkgs:
        return {"nodes": [], "edges": [], "edge_frequencies": {}, "node_texts": {}}

    # ── Trace weights ──────────────────────────────────────────────────────
    # "uniform"   : every trace counts as 1.0 (default — plain MV)
    # "gold_depth": weight by how near a trace lands to the depth the dataset is
    #               drawn at, and only where that depth is a property of the
    #               selection rather than of the sample. On the depth-5
    #               ProofWriter slice a trace within two steps of it is right 90%
    #               of the time against 50% for one six steps over, and an equal
    #               vote spends the two the same.
    trace_weights: Dict[int, float] = {}
    for rkg_trace in trace_rkgs:
        tidx = rkg_trace.get("trace_idx", 0)
        n_steps = sum(1 for n in rkg_trace.get("nodes", [])
                      if n.get("type") in ("step", "conclusion"))
        # gold_depth compares the trace's own length, not the graph's
        n_trace_steps = rkg_trace.get("n_trace_steps") or n_steps
        if weight_by == "gold_depth" and expected_depth:
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
        restore_original_ids(rkg_trace)
    for rkg_trace in trace_rkgs:
        # Find all conclusion-like nodes in this trace
        concl_node_ids = {
            n["id"] for n in rkg_trace.get("nodes", [])
            if _LBL_PAT.search(n.get("text", ""))
        }
        if not concl_node_ids:
            # A maths trace carries no verdict marker; its last step is the node
            # extraction typed "conclusion". Left under its own StepN id it could
            # share a name with an aligned step group, which merged one trace's
            # answer with another trace's intermediate step.
            concl_node_ids = {
                n["id"] for n in rkg_trace.get("nodes", [])
                if n.get("type") == "conclusion"
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
                nn["orig_conc_id"] = primary_id
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

    # ── Identical steps, identical nodes (Section 3.2) ───────────────────
    node_alignment = align_step_nodes(trace_rkgs) if align_steps else {}
    for rkg_trace in trace_rkgs:
        for n in rkg_trace.get("nodes", []):
            if n["id"] == _CONC_ID and n.get("orig_conc_id"):
                node_alignment.setdefault(rkg_trace.get("trace_idx", 0), {})[n["orig_conc_id"]] = _CONC_ID

    # ── W_S(e) per trace, from that trace's own step texts ────────────────
    node_trace_count: Dict[str, int] = defaultdict(int)
    for rkg_trace in trace_rkgs:
        for nid in {n["id"] for n in rkg_trace.get("nodes", [])}:
            node_trace_count[nid] += 1      # how many traces contain this node at all

    edge_weight_sum:  Dict[str, float] = defaultdict(float)   # sum of trace weights (support)
    edge_ws_sum:      Dict[str, float] = defaultdict(float)   # sum of w_S * W_S(e)
    edge_count:       Dict[str, int]   = defaultdict(int)

    per_trace_ws: List[Tuple[float, Dict[Tuple[str, str], float]]] = []
    for rkg_trace in trace_rkgs:
        tidx   = rkg_trace.get("trace_idx", 0)
        weight = trace_weights.get(tidx, 1.0)
        texts  = {n["id"]: n.get("text", "") for n in rkg_trace.get("nodes", [])}
        ws_direct: Dict[Tuple[str, str], float] = {}
        for e in rkg_trace.get("edges", []):
            src, dst = e["src"], e["dst"]
            if (src, dst) in ws_direct:
                continue
            conf    = float(e.get("confidence", 0.7))
            overlap = _similarity_score(texts.get(src, ""), texts.get(dst, ""))
            w_s     = (1 - lam) * conf + lam * overlap
            e["weight"] = round(w_s, 4)      # W_S(e), kept on the trace's own graph
            ws_direct[(src, dst)] = w_s
        per_trace_ws.append((weight, ws_direct))

    # E* is the union of the edges the traces actually draw.
    candidates = {k for _, wsd in per_trace_ws for k in wsd}

    # Path support (adaptation). Traces disagree less on what depends on what
    # than on which dependency they state directly: one writes u -> v, another
    # u -> w -> v. With support="path" a trace that reaches v from u without the
    # direct edge still backs u -> v, with the W_S of the weakest edge on its
    # strongest path, so W(e) counts agreement on the dependency. With
    # support="direct" only the edge itself counts, the literal reading.
    for weight, wsd in per_trace_ws:
        if support == "path":
            nodes_ = {x for k in wsd for x in k}
            best: Dict[Tuple[str, str], float] = dict(wsd)
            for mid in nodes_:
                for (a, b_), w1 in list(best.items()):
                    if b_ != mid:
                        continue
                    for (c, d), w2 in list(best.items()):
                        if c != mid or d == a:
                            continue
                        w = min(w1, w2)
                        if w > best.get((a, d), 0.0):
                            best[(a, d)] = w
        else:
            best = wsd
        for (src, dst) in candidates:
            w_s = best.get((src, dst), 0.0)
            if w_s <= 0.0:
                continue
            key = f"{src}->{dst}"
            edge_weight_sum[key] += weight
            edge_ws_sum[key]     += w_s * weight
            edge_count[key]      += 1

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
        best = max(label_w, key=lambda l: label_w[l])
        node_texts[nid] = f"Final Conclusion: {best}"

    # ── W(e) and the edge filter ──────────────────────────────────────────
    total = total_weight if total_weight > 0 else 1.0
    edge_weights = {key: round(ws / total, 4) for key, ws in edge_ws_sum.items()}
    edge_frequencies = {key: round(wsum / total, 4) for key, wsum in edge_weight_sum.items()}

    consensus_edges: List[Dict[str, Any]] = []
    for key, w in edge_weights.items():
        if w < theta:
            continue
        src, dst = key.split("->", 1)
        consensus_edges.append({
            "src":        src,
            "dst":        dst,
            "type":       "uses",
            "weight":     w,                                    # W(e)
            "frequency":  edge_frequencies[key],                # support over K
            # mean W_S(e) over the traces that draw the edge; the field downstream
            # code already reads as the edge's confidence
            "confidence": round(edge_ws_sum[key] / edge_weight_sum[key], 4)
                          if edge_weight_sum[key] else 0.0,
        })

    # ── Conclusion anchoring (adaptation) ─────────────────────────────────
    # Every trace ends in the verdict, but each reaches it from a different last
    # step, so no single edge into the conclusion need carry theta and the node
    # would be filtered as isolated. When none survives, the conclusion keeps its
    # one highest-W(e) incoming edge whose source is still in the graph. Every
    # other edge is filtered by theta exactly as above.
    conclusion_ids = {nid for nid, t in node_types.items() if t == "conclusion"}
    if anchor_conclusion and conclusion_ids and consensus_edges:
        kept_ids = {e["src"] for e in consensus_edges} | {e["dst"] for e in consensus_edges}
        for cid in conclusion_ids:
            if any(e["dst"] == cid for e in consensus_edges):
                continue
            into = [(w, k) for k, w in edge_weights.items()
                    if k.split("->", 1)[1] == cid and k.split("->", 1)[0] in kept_ids]
            if into:
                w, key = max(into)
                src, dst = key.split("->", 1)
                consensus_edges.append({
                    "src": src, "dst": dst, "type": "uses", "weight": w,
                    "frequency": edge_frequencies[key],
                    "confidence": round(edge_ws_sum[key] / edge_weight_sum[key], 4)
                                  if edge_weight_sum[key] else 0.0,
                    "anchor": True,
                })

    # ── Node filter: remove isolated nodes, d+(v) = 0 and d-(v) = 0 ───────
    consensus_node_ids: Set[str] = set()
    for e in consensus_edges:
        consensus_node_ids.add(e["src"])
        consensus_node_ids.add(e["dst"])

    n_traces = max(len(trace_rkgs), 1)
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
        "nodes":            consensus_nodes,
        "edges":            consensus_edges,
        "edge_weights":     edge_weights,
        "edge_frequencies": edge_frequencies,
        "node_frequencies": {nid: round(c / n_traces, 4) for nid, c in node_trace_count.items()},
        "node_texts":       node_texts,
        "trace_weights":    trace_weights,
        "theta":            theta,
        "lambda":           lam,
        "node_alignment":   {str(k): v for k, v in node_alignment.items()},
    }


#########################
# Sample-Level Batch Processing
#########################

async def build_rkgs_for_sample(
    session: aiohttp.ClientSession,
    sample: Dict[str, Any],
    model: str = DEFAULT_MODEL,
    domain: str = "logical",
    theta: float = THETA,
    lam: float = LAMBDA,
    weight_by: str = "uniform",
    expected_depth: Optional[int] = None,
) -> Dict[str, Any]:
    """Concurrently build per-trace RKGs for a sample, then compute the consensus RKG."""
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

    consensus = build_consensus_rkg(
        valid_rkgs,
        theta=theta,
        lam=lam,
        weight_by=weight_by,
        expected_depth=expected_depth,
    )

    return {
        "sample_id":       sample.get("sample_id", "unknown"),
        "trace_rkgs":      trace_rkgs,
        "consensus_rkg":   consensus,
    }


async def build_rkgs_for_dataset(
    input_file: Path,
    output_file: Path,
    model: str = DEFAULT_MODEL,
    concurrency: int = 10,
    domain: str = "logical",
    theta: float = THETA,
    max_samples: Optional[int] = None,
    lam: float = LAMBDA,
    weight_by: str = "uniform",
    expected_depth: Optional[int] = None,
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
                        theta=theta,
                        lam=lam,
                        weight_by=weight_by,
                        expected_depth=expected_depth,
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
            "theta": theta,
            "lambda": lam,
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
    theta: float = THETA,
    lam: float = LAMBDA,
    weight_by: str = "uniform",
    gt_file: Optional[Path] = None,
    expected_depth: Optional[int] = None,
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
        # lambda and the node threshold travel with the rest. They used not to:
        # a rebuild always used build_consensus_rkg's defaults for both, so
        # --edge_lambda 0 rebuilt a graph byte-identical to --edge_lambda 0.3 —
        # node sets, edge sets and edge confidences all unchanged on 100 of 100
        # graphs, so a lambda sweep compared the full model with itself.
        consensus = build_consensus_rkg(
            valid,
            theta=theta,
            lam=lam,
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
    parser = argparse.ArgumentParser(description="Module II Consensus RKG Construction: per-trace RKGs, edge and node filtering, aggregation into the consensus RKG G*")
    parser.add_argument("--input", required=True, help="Path to cleaned_traces.json")
    parser.add_argument("--output", default="rkg.json",
                        help="Output RKG JSON path (relative paths resolve under the results root)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"LLM model (default: {DEFAULT_MODEL})")
    parser.add_argument("--api_key", default=None, help="API Key")
    parser.add_argument("--base_url", default=None, help="API Base URL")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrency level (default 10)")
    parser.add_argument("--domain", default="logical", choices=["logical", "math"])
    parser.add_argument("--theta", "--consensus_threshold", dest="theta", type=float, default=THETA,
                        help="Edge filtering threshold theta: an edge stays in G* when its consensus "
                             "weight W(e) = 1/K * sum_S W_S(e) is at least theta (paper: 0.3)")
    parser.add_argument("--node_threshold", type=float, default=None,
                        help=argparse.SUPPRESS)  # no effect: nodes are filtered by isolation (paper)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--similarity", choices=["jaccard", "embedding"],
                        default="jaccard",
                        help="The overlap measure fused into W(e): term Jaccard "
                             "(the paper) or the cosine of all-mpnet-base-v2 "
                             "embeddings (the ablation's row)")
    parser.add_argument("--lambda", "--edge_lambda", dest="lam", type=float, default=LAMBDA,
                        help="Weight of Jaccard in W_S(e), lambda: W_S(e) = (1-lambda)*conf(e) "
                             "+ lambda*Jaccard(u,v) (paper: 0.3)")

    # Offline mode: re-vote an existing RKG file's cached trace_rkgs, no LLM calls.
    parser.add_argument("--rebuild_consensus", action="store_true",
                        help="Recompute consensus_rkg in --input in place from its cached "
                             "trace_rkgs (no LLM calls); --output is ignored")
    parser.add_argument("--expected_depth", type=int, default=None,
                        help="The proof depth this dataset is drawn at, for --weight_by "
                             "gold_depth. Only meaningful where the depth is a property of "
                             "the selection and not gold annotation about the sample")
    parser.add_argument("--weight_by", choices=["uniform", "gold_depth"],
                        default="uniform",
                        help="--rebuild_consensus: trace weighting; 'gold_depth' favors traces "
                             "whose length is near --expected_depth")
    parser.add_argument("--gt_file", default=None,
                        help="--rebuild_consensus: cleaned_with_problem.json, for conclusion-label diagnostics")
    args = parser.parse_args()
    set_similarity(args.similarity)

    global OPENAI_API_KEY, OPENAI_BASE_URL, CHAT_URL, HEADERS
    if args.api_key:
        OPENAI_API_KEY = args.api_key
        HEADERS["Authorization"] = f"Bearer {OPENAI_API_KEY}"
    if args.base_url:
        OPENAI_BASE_URL = args.base_url
        CHAT_URL = OPENAI_BASE_URL.rstrip("/") + "/chat/completions"

    if args.rebuild_consensus:
        rebuild_consensus(
            input_file=_cfg.resolve_input(args.input),
            theta=args.theta,
            lam=args.lam,
            weight_by=args.weight_by,
            expected_depth=args.expected_depth,
            gt_file=_cfg.resolve_input(args.gt_file) if args.gt_file else None,
        )
        return

    asyncio.run(build_rkgs_for_dataset(
        input_file=_cfg.resolve_input(args.input),
        output_file=_cfg.resolve_output(args.output),
        model=args.model,
        concurrency=args.concurrency,
        domain=args.domain,
        theta=args.theta,
        max_samples=args.max_samples,
        lam=args.lam,
        weight_by=args.weight_by,
        expected_depth=args.expected_depth,
    ))


if __name__ == "__main__":
    main()
