#!/usr/bin/env python3
"""The five main-table baselines that baseline_comparisons.py does not cover.

RAP, ConCISE, LCoT2Tree, DICE and PNS-Optimization, each following the setting
stated for it in the paper's appendix. These are reimplementations from those
descriptions, not the authors' original code — the numbers they produce are
comparable to the other settings here, not to the numbers in the original papers.

Shares dataset loading, answer extraction and metrics with baseline_comparisons.py,
so predictions and summaries have the same shape as settings A and F-K.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_common import (  # noqa: E402
    SYSTEM_LOGICIAN, SYSTEM_MATH, build_zeroshot_prompt, call_llm, compute_metrics,
    compute_per_dataset, count_steps, count_tokens, load_raw_dataset, print_metrics,
    print_comparison_table, _extract_pred, DEFAULT_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL,
    attach, save_run, by_domain,
)

logger = logging.getLogger("baseline_extended")
ALL_SETTINGS = ["TOT", "RAP", "CONCISE", "LCOT2TREE", "DICE", "PNS"]


def _sys_for(domain: str) -> str:
    return SYSTEM_MATH if domain == "math" else SYSTEM_LOGICIAN


def _answer_format(domain: str) -> str:
    return ("the final line \\boxed{<numeric answer>}" if domain == "math"
            else "the final line __PROVED__ or __DISPROVED__")


def _pack(sample: Dict, pred, texts: List[str], domain: str,
          answer_text: Optional[str] = None,
          artifacts: Optional[Dict] = None, **extra) -> Dict:
    """Build one prediction record, keeping the generations behind it.

    `texts` is every generation this baseline made for the sample, in order.
    It used to be reduced to two averages and dropped; it now travels with the
    record and `save_run` writes it to the traces file. `artifacts` carries the
    search state a particular baseline wants kept — the tree it explored, the
    sub-problems it split into — which is what makes a wrong answer readable
    afterwards.

    `answer_text` is the generation the reported answer was read from, and it is
    what the step and token counts measure. Averaging those counts over `texts`
    instead — which is what happened before — measures the wrong thing for a
    search baseline: Tree-of-Thought's texts are mostly one-line THOUGHT
    expansions and single-number evaluations, so the mean reports the length of
    its scratch work rather than the length of its answer, and a comparison of
    "average steps" across baselines then compares different things.
    """
    texts = [t for t in texts if t]
    measured = [answer_text] if answer_text else texts
    measured = [t for t in measured if t]
    return attach({
        "sample_id":       sample["sample_id"],
        "source_dataset":  sample["source_dataset"],
        "ground_truth":    sample["ground_truth"],
        "predicted":       pred,
        "response_steps":  float(np.mean([count_steps(t) for t in measured])) if measured else 0.0,
        "response_tokens": float(np.mean([count_tokens(t) for t in measured])) if measured else 0.0,
        "domain":          domain,
        **extra,
    }, traces=texts, **(artifacts or {}))


# ===========================================================================
# Tree-of-Thought (Yao et al., 2023)
# Real breadth-first tree search: at each depth expand every surviving node into
# `width` candidate thoughts, self-evaluate each, keep the best `width`. The
# existing baseline_tot_legacy.py only simulates this inside a single prompt.
# ===========================================================================

_TOT_EXPAND = (
    "Problem:\n{problem}\n\nThoughts so far:\n{state}\n\n"
    "Generate {w} DIFFERENT candidate next thoughts, one per line, each prefixed 'THOUGHT:'. "
    "Each is one reasoning move, not a full solution."
)
_TOT_EVAL = (
    "Problem:\n{problem}\n\nCandidate reasoning:\n{state}\n\n"
    "Rate how likely this leads to the correct answer. Reply with ONLY a number 0.0-1.0."
)
_TOT_FINISH = (
    "Problem:\n{problem}\n\nSelected reasoning path:\n{state}\n\n"
    "Finish the solution from this path. End with {fmt}."
)


async def run_tot(samples, model, api_key, base_url, semaphore, session, width=5, depth=2):
    domain = samples[0].get("domain", "logical") if samples else "logical"
    system, fmt = _sys_for(domain), _answer_format(domain)

    async def one(s):
        problem, texts = s["problem_text"], []
        frontier: List[List[str]] = [[]]
        for _ in range(depth):
            cands: List[List[str]] = []
            raws = await asyncio.gather(*[
                call_llm(session, semaphore, system,
                         _TOT_EXPAND.format(problem=problem, w=width, state="\n".join(st) or "(none)"),
                         model, api_key, base_url, temperature=0.8, max_tokens=512)
                for st in frontier])
            for st, raw in zip(frontier, raws):
                texts.append(raw or "")
                for line in (raw or "").splitlines():
                    line = line.strip()
                    if line.upper().startswith("THOUGHT:"):
                        cands.append(st + [line[8:].strip()])
            if not cands:
                break
            scores = await asyncio.gather(*[
                call_llm(session, semaphore, system,
                         _TOT_EVAL.format(problem=problem, state="\n".join(c)),
                         model, api_key, base_url, temperature=0.0, max_tokens=16)
                for c in cands])
            ranked = sorted(zip(cands, [_parse_score01(x) for x in scores]),
                            key=lambda t: t[1], reverse=True)
            frontier = [c for c, _ in ranked[:width]]
        best = frontier[0] if frontier else []
        final = await call_llm(session, semaphore, system,
                               _TOT_FINISH.format(problem=problem, fmt=fmt,
                                                  state="\n".join(best) or "(none)"),
                               model, api_key, base_url, temperature=0.0, max_tokens=1024)
        texts.append(final or "")
        return _pack(s, _extract_pred(final or "", domain), texts, domain,
                     answer_text=final or "", tot_depth=len(best),
                     artifacts={"selected_path": best, "final": final or ""})

    return list(await asyncio.gather(*[one(s) for s in samples]))


# ===========================================================================
# RAP — Reasoning via Planning (Hao et al., 2023)
# MCTS over partial reasoning states; the LLM is both world model and reward.
# ===========================================================================

_RAP_EXPAND = (
    "You are planning a solution one step at a time.\n"
    "Problem:\n{problem}\n\n"
    "Steps taken so far:\n{state}\n\n"
    "Propose {b} DIFFERENT possible next steps, one per line, each prefixed 'STEP:'. "
    "Each must be a single concrete reasoning step, not a full solution."
)
_RAP_REWARD = (
    "Rate how promising this partial reasoning is for reaching the correct answer.\n"
    "Problem:\n{problem}\n\nPartial reasoning:\n{state}\n\n"
    "Reply with ONLY a number between 0.0 and 1.0."
)
_RAP_ROLLOUT = (
    "Problem:\n{problem}\n\nReasoning so far:\n{state}\n\n"
    "Continue and finish the solution. End with {fmt}."
)


def _parse_score01(text: Optional[str]) -> float:
    if not text:
        return 0.0
    m = re.search(r"([01](?:\.\d+)?|\.\d+)", text)
    if not m:
        return 0.0
    try:
        return max(0.0, min(1.0, float(m.group(1))))
    except ValueError:
        return 0.0


class _Node:
    __slots__ = ("steps", "parent", "children", "visits", "value")

    def __init__(self, steps: List[str], parent: Optional["_Node"] = None):
        self.steps, self.parent = steps, parent
        self.children: List["_Node"] = []
        self.visits, self.value = 0, 0.0

    def ucb(self, c: float) -> float:
        if self.visits == 0:
            return float("inf")
        exploit = self.value / self.visits
        explore = c * math.sqrt(math.log(max(1, self.parent.visits)) / self.visits) if self.parent else 0.0
        return exploit + explore


async def run_rap(samples, model, api_key, base_url, semaphore, session,
                  iterations=8, branch=3, max_depth=4, c_ucb=1.0):
    domain = samples[0].get("domain", "logical") if samples else "logical"
    system, fmt = _sys_for(domain), _answer_format(domain)

    async def one(s):
        problem = s["problem_text"]
        root, texts = _Node([]), []
        for _ in range(iterations):
            node = root
            while node.children and len(node.steps) < max_depth:
                node = max(node.children, key=lambda n: n.ucb(c_ucb))
            if len(node.steps) < max_depth and not node.children:
                raw = await call_llm(session, semaphore, system,
                                     _RAP_EXPAND.format(problem=problem, b=branch,
                                                        state="\n".join(node.steps) or "(none)"),
                                     model, api_key, base_url, temperature=0.8, max_tokens=512)
                texts.append(raw or "")
                for line in (raw or "").splitlines():
                    line = line.strip()
                    if line.upper().startswith("STEP:"):
                        node.children.append(_Node(node.steps + [line[5:].strip()], node))
                if node.children:
                    node = node.children[0]
            rew = await call_llm(session, semaphore, system,
                                 _RAP_REWARD.format(problem=problem, state="\n".join(node.steps) or "(none)"),
                                 model, api_key, base_url, temperature=0.0, max_tokens=16)
            r, cur = _parse_score01(rew), node
            while cur is not None:
                cur.visits += 1
                cur.value += r
                cur = cur.parent
        best, node = root, root
        while node.children:
            node = max(node.children, key=lambda n: (n.value / n.visits) if n.visits else -1.0)
            best = node
        final = await call_llm(session, semaphore, system,
                               _RAP_ROLLOUT.format(problem=problem, fmt=fmt,
                                                   state="\n".join(best.steps) or "(none)"),
                               model, api_key, base_url, temperature=0.0, max_tokens=1024)
        texts.append(final or "")
        return _pack(s, _extract_pred(final or "", domain), texts, domain,
                     answer_text=final or "", rap_depth=len(best.steps),
                     artifacts={"selected_path": list(best.steps), "final": final or ""})

    return list(await asyncio.gather(*[one(s) for s in samples]))


# ===========================================================================
# ConCISE (Qiao et al., 2025)
# Confidence-guided cues that suppress redundant reflection; one pass per sample.
# ===========================================================================

_CONCISE_SYS = (
    "{base}\n\n"
    "Reason CONCISELY under confidence injection:\n"
    "- State each intermediate conclusion together with your confidence in it.\n"
    "- Once an intermediate conclusion reaches high confidence, COMMIT to it and move on.\n"
    "- Do NOT re-derive, re-check or backtrack over a committed conclusion.\n"
    "- Omit exploratory dead ends entirely; write only the steps that carry the solution.\n"
    "End with {fmt}."
)


async def run_concise(samples, model, api_key, base_url, semaphore, session):
    domain = samples[0].get("domain", "logical") if samples else "logical"
    system = _CONCISE_SYS.format(base=_sys_for(domain), fmt=_answer_format(domain))

    async def one(s):
        out = await call_llm(session, semaphore, system,
                             build_zeroshot_prompt(s["problem_text"], domain=domain),
                             model, api_key, base_url, temperature=0.0, max_tokens=1024)
        return _pack(s, _extract_pred(out or "", domain), [out], domain,
                     answer_text=out or "")

    return list(await asyncio.gather(*[one(s) for s in samples]))


# ===========================================================================
# LCoT2Tree (Jiang et al., 2025)
# K linear traces -> merged tree (equivalent steps collapse) -> structural scoring.
# Merging and scoring are pure computation, so the only LLM cost is the K rollouts.
# ===========================================================================

_STOP = set("the a an of to in is are and or that this it be we have has was were with as for on "
            "so then thus therefore hence since if not no do does by from at".split())


def _bag(step: str) -> frozenset:
    return frozenset(w for w in re.findall(r"[a-z0-9]+", step.lower()) if w not in _STOP and len(w) > 2)


def _equivalent(a: frozenset, b: frozenset, thr: float = 0.6) -> bool:
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= thr


def _split_steps(text: str) -> List[str]:
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    steps = [l for l in lines if re.match(r"^(step\s*\d+|[-*\d]+[.)])", l, re.I)]
    return (steps or lines)[:12]


async def run_lcot2tree(samples, model, api_key, base_url, semaphore, session, k=10):
    domain = samples[0].get("domain", "logical") if samples else "logical"
    system = _sys_for(domain)

    async def one(s):
        user = build_zeroshot_prompt(s["problem_text"], domain=domain)
        traces = await asyncio.gather(*[
            call_llm(session, semaphore, system, user, model, api_key, base_url,
                     temperature=0.7, max_tokens=1024) for _ in range(k)])
        traces = [t for t in traces if t]
        if not traces:
            return _pack(s, None, [], domain)

        # merge: a node is a (depth, bag-of-content) class shared across traces
        nodes: Dict[int, List[Dict]] = defaultdict(list)
        paths = []
        for t in traces:
            path, ans = [], _extract_pred(t, domain)
            for d, st in enumerate(_split_steps(t)):
                bag = _bag(st)
                hit = next((n for n in nodes[d] if _equivalent(n["bag"], bag)), None)
                if hit is None:
                    hit = {"bag": bag, "count": 0, "children": set()}
                    nodes[d].append(hit)
                hit["count"] += 1
                if path:
                    path[-1]["children"].add(id(hit))
                path.append(hit)
            paths.append((path, ans))

        n_tr = len(traces)
        best_ans, best_score = None, -1.0
        for path, ans in paths:
            if ans is None or not path:
                continue
            consensus = sum(n["count"] for n in path) / (len(path) * n_tr)   # shared-node mass
            depth     = len(path) / 12.0                                      # longer = more derived
            branching = np.mean([len(n["children"]) for n in path]) if path else 0.0
            score = 0.6 * consensus + 0.25 * depth - 0.15 * min(branching / 3.0, 1.0)
            if score > best_score:
                best_score, best_ans = score, ans
        if best_ans is None:
            valid = [a for a in (_extract_pred(t, domain) for t in traces) if a]
            best_ans = Counter(valid).most_common(1)[0][0] if valid else None
        # The tree is a view over the traces; the answer was read from whichever
        # trace carried it, so that trace is what the step count measures.
        answer_trace = next((t for t in traces
                             if _extract_pred(t, domain) == best_ans), traces[0] if traces else "")
        return _pack(s, best_ans, traces, domain, answer_text=answer_trace,
                     tree_score=round(best_score, 4))

    return list(await asyncio.gather(*[one(s) for s in samples]))


# ===========================================================================
# DICE (Li et al., 2025)
# Decompose -> solve sub-problems independently -> verify -> recompose.
# ===========================================================================

_DICE_DECOMP = (
    "Break this problem into at most {n} independent sub-problems that together solve it.\n"
    "Problem:\n{problem}\n\nOutput one sub-problem per line, prefixed 'SUB:'. No solutions."
)
_DICE_SOLVE = "Solve this sub-problem concisely.\n\nContext problem:\n{problem}\n\nSub-problem:\n{sub}"
_DICE_VERIFY = (
    "Check each sub-answer for errors against the problem.\n"
    "Problem:\n{problem}\n\nSub-answers:\n{subs}\n\n"
    "For each, reply one line 'OK' or 'FIX: <corrected answer>', in order."
)
_DICE_RECOMPOSE = (
    "Compose the verified sub-answers into the final answer.\n"
    "Problem:\n{problem}\n\nVerified sub-answers:\n{subs}\n\nEnd with {fmt}."
)


async def run_dice(samples, model, api_key, base_url, semaphore, session, max_subs=4):
    domain = samples[0].get("domain", "logical") if samples else "logical"
    system, fmt = _sys_for(domain), _answer_format(domain)

    async def one(s):
        problem, texts = s["problem_text"], []
        dec = await call_llm(session, semaphore, system,
                             _DICE_DECOMP.format(problem=problem, n=max_subs),
                             model, api_key, base_url, temperature=0.0, max_tokens=512)
        texts.append(dec or "")
        subs = [l.strip()[4:].strip() for l in (dec or "").splitlines()
                if l.strip().upper().startswith("SUB:")][:max_subs]
        if not subs:
            subs = [problem]
        sols = await asyncio.gather(*[
            call_llm(session, semaphore, system, _DICE_SOLVE.format(problem=problem, sub=sub),
                     model, api_key, base_url, temperature=0.0, max_tokens=512) for sub in subs])
        texts.extend(s_ or "" for s_ in sols)
        joined = "\n".join(f"{i+1}. {sub}\n   -> {sol or '(no answer)'}"
                           for i, (sub, sol) in enumerate(zip(subs, sols)))
        ver = await call_llm(session, semaphore, system,
                             _DICE_VERIFY.format(problem=problem, subs=joined),
                             model, api_key, base_url, temperature=0.0, max_tokens=512)
        texts.append(ver or "")
        fixes = [l.strip() for l in (ver or "").splitlines() if l.strip()]
        verified = []
        for i, (sub, sol) in enumerate(zip(subs, sols)):
            fix = fixes[i] if i < len(fixes) else "OK"
            verified.append(f"{i+1}. {sub}\n   -> " +
                            (fix[4:].strip() if fix.upper().startswith("FIX:") else (sol or "")))
        final = await call_llm(session, semaphore, system,
                               _DICE_RECOMPOSE.format(problem=problem, subs="\n".join(verified), fmt=fmt),
                               model, api_key, base_url, temperature=0.0, max_tokens=1024)
        texts.append(final or "")
        return _pack(s, _extract_pred(final or "", domain), texts, domain,
                     answer_text=final or "", n_subs=len(subs),
                     artifacts={"sub_problems": subs, "sub_answers": list(sols),
                                "verified": verified, "final": final or ""})

    return list(await asyncio.gather(*[one(s) for s in samples]))


# ===========================================================================
# PNS-Optimization (Yu et al., 2025)
# Score every step by Probability of Necessity and Sufficiency, keep steps above
# the threshold, then re-derive the answer from the retained steps. K traces.
# ===========================================================================

_PNS_SCORE = (
    "For each numbered step below, estimate two probabilities with respect to reaching "
    "the correct answer of the problem:\n"
    "  N = probability the answer would become wrong if this step were removed (necessity)\n"
    "  S = probability this step alone drives the answer to be right (sufficiency)\n\n"
    "Problem:\n{problem}\n\nSteps:\n{steps}\n\n"
    "Reply one line per step, exactly: '<index> N=<0-1> S=<0-1>'. No other text."
)
_PNS_FINAL = (
    "These reasoning steps survived a causal-contribution filter.\n"
    "Problem:\n{problem}\n\nRetained steps:\n{steps}\n\n"
    "Derive the answer from them. End with {fmt}."
)


async def run_pns(samples, model, api_key, base_url, semaphore, session, k=10, threshold=0.5):
    domain = samples[0].get("domain", "logical") if samples else "logical"
    system, fmt = _sys_for(domain), _answer_format(domain)

    async def one(s):
        problem = s["problem_text"]
        user = build_zeroshot_prompt(problem, domain=domain)
        traces = await asyncio.gather(*[
            call_llm(session, semaphore, system, user, model, api_key, base_url,
                     temperature=0.7, max_tokens=1024) for _ in range(k)])
        traces = [t for t in traces if t]
        if not traces:
            return _pack(s, None, [], domain)

        step_lists = [_split_steps(t) for t in traces]
        scored = await asyncio.gather(*[
            call_llm(session, semaphore, system,
                     _PNS_SCORE.format(problem=problem,
                                       steps="\n".join(f"{i+1}. {x}" for i, x in enumerate(steps))),
                     model, api_key, base_url, temperature=0.0, max_tokens=512)
            if steps else _noop()
            for steps in step_lists])

        retained, n_drop = [], 0
        for steps, sc in zip(step_lists, scored):
            pns = {}
            for line in (sc or "").splitlines():
                m = re.match(r"\s*(\d+)\D+N\s*=\s*([01]?\.?\d*)\D+S\s*=\s*([01]?\.?\d*)", line, re.I)
                if m:
                    try:
                        pns[int(m.group(1))] = float(m.group(2)) * float(m.group(3))
                    except ValueError:
                        pass
            for i, st in enumerate(steps, 1):
                if pns.get(i, 1.0) >= threshold:
                    retained.append(st)
                else:
                    n_drop += 1
        if not retained:
            retained = [x for steps in step_lists for x in steps]

        seen, uniq = [], []
        for st in retained:
            b = _bag(st)
            if not any(_equivalent(b, o) for o in seen):
                seen.append(b); uniq.append(st)
        final = await call_llm(session, semaphore, system,
                               _PNS_FINAL.format(problem=problem, fmt=fmt,
                                                 steps="\n".join(f"- {x}" for x in uniq[:20])),
                               model, api_key, base_url, temperature=0.0, max_tokens=1024)
        pred = _extract_pred(final or "", domain)
        if pred is None:
            valid = [a for a in (_extract_pred(t, domain) for t in traces) if a]
            pred = Counter(valid).most_common(1)[0][0] if valid else None
        return _pack(s, pred, traces + [final], domain,
                     answer_text=final or "", steps_dropped=n_drop,
                     artifacts={"kept_steps": uniq, "n_dropped": n_drop,
                                "final": final or ""})

    return list(await asyncio.gather(*[one(s) for s in samples]))


async def _noop():
    return None


# ===========================================================================

async def run(args: argparse.Namespace) -> None:
    samples = load_raw_dataset([Path(p) for p in args.datasets], seed=args.seed,
                               per_dataset=args.per_dataset, max_samples=args.max_samples)
    logger.info("Loaded %d samples", len(samples))
    sem = asyncio.Semaphore(args.concurrency)
    names = {
        "TOT":       f"Tree-of-Thought (width={args.tot_width}, depth={args.tot_depth})",
        "RAP":       f"RAP (MCTS, {args.rap_iterations} iters, branch={args.rap_branch})",
        "CONCISE":   "ConCISE (confidence-guided)",
        "LCOT2TREE": f"LCoT2Tree (K={args.shots})",
        "DICE":      f"DICE (decompose<={args.dice_subs})",
        "PNS":       f"PNS-Optimization (K={args.shots}, thr={args.pns_threshold})",
    }
    results: List[Tuple[str, Dict]] = []
    out: Dict[str, Any] = {}
    timeout = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for key in args.settings:
            logger.info("Running %s ...", names[key])
            if key == "TOT":
                preds = await by_domain(run_tot, samples, args.model, args.api_key, args.base_url, sem, session,
                                      width=args.tot_width, depth=args.tot_depth)
            elif key == "RAP":
                preds = await by_domain(run_rap, samples, args.model, args.api_key, args.base_url, sem, session,
                                      iterations=args.rap_iterations, branch=args.rap_branch,
                                      max_depth=args.rap_depth)
            elif key == "CONCISE":
                preds = await by_domain(run_concise, samples, args.model, args.api_key, args.base_url, sem, session)
            elif key == "LCOT2TREE":
                preds = await by_domain(run_lcot2tree, samples, args.model, args.api_key, args.base_url, sem,
                                            session, k=args.shots)
            elif key == "DICE":
                preds = await by_domain(run_dice, samples, args.model, args.api_key, args.base_url, sem, session,
                                       max_subs=args.dice_subs)
            else:
                preds = await by_domain(run_pns, samples, args.model, args.api_key, args.base_url, sem, session,
                                      k=args.shots, threshold=args.pns_threshold)
            m = compute_metrics(preds)
            m["per_dataset"] = compute_per_dataset(preds)
            print_metrics(m, names[key])
            results.append((names[key], m))
            out[key] = {"setting": names[key], "overall": m, "predictions": preds}

    print_comparison_table(results)
    op = Path(args.output); op.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(op, "w"), ensure_ascii=False, indent=1)
    summary = {"settings": [{"key": k, "name": v["setting"], **{kk: vv for kk, vv in v["overall"].items()
                                                                if kk != "per_dataset"}} for k, v in out.items()],
               "model": args.model, "shots": args.shots, "seed": args.seed,
               "per_dataset": args.per_dataset}
    json.dump(summary, open(op.with_suffix(".summary.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\nSaved: {op}")


def main() -> None:
    p = argparse.ArgumentParser(description="RAP / ConCISE / LCoT2Tree / DICE / PNS baselines")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--per_dataset", type=int, default=250)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--settings", nargs="+", choices=ALL_SETTINGS, default=ALL_SETTINGS)
    p.add_argument("--shots", type=int, default=10, help="K for LCoT2Tree / PNS (paper: 10)")
    p.add_argument("--tot_width", type=int, default=5, help="ToT branching width (paper: 5)")
    p.add_argument("--tot_depth", type=int, default=2, help="ToT search depth (paper: 2)")
    p.add_argument("--rap_iterations", type=int, default=8, help="RAP MCTS iterations")
    p.add_argument("--rap_branch", type=int, default=3, help="RAP expansion width")
    p.add_argument("--rap_depth", type=int, default=4, help="RAP max planning depth")
    p.add_argument("--dice_subs", type=int, default=4, help="DICE max sub-problems")
    p.add_argument("--pns_threshold", type=float, default=0.5, help="PNS retention threshold")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--api_key", default=OPENAI_API_KEY)
    p.add_argument("--base_url", default=OPENAI_BASE_URL)
    p.add_argument("--concurrency", type=int, default=30)
    p.add_argument("--output", default="extra_baseline_results.json")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
