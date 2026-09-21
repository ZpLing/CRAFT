#!/usr/bin/env python3
"""Work the problem again where the traces disagree, and keep the answer it lands on.

The headroom on the mathematical sets is entirely in the samples where the k
traces do not agree. Where all five agree there is nothing to find — the oracle
equals the vote, 92% on OlympiadBench and 85% on Omni-MATH, so no method that
picks among the traces can do better there. Where they split, the oracle runs
20 to 30 points above the vote and synthesis recovers almost none of it: 46% to
48% on the three-two split, 42% to 42% on Omni-MATH's.

So this pass runs only on the split samples, and rather than asking which
candidate looks right — checking a claim is the direction that failed on
ProofWriter at 22% precision against 92% for searching — it asks for the
problem to be worked again, from the problem statement, and only then compares
what it reached against the candidates. An answer is taken only when the fresh
derivation lands on one of them; when it lands somewhere else, or nowhere, the
consensus answer stands. That keeps the pass to positive evidence, and keeps it
from inventing a third answer that no trace ever produced.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthesize_trace as _st
from synthesize_trace import generate_reasoning_trace, _extract_boxed_content

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as _cfg
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation" / "label_prediction"))
from answer_match import answers_match  # noqa: E402


def boxed(text: str) -> Optional[str]:
    hits = _extract_boxed_content(text or "")
    return hits[-1] if hits else None


def candidates(traces: List[Dict], dataset: str) -> List[List[str]]:
    """The distinct answers the traces reached, largest group first."""
    groups: List[List[str]] = []
    for t in traces:
        a = t.get("label")
        if not a:
            continue
        for g in groups:
            if answers_match(a, g[0], dataset):
                g.append(a)
                break
        else:
            groups.append([a])
    return sorted(groups, key=len, reverse=True)


def build_prompt(problem: str, options: List[str]) -> str:
    listed = "\n".join(f"  ({i + 1}) {o}" for i, o in enumerate(options))
    return (
        "Solve this problem from scratch. Independent attempts at it disagreed, "
        "so work it yourself before looking at what they said.\n\n"
        f"{problem}\n\n"
        "Work the problem step by step. Then, and only then, compare what you "
        "reached with these answers that the earlier attempts gave:\n"
        f"{listed}\n\n"
        "Finish with \\boxed{<your answer>} — the answer YOUR derivation "
        "reached. If it matches one of the listed answers, box that. If your "
        "derivation disagrees with all of them, box your own."
    )


async def one(session, sem, rec, problem, model, dataset) -> Dict[str, Any]:
    traces = rec.get("traces") or []
    groups = candidates(traces, dataset)
    out = dict(rec)
    if len(groups) < 2:
        out["adjudication"] = {"ran": False, "reason": "traces agree"}
        return out
    options = [g[0] for g in groups]
    async with sem:
        reply = await generate_reasoning_trace(session, build_prompt(problem, options), model)
    got = boxed(reply or "")
    picked = None
    if got:
        for o in options:
            if answers_match(got, o, dataset):
                picked = o
                break
    out["adjudication"] = {"ran": True, "reply_answer": got, "picked": picked,
                           "options": options, "consensus": options[0],
                           "text": (reply or "").strip()}
    return out


async def main_async(args) -> None:
    raw = json.loads(Path(_cfg.resolve_input(args.k_traces)).read_text(encoding="utf-8"))
    recs = raw.get("results", raw) if isinstance(raw, dict) else raw
    split = [r for r in recs if len(candidates(r.get("traces") or [], args.dataset)) > 1]
    print(f"  {len(recs)} samples, {len(split)} where the traces disagree")

    sem = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as session:
        done = await asyncio.gather(*[
            one(session, sem, r, r.get("problem_text") or "", args.model, args.dataset)
            for r in split])

    by_id = {r["sample_id"]: r for r in done}
    merged = [by_id.get(r["sample_id"], r) for r in recs]
    n_pick = sum(1 for r in done if (r.get("adjudication") or {}).get("picked"))
    n_moved = sum(1 for r in done
                  if (a := r.get("adjudication") or {}).get("picked")
                  and not answers_match(a["picked"], a["consensus"], args.dataset))
    print(f"  landed on a listed answer: {n_pick}/{len(split)}   "
          f"of those, different from the consensus: {n_moved}")

    out = Path(_cfg.resolve_output(args.output))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged), encoding="utf-8")
    print(f"  Saved: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k_traces", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--base_url", default=None)
    ap.add_argument("--concurrency", type=int, default=60)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.api_key:
        _st.OPENAI_API_KEY = args.api_key
        _st.HEADERS = {**_st.HEADERS,
                       "Authorization": _st._build_auth_header(
                           args.api_key, args.base_url or _st.OPENAI_BASE_URL)}
    if args.base_url:
        _st.CHAT_COMPLETIONS_URL = args.base_url.rstrip("/") + "/chat/completions"
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
