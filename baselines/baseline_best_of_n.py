#!/usr/bin/env python3
"""Best-of-N baseline (main table).

Generates K candidate traces and keeps the one the model scores highest for itself —
selection rather than aggregation, which is what separates it from the voting methods.

Run standalone:
    python baseline_best_of_n.py --datasets ../dataset/FLD.json --per_dataset 100 \
        --model o4-mini --api_key <key> --base_url <url> --output out.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path

import aiohttp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_common  # noqa: E402
from baseline_common import (  # noqa: E402
    attach, save_run, by_domain, run_chunked, resume_filter,
    SYSTEM_LOGICIAN, SYSTEM_MATH, build_zeroshot_prompt, call_llm, compute_metrics,
    compute_per_dataset, count_steps, count_tokens, load_raw_dataset, print_metrics,
    _extract_pred, DEFAULT_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL, default_output,
)

LABEL = "Best-of-N (N=10, self-scored)"
logger = logging.getLogger("best_of_n")

_SCORE = (
    "Problem:\n{problem}\n\nCandidate solution:\n{cand}\n\n"
    "Rate how likely this solution is correct. Reply with ONLY a number 0.0-1.0."
)


def _score(text):
    m = re.search(r"([01](?:\.\d+)?|\.\d+)", text or "")
    try:
        return max(0.0, min(1.0, float(m.group(1)))) if m else 0.0
    except ValueError:
        return 0.0


async def run_best_of_n(samples, n, model, api_key, base_url, semaphore, session):
    domain = samples[0].get("domain", "logical") if samples else "logical"
    system = SYSTEM_MATH if domain == "math" else SYSTEM_LOGICIAN

    async def one(s):
        user = build_zeroshot_prompt(s["problem_text"], domain=domain)
        cands = await asyncio.gather(*[
            call_llm(session, semaphore, system, user, model, api_key, base_url,
                     temperature=0.7, max_tokens=1024) for _ in range(n)])
        cands = [c for c in cands if c]
        if not cands:
            return attach({"sample_id": s["sample_id"], "source_dataset": s["source_dataset"],
                           "ground_truth": s["ground_truth"], "predicted": None,
                           "response_steps": 0.0, "response_tokens": 0.0, "domain": domain},
                          traces=[], scores=[], chosen=None)
        scores = await asyncio.gather(*[
            call_llm(session, semaphore, system,
                     _SCORE.format(problem=s["problem_text"], cand=c),
                     model, api_key, base_url, temperature=0.0, max_tokens=16)
            for c in cands])
        numeric = [_score(x) for x in scores]
        chosen  = int(max(range(len(cands)), key=lambda i: numeric[i]))
        best    = cands[chosen]
        # The rejected candidates and the self-scores that rejected them are the
        # whole of what Best-of-N did; keeping only the winner hides the choice.
        return attach({"sample_id": s["sample_id"], "source_dataset": s["source_dataset"],
                       "ground_truth": s["ground_truth"], "predicted": _extract_pred(best, domain),
                       # Best-of-N reports the candidate it picked, so that is
                       # what the step count measures — not the mean over the
                       # candidates it rejected.
                       "response_steps": float(count_steps(best)),
                       "response_tokens": float(count_tokens(best)),
                       "domain": domain},
                      traces=cands, scores=numeric, raw_scores=[x or "" for x in scores],
                      chosen=chosen,
                      candidate_labels=[_extract_pred(c, domain) for c in cands])

    return list(await asyncio.gather(*[one(s) for s in samples]))


async def run(args) -> None:
    samples = load_raw_dataset([Path(p) for p in args.datasets], seed=args.seed,
                               per_dataset=args.per_dataset, max_samples=args.max_samples)
    logger.info("Loaded %d samples", len(samples))
    samples, previous = resume_filter(samples, args.output, args.resume)
    if not samples:
        logger.info("Nothing left to run")
        return
    sem = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as session:
        preds = await run_chunked(run_best_of_n, samples, previous, args.output, LABEL, "best_of_n", args.chunk,
            {"model": args.model, "shots": args.shots, "seed": args.seed, "per_dataset": args.per_dataset}, args.shots, args.model, args.api_key,
                                    args.base_url, sem, session)
    m = compute_metrics(preds)
    m["per_dataset"] = compute_per_dataset(preds)
    print_metrics(m, LABEL)


def main() -> None:
    p = argparse.ArgumentParser(description=LABEL)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--per_dataset", type=int, default=100)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--chunk", type=int, default=200,
                   help="save after this many samples, so a run that dies "
                        "loses at most one chunk")
    p.add_argument("--max_tokens", type=int, default=0,
                   help="raise the output budget floor for every call (0 = leave each "
                        "call's own ceiling alone). OlympiadBench needs about 2048.")
    p.add_argument("--no_resume", dest="resume", action="store_false",
                   help="re-run every sample instead of continuing from what is on disk")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shots", type=int, default=10, help="N candidates (paper: 10)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--api_key", default=OPENAI_API_KEY)
    p.add_argument("--base_url", default=OPENAI_BASE_URL)
    p.add_argument("--concurrency", type=int, default=30)
    p.add_argument("--output", default=None,
                   help="Default: baseline_results/<model>/best_of_n/results.json "
                        "under Part2_CRAFT/results")
    args = p.parse_args()
    if args.max_tokens:
        baseline_common.MAX_TOKENS_FLOOR = args.max_tokens
    args.output = args.output or default_output("best_of_n", args.model)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
