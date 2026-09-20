#!/usr/bin/env python3
"""ConCISE (confidence-guided) baseline (main table).

Run standalone:
    python baseline_concise.py --datasets ../dataset/FLD.json --per_dataset 100 \
        --model o4-mini --api_key <key> --base_url <url> --output out.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_common  # noqa: E402
from baseline_common import (  # noqa: E402
    load_raw_dataset, compute_metrics, compute_per_dataset, print_metrics,
    DEFAULT_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL, default_output, save_run, by_domain, run_chunked,
    resume_filter,
)
from baseline_common_extended import run_concise  # noqa: E402

LABEL = "ConCISE (confidence-guided)"
logger = logging.getLogger("concise")


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
        preds = await run_chunked(run_concise, samples, previous, args.output, LABEL, "concise", args.chunk,
            {"model": args.model, "shots": args.shots, "seed": args.seed, "per_dataset": args.per_dataset}, args.model, args.api_key, args.base_url, sem, session)
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
    p.add_argument("--shots", type=int, default=10, help="K candidate traces (paper: 10)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--api_key", default=OPENAI_API_KEY)
    p.add_argument("--base_url", default=OPENAI_BASE_URL)
    p.add_argument("--concurrency", type=int, default=30)
    p.add_argument("--output", default=None,
                   help="Default: baseline_results/<model>/concise/results.json "
                        "under Part2_CRAFT/results")
    args = p.parse_args()
    if args.max_tokens:
        baseline_common.MAX_TOKENS_FLOOR = args.max_tokens
    args.output = args.output or default_output("concise", args.model)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
