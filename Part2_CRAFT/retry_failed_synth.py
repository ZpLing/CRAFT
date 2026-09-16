#!/usr/bin/env python3
"""
retry_failed_synth.py
─────────────────────
Re-synthesize only the samples that failed in a prior run (no pred_label).
Merges the new results back into the original synthesized JSON.

Usage:
  python retry_failed_synth.py \
    --synth alignment_comparison/synthesized_baseline_iter1.json \
    --rkg   alignment_comparison/rkg_baseline.json \
    --cleaned alignment_comparison/cleaned_with_problem.json \
    --model gpt-5.4-nano \
    --concurrency 4 \
    [--anchor_conclusion]
"""

from __future__ import annotations
import argparse
import asyncio
import json
import sys
from pathlib import Path

import aiohttp
import backoff

sys.path.insert(0, str(Path(__file__).parent))

from config import require_bosch, resolve_input
import module3_synthesis.synthesize_trace as _synth_mod
from module1_trace_generation.extract_terms import (
    DocFreqTable,
    FlatDocFreqTable,
    resolve_df_table_path,
)
from module3_synthesis.synthesize_trace import synthesize_trace_rkg

# Pinned endpoint, read from the gitignored repo-root config.py.
BOSCH_KEY, BOSCH_URL = require_bosch()


def patch_creds(model: str):
    _synth_mod.OPENAI_API_KEY        = BOSCH_KEY
    _synth_mod.OPENAI_BASE_URL       = BOSCH_URL
    _synth_mod.CHAT_COMPLETIONS_URL  = BOSCH_URL.rstrip("/") + "/chat/completions"
    _synth_mod.HEADERS["Authorization"] = f"Bearer {BOSCH_KEY}"
    _synth_mod.DEFAULT_MODEL         = model
    _synth_mod.REQUEST_TIMEOUT       = 60

    @backoff.on_exception(backoff.expo,
                          (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError),
                          max_tries=4, factor=2, max_value=20)
    async def _gen(session, prompt, model=model, max_tokens=None):
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "temperature": _synth_mod.REQUEST_TEMPERATURE,
                   "max_tokens": max_tokens or _synth_mod.RESPONSE_TOKENS}
        async with session.post(_synth_mod.CHAT_COMPLETIONS_URL, json=payload,
                                headers=_synth_mod.HEADERS,
                                timeout=aiohttp.ClientTimeout(total=_synth_mod.REQUEST_TIMEOUT)) as resp:
            if resp.status != 200:
                detail = await resp.text()
                raise RuntimeError(f"HTTP {resp.status}: {detail[:200]}")
            data = await resp.json()
            return data["choices"][0]["message"].get("content", "") or ""
    _synth_mod.generate_reasoning_trace = _gen


async def main_async(args):
    patch_creds(args.model)

    # Load synth (find failures)
    with open(args.synth) as f:
        sd = json.load(f)
    synth_list = sd.get("results", sd) if isinstance(sd, dict) else sd
    failed_sids = {r["sample_id"] for r in synth_list if not r.get("pred_label")}
    print(f"Failed samples to retry: {len(failed_sids)}")

    if not failed_sids:
        print("Nothing to retry.")
        return

    # Load RKG
    with open(args.rkg) as f:
        rd = json.load(f)
    rkg_list = rd.get("results", rd) if isinstance(rd, dict) else rd
    rkg_lookup = {r["sample_id"]: r for r in rkg_list}

    # Load cleaned (problem_text source)
    with open(args.cleaned) as f:
        cd = json.load(f)
    cleaned_list = cd.get("results", cd) if isinstance(cd, dict) else cd
    cleaned_lookup = {s["sample_id"]: s for s in cleaned_list}

    # Re-synthesize
    sem = asyncio.Semaphore(args.concurrency)
    new_results = {}
    connector = aiohttp.TCPConnector(limit=args.concurrency * 3)

    # Must match the scope the original run used, or the retried samples are scored on a
    # different term weighting than the ones beside them in the same output file.
    df_table = None
    if args.idf_scope == "global":
        if not args.df_table:
            raise ValueError(
                "--idf_scope global requires --df_table pointing at the table the original "
                "run saved; a retry only sees the failed subset and cannot rebuild that corpus"
            )
        df_table = DocFreqTable.load(Path(args.df_table))
        print(f"IRF scope: global | {df_table.n_docs} step documents, {len(df_table)} terms")
        if df_table.normalize != (args.idf_norm == "log_n"):
            raise ValueError(
                f"--df_table was built with idf_norm="
                f"{'log_n' if df_table.normalize else 'raw'}, but this retry asks for "
                f"{args.idf_norm}; match the flag the original run used"
            )
    elif args.idf_scope == "none":
        df_table = FlatDocFreqTable()
        print("IRF scope: none (IRF factor disabled, TF-IRF == TF)")
    else:
        print("IRF scope: sample (IDF computed within each sample's own steps)")
    print(f"IDF scale: {args.idf_norm}")

    async with aiohttp.ClientSession(connector=connector) as session:
        async def worker(sid):
            async with sem:
                sample = cleaned_lookup.get(sid)
                rkg = rkg_lookup.get(sid)
                if not sample or not rkg:
                    return sid, {"sample_id": sid, "error": "missing_input", "synthesized_trace": None}
                try:
                    res = await synthesize_trace_rkg(
                        session, sample, rkg, model=args.model,
                        domain="logical", anchor_conclusion=args.anchor_conclusion,
                        df_table=df_table, idf_norm=(args.idf_norm == "log_n"),
                    )
                    return sid, res
                except Exception as e:
                    return sid, {"sample_id": sid, "error": f"retry_failed: {e}", "synthesized_trace": None}

        tasks = [asyncio.create_task(worker(sid)) for sid in failed_sids]
        done = 0
        for coro in asyncio.as_completed(tasks):
            sid, res = await coro
            new_results[sid] = res
            done += 1
            ok = bool(res.get("pred_label"))
            err = (res.get("error") or "")[:60]
            print(f"  [{done}/{len(failed_sids)}] {sid} pred={res.get('pred_label')!r} err={err!r}")

    # Merge into synth list
    merged = []
    for r in synth_list:
        sid = r["sample_id"]
        if sid in new_results and new_results[sid].get("pred_label"):
            merged.append(new_results[sid])
        else:
            merged.append(r)

    # Write back
    out_path = Path(args.synth)
    if args.in_place:
        backup = out_path.with_suffix(".bak.json")
        out_path.rename(backup)
        print(f"Backup: {backup}")
    else:
        out_path = out_path.with_name(out_path.stem + "_retried.json")
    with open(out_path, "w") as f:
        json.dump({"results": merged}, f, indent=2)

    # Summary
    n_total = len(merged)
    n_with_pred = sum(1 for r in merged if r.get("pred_label"))
    print(f"\nWrote: {out_path}")
    print(f"Total: {n_total}, with pred_label: {n_with_pred}, recovered: {sum(1 for r in new_results.values() if r.get('pred_label'))}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synth", required=True,
                   help="Synthesis output to repair; relative paths resolve under the results root")
    p.add_argument("--rkg",   required=True)
    p.add_argument("--cleaned", required=True)
    p.add_argument("--model", default="gpt-5.4-nano")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--anchor_conclusion", action="store_true")
    p.add_argument("--in_place", action="store_true")
    p.add_argument("--idf_scope", default="sample", choices=["sample", "global", "none"],
                   help="Must match the --idf_scope the original run used, otherwise retried "
                        "samples get a different term weighting than the rest of the file")
    p.add_argument("--idf_norm", default="raw", choices=["raw", "log_n"],
                   help="Must match the --idf_norm the original run used")
    p.add_argument("--df_table", default=None,
                   help="Global DF table saved by the original run (--idf_scope global). "
                        "Required to reproduce that run's IRF scores on a retried subset")

    args = p.parse_args()
    if args.idf_scope == "global" and not args.df_table:
        p.error("--idf_scope global requires --df_table pointing at the table the original "
                "run saved; a retry only sees the failed subset and cannot rebuild that corpus")
    args.synth   = str(resolve_input(args.synth))
    args.rkg     = str(resolve_input(args.rkg))
    args.cleaned = str(resolve_input(args.cleaned))
    if args.df_table:
        args.df_table = str(resolve_df_table_path(args.df_table))
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
