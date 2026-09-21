#!/usr/bin/env python3
"""A directed proof attempt on the conclusions that say no proof exists.

ProofWriter's slice is closed-world at a known depth: `__DISPROVED__` does not
mean the hypothesis was refuted, it means no derivation of it was found, and
that is only sound if the search was exhaustive. A model reaches it by running
out of ideas, which is a different event from proving a negative.

The traces show what that costs. Of nano's 25 wrong answers on the 500-sample
slice, 24 are a provable hypothesis read as `__DISPROVED__` and 1 is the
reverse; gemini's split is 91 against 72. So the errors sit almost entirely on
one side, and they sit there for a reason a second pass can address: the first
pass was searching for anything interesting, this one is searching for one
specific chain.

What this pass does NOT do is flip on the model's say-so. It asks for the
derivation, and a conclusion of `__PROVED__` is accepted only when the reply
actually cites the premises it chained — a bare "yes, it follows" is discarded
and the original answer stands. Everything it sees is the problem and its own
earlier answer; no gold label is read, and samples already answered
`__PROVED__` are not touched, so the pass can only move in the direction the
closed-world asymmetry predicts.

    python cwa_recheck.py --synth <synthesized.json> --k_traces <k_traces.json> \
        --expected_depth 5 --model <m> --output <rechecked.json>
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
from synthesize_trace import generate_reasoning_trace  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as _cfg  # noqa: E402

LABEL_RE = re.compile(r"(__PROVED__|__DISPROVED__)", re.IGNORECASE)
# A derivation names what it chained. Two or more references is the bar; the
# replies that flip an answer without them are the ones that assert rather
# than derive.
CITE_RE = re.compile(r"\b(?:fact|rule|sent|premise|statement)\s*#?\d+", re.IGNORECASE)
MIN_CITATIONS = 2


def last_label(text: str) -> Optional[str]:
    hits = LABEL_RE.findall(text or "")
    return f"__{hits[-1].upper().strip('_')}__" if hits else None


def build_prompt(problem: str, depth: Optional[int]) -> str:
    budget = (f"The dataset's hypotheses are derivable in at most {depth} steps "
              f"when they are derivable at all, so a chain longer than that is a "
              f"sign of a wrong turn.\n" if depth else "")
    return (
        "You are checking one specific thing about a closed-world deduction problem.\n\n"
        f"{problem}\n\n"
        "An earlier attempt concluded that the hypothesis cannot be derived. That "
        "conclusion is only correct if the search was exhaustive, and a search that "
        "simply ran out of ideas looks the same from the outside. So search again, "
        "this time for one specific thing: a forward chain from the given facts, "
        "through the given rules, that ends at the hypothesis.\n"
        f"{budget}"
        "Work forward from the facts. At each step name the fact or rule you are "
        "using and what it lets you conclude.\n\n"
        "Then finish in exactly one of these two ways:\n"
        "  - If you reached the hypothesis: write the chain, then __PROVED__\n"
        "  - If every rule that could apply has been applied and the hypothesis is "
        "still not reachable: say which rules you tried, then __DISPROVED__\n\n"
        "Do not write __PROVED__ unless you can show the chain that reaches it."
    )


async def recheck_one(session, sem, rec, problem, depth, model) -> Dict[str, Any]:
    async with sem:
        reply = await generate_reasoning_trace(session, build_prompt(problem, depth), model)
    label = last_label(reply or "")
    cites = len(CITE_RE.findall(reply or ""))
    flipped = label == "__PROVED__" and cites >= MIN_CITATIONS
    out = dict(rec)
    out["cwa_recheck"] = {"label": label, "citations": cites, "flipped": flipped}
    if flipped:
        out["synthesized_trace"] = (rec.get("synthesized_trace") or "").rstrip() + \
            "\n\n[Directed proof search]\n" + (reply or "").strip()
    return out


async def main_async(args) -> None:
    synth_path = Path(_cfg.resolve_input(args.synth))
    raw = json.loads(synth_path.read_text(encoding="utf-8"))
    rows = raw.get("results", raw) if isinstance(raw, dict) else raw

    kraw = json.loads(Path(_cfg.resolve_input(args.k_traces)).read_text(encoding="utf-8"))
    krows = kraw.get("results", kraw) if isinstance(kraw, dict) else kraw
    problems = {r["sample_id"]: r.get("problem_text") or "" for r in krows}

    targets = [r for r in rows
               if last_label(r.get("synthesized_trace") or "") == "__DISPROVED__"]
    print(f"  {len(rows)} samples, {len(targets)} answered __DISPROVED__ — rechecking those")

    sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        done = await asyncio.gather(*[
            recheck_one(session, sem, r, problems.get(r["sample_id"], ""),
                        args.expected_depth, args.model)
            for r in targets])

    by_id = {r["sample_id"]: r for r in done}
    merged = [by_id.get(r["sample_id"], r) for r in rows]
    n_flip = sum(1 for r in done if r["cwa_recheck"]["flipped"])
    said_proved = sum(1 for r in done if r["cwa_recheck"]["label"] == "__PROVED__")
    print(f"  said __PROVED__: {said_proved}   of those with a cited chain: {n_flip}")

    out = Path(_cfg.resolve_output(args.output))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=1), encoding="utf-8")
    print(f"  Saved: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth", required=True)
    ap.add_argument("--k_traces", required=True)
    ap.add_argument("--expected_depth", type=int, default=None)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--base_url", default=None)
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    # synthesize_trace builds its endpoint and headers at import, so pointing
    # this run at a different gateway means setting them on that module.
    import synthesize_trace as _st
    if args.api_key:
        _st.OPENAI_API_KEY = args.api_key
        _st.HEADERS = {**_st.HEADERS,
                       "Authorization": _st._build_auth_header(
                           args.api_key, args.base_url or _st.OPENAI_BASE_URL)}
    if args.base_url:
        _st.OPENAI_BASE_URL = args.base_url
        _st.CHAT_COMPLETIONS_URL = args.base_url.rstrip("/") + "/chat/completions"
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
