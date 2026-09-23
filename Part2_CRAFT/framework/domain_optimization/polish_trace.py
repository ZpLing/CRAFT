#!/usr/bin/env python3
"""Rewrite a synthesized trace for how it reads, with its answer pinned.

Module III writes the derivation and the passes after it settle the answer;
this pass changes neither. It hands the trace back to the model with the
conclusion it must end on and asks for the same steps, written to read well:
complete sentences, each thing said once, quotations kept word for word. A
rewrite is accepted only if it still reads as the same answer, keeps the
step structure, and carries none of the shorthand the styles forbid.
Otherwise the trace is left as it was, so a cell's accuracy cannot move.

The styles are the experiments. Each names one hypothesis about what the
trace should read like; the record keeps which style produced it.

    python polish_trace.py --synth <synth.json> --k_traces <k_traces.json> \
        --dataset ProofWriter --style two3 --model <m> --output <out.json>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "module3_topology_guided_synthesis"))
from synthesize_trace import generate_reasoning_trace  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as _cfg  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation" / "label_prediction"))
from answer_match import answers_match  # noqa: E402
from extract_label import extract_label, extract_math_answer  # noqa: E402

LOGIC = {"ProofWriter", "FLD"}
_STEP_HEAD = re.compile(r"(?m)^\s*\**\s*Step\s*(\d+)\s*\**\s*[:.\-]\s*")
# What no style may leave behind: markdown structure, bullets, the searcher's
# shorthand, arrows standing for "then".
_FORBIDDEN = re.compile(
    r"(?m)^\s*(?:#{1,6}\s|[-*•]\s|\d+\.\s)|\b[FR]\d+\b|->|=>|→|⇒|\[(?:Two-sided|Directed) proof search\]")

STYLES: Dict[str, str] = {
    "varied2": (
        "- Keep every step in order under its 'Step N:' header, and keep the same "
        "number of steps and the same claims and citations.\n"
        "- Every quotation of a fact or rule stays exactly as written, wherever it "
        "occurs, including when the same fact is applied again.\n"
        "- Rewrite only the sentences around the quotations: no two of them in the "
        "whole trace may share their opening words or their main verb phrase; a "
        "result derived earlier is referred to by its step, never restated in the "
        "same words.\n"
        "- Complete sentences, with articles and verbs. No bullet points, no "
        "headings, no arrows, no shorthand such as F1 or R8.\n"
        "- Do not add reasoning, hedging or commentary, and do not drop a derivation."
    ),
    "two4": (
        "- Keep every step in order under its 'Step N:' header, and keep the same "
        "number of steps.\n"
        "- Each step is exactly two sentences.\n"
        "  The first begins with 'According to Fact N,' and then gives that fact or "
        "rule in its own words, ending with a full stop: 'According to Fact 17, if "
        "something is red and young then it chases the bald eagle.'\n"
        "  The second applies it in one complete sentence that names each premise "
        "with the Fact or an EARLIER Step that established it (a step never cites "
        "itself or a later step) and states what follows: 'Since Step 2 established "
        "that the tiger is red and Fact 4 states that the tiger is young, it follows "
        "that the tiger chases the bald eagle.'\n"
        "- Every sentence has its subject, verb, articles and commas. No quotation "
        "marks except around the hypothesis in the last step, no arrows, no shorthand "
        "such as F1 or R8, no parentheses, no bullet points, no headings.\n"
        "- Say each thing once; do not restate the hypothesis inside the steps. The "
        "last step reads: 'Since Step M established that <statement>, the hypothesis "
        "\u201c<the hypothesis exactly as the problem states it>\u201d is <label>.'\n"
        "- Do not add reasoning, hedging or commentary, and do not drop a derivation."
    ),
    "two3": (
        "- Keep every step in order under its 'Step N:' header, and keep the same "
        "number of steps.\n"
        "- Each step is exactly two sentences.\n"
        "  The first begins with 'According to Fact N,' and then gives that fact or "
        "rule in its own words, ending with a full stop: 'According to Fact 17, if "
        "something is red and young then it chases the bald eagle.'\n"
        "  The second applies it in one complete sentence that names each premise "
        "with the Fact or earlier Step that established it and states what follows: "
        "'Since Step 2 established that the tiger is red and Fact 4 states that the "
        "tiger is young, it follows that the tiger chases the bald eagle.'\n"
        "- Every sentence has its subject, verb, articles and commas. No quotation "
        "marks, no arrows, no shorthand such as F1 or R8, no parentheses, no bullet "
        "points, no headings.\n"
        "- Say each thing once; do not restate the hypothesis inside the steps; only "
        "the final step names it, as 'Since Step M established that <statement>, the "
        "hypothesis is <label>.'\n"
        "- Do not add reasoning, hedging or commentary, and do not drop a derivation."
    ),
    "norecap2": (
        "- Keep every step in order under its 'Step N:' header, and keep the same "
        "number of steps.\n"
        "- A step may begin by naming the condition of the problem it uses, in the "
        "problem's own words ('Using the given condition that the sum of the digits "
        "is 12, ...'). It never repeats or re-derives what an earlier step computed: "
        "it refers to that result by its step ('From Step 3, $x=-r$ is a root, so "
        "...') and continues from there. Each equation, substitution and numerical "
        "result appears once in the whole trace.\n"
        "- The final answer is stated once, in the last step only.\n"
        "- Every sentence is a complete sentence; all mathematics is written in "
        "LaTeX inside $...$, attached to a sentence rather than standing alone as a "
        "line. No bullet points, no headings, no arrows outside mathematics.\n"
        "- Do not add reasoning, hedging or commentary, and do not drop a derivation."
    ),
}


def reader(dataset: str):
    return extract_label if dataset in LOGIC else extract_math_answer


def closing_for(answer: str, dataset: str) -> str:
    if dataset in LOGIC:
        return answer
    return answer if answer.startswith("\\boxed") else f"\\boxed{{{answer}}}"


def build_prompt(problem: str, trace: str, style: str, closing: str) -> str:
    return (
        "Rewrite the reasoning trace below so that it reads well, keeping what it "
        "says exactly.\n\n"
        f"Problem:\n{problem}\n\n"
        f"Trace:\n{trace}\n\n"
        "Rules:\n"
        f"{STYLES[style]}\n"
        f"- The last step must reach the same conclusion and end with exactly: {closing}\n\n"
        "Output the rewritten trace only."
    )


# A citation, alone or in a run: "Step 4", "Steps 7, 8, 5, and 4".
_CITE = re.compile(r"\bSteps?\s*\d+(?:\s*,\s*(?:and\s+)?\d+)*(?:\s*,?\s*and\s+\d+)?")


def bad_citation(text: str) -> Optional[str]:
    """A step that cites itself or a later step, or None.

    gpt-5.4-nano's rewrites did this in 62 of 50 ProofWriter traces -- "Step 3:
    ... Since Step 3 established ..." -- where the step before was meant.
    """
    parts = _STEP_HEAD.split(text)
    current = None
    for i, part in enumerate(parts):
        if i % 2 == 1:
            current = int(part)
            continue
        if current is None:
            continue
        for m in _CITE.finditer(part):
            for n in re.findall(r"\d+", m.group(0)):
                if int(n) >= current:
                    return f"Step {current} cites Step {n}"
    return None


def accept(new: str, old: str, dataset: str) -> Optional[str]:
    """None if the rewrite may replace the trace, else why not."""
    ex = reader(dataset)
    a, b = ex(new) or "", ex(old) or ""
    if not b:
        return "no answer in the original"
    if not a or not answers_match(a, b, dataset):
        return f"answer {a!r} != {b!r}"
    n_new, n_old = len(_STEP_HEAD.findall(new)), len(_STEP_HEAD.findall(old))
    if n_new < 2:
        return "no steps"
    if n_old and not (0.6 * n_old <= n_new <= 1.4 * n_old):
        return f"steps {n_old} -> {n_new}"
    if _FORBIDDEN.search(new):
        return "forbidden markup: " + _FORBIDDEN.search(new).group(0)[:20]
    if len(new) > 2.0 * len(old):
        return "twice as long"
    if dataset in LOGIC:
        why = bad_citation(new)
        if why:
            return "citation: " + why
    return None


async def polish_one(session, sem, rec, problem, dataset, style, model) -> Dict[str, Any]:
    old = rec.get("synthesized_trace") or ""
    out = dict(rec)
    note: Dict[str, Any] = {"style": style, "model": model}
    answer = reader(dataset)(old) or ""
    if not old.strip() or not answer:
        note["applied"] = False
        note["why"] = "no trace or no answer"
        out["polish"] = note
        return out
    prompt = build_prompt(problem, old, style, closing_for(answer, dataset))
    async with sem:
        try:
            reply = (await generate_reasoning_trace(session, prompt, model) or "").strip()
        except Exception as exc:
            note.update({"applied": False, "why": f"error: {str(exc)[:120]}"})
            out["polish"] = note
            return out
    reply = reply.strip().strip("`").strip()
    why = accept(reply, old, dataset)
    if why is not None:
        # One more attempt, told what was wrong with the first: most refusals
        # are a citation to the wrong step or a stray piece of markup.
        fix_prompt = (prompt + "\n\nYour previous rewrite was rejected because: "
                      + why + ". Rewrite the trace again and avoid this.")
        async with sem:
            try:
                second = (await generate_reasoning_trace(session, fix_prompt, model) or "").strip()
                second = second.strip("`").strip()
                if accept(second, old, dataset) is None:
                    reply, why = second, None
                    note["retried"] = True
            except Exception:
                pass
    note["applied"] = why is None
    if why:
        note["why"] = why
    else:
        note["before"] = old
        out["synthesized_trace"] = reply
    out["polish"] = note
    return out


async def main_async(args) -> None:
    raw = json.loads(Path(_cfg.resolve_input(args.synth)).read_text(encoding="utf-8"))
    rows = raw.get("results", raw) if isinstance(raw, dict) else raw
    problems: Dict[str, str] = {}
    if args.k_traces:
        kraw = json.loads(Path(_cfg.resolve_input(args.k_traces)).read_text(encoding="utf-8"))
        krows = kraw.get("results", kraw) if isinstance(kraw, dict) else kraw
        problems = {r["sample_id"]: r.get("problem_text") or "" for r in krows}
    for r in rows:
        problems.setdefault(r["sample_id"], r.get("problem_text") or r.get("problem_input") or "")
    sem = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
        done = await asyncio.gather(*[
            polish_one(session, sem, r, problems.get(r["sample_id"], ""),
                       args.dataset, args.style, args.model) for r in rows])
    n_ok = sum(1 for r in done if r["polish"].get("applied"))
    whys: Dict[str, int] = {}
    for r in done:
        if not r["polish"].get("applied"):
            k = (r["polish"].get("why") or "?").split(":")[0].split(" ->")[0]
            whys[k] = whys.get(k, 0) + 1
    print(f"  {len(done)} samples, style={args.style}: rewritten {n_ok}, kept {len(done) - n_ok} {whys}")
    out = Path(_cfg.resolve_output(args.output))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(done, indent=1), encoding="utf-8")
    print(f"  Saved: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth", required=True)
    ap.add_argument("--k_traces", default=None,
                    help="Where the problem text lives; records may carry problem_text themselves")
    ap.add_argument("--dataset", required=True, choices=["ProofWriter", "FLD", "OmniMATH", "OlympiadBench"])
    ap.add_argument("--style", required=True, choices=sorted(STYLES),
                    help="Which rewrite to ask for; the styles in STYLES are the ones kept")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--base_url", default=None)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
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
