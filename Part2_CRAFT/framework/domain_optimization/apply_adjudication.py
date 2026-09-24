#!/usr/bin/env python3
"""Fold an adjudication pass back into the synthesized traces it re-answered.

The pass runs over k_traces and records which answer a fresh derivation landed
on. This writes that back into the synthesis output so the result is scored by
the same loader and the same metric as every other cell, rather than by a
count written for the occasion — two numbers computed different ways are not
comparable even when both are right.

An answer is taken only where the pass ran, landed on one of the candidates,
and the consensus behind the original answer was no stronger than --max_votes.
That threshold is chosen on a validation split, not here.
"""
from __future__ import annotations
import argparse, asyncio, json, re, sys
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as _cfg
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation" / "label_prediction"))
from answer_match import answers_match  # noqa: E402
from extract_label import extract_math_answer  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "module3_topology_guided_synthesis"))
from synthesize_trace import generate_reasoning_trace  # noqa: E402

_STEP_HEAD = re.compile(r"(?m)^\s*\**\s*Step\s*(\d+)\s*\**\s*[:.\-]\s*")


def build_integrate_prompt(problem: str, derivation: str, answer: str) -> str:
    """Write the adjudicator's fresh derivation as the steps of a trace.

    The pass's reply is a worked solution in the model's own layout --
    markdown headings, bullets, a boxed answer somewhere in the middle -- and
    pasted under a header it left a trace whose steps reached one answer and
    whose appendix reached another. The derivation is what the new answer was
    accepted on; this asks for the same derivation as the trace's own steps,
    ending on that answer and nothing else.
    """
    return (
        "Write the following solution as the steps of a reasoning trace.\n\n"
        f"Problem:\n{problem}\n\n"
        f"Solution to write up:\n{derivation}\n\n"
        "Rules:\n"
        "- One line per step, 'Step N: ...', numbered from 1, each a complete "
        "sentence or two that states what is used and what follows; keep every "
        "computation the solution needs and none it does not.\n"
        "- Write in the indicative, as a record of reasoning already done "
        "('Using ..., we obtain ...'; 'Since ..., it follows that ...'), never as "
        "instructions to a reader ('Calculate ...', 'Conclude that ...').\n"
        "- Mathematics in LaTeX inside $...$, attached to a sentence; no headings, "
        "no bullet points, no commentary about the attempts that disagreed.\n"
        "- The last step is a declarative sentence that names the quantity the "
        f"problem asks for and ends with the answer, written once, as \\boxed{{{answer}}}; "
        "it is not a bare box.\n"
        "Output the steps only."
    )


_IMPERATIVE = re.compile(
    r"(?i)^(?:state|write|output|give|report|box|put|express|conclude|calculate|"
    r"compute|determine|identify|observe|define|note|show|find|verify|check|"
    r"confirm|apply|use|consider|evaluate|simplify|solve|substitute|recall|assume)\b")


def steps_of(reply: str) -> list:
    return [s.strip() for s in _STEP_HEAD.split(reply.strip())[1:]]


def written_as_instructions(reply: str) -> bool:
    """Whether the steps read as a recipe rather than as reasoning.

    The pass's derivation is laid out as a procedure ("Calculate the rate",
    "Conclude that ...") and a transcription keeps that voice, while the
    traces around it say what was used and what followed. A trace is refused
    when more than a quarter of its steps, or its last one, open on an
    instruction. The last one also has to say something besides the box: a
    step that is only the box ends the trace on a fragment, which is also what
    the grammar and informativeness scorers read last.
    """
    steps = steps_of(reply)
    if not steps:
        return True
    if sum(1 for s in steps if _IMPERATIVE.match(s)) * 4 > len(steps):
        return True
    last = steps[-1]
    if _IMPERATIVE.match(last):
        return True
    words = re.sub(r"\\boxed\{.*\}", " ", last).replace("$", " ").split()
    return len(words) < 4


def accept(reply: str, answer: str, dataset: str):
    """None when the rewrite is steps ending on the picked answer, else why not."""
    if len(_STEP_HEAD.findall(reply)) < 2:
        return "no steps"
    got = extract_math_answer(reply) or ""
    if not got or not answers_match(got, answer, dataset):
        return f"answer {got!r} != {answer!r}"
    if reply.count("\\boxed") != 1:
        return "boxed more than once"
    if written_as_instructions(reply):
        return "written as instructions"
    return None


async def integrate(session, sem, problem: str, derivation: str, answer: str,
                    model: str, dataset: str):
    """(trace, None) when a rewrite passes accept(), else (None, why).

    A rewrite that fails the gate gets one more try that says what was wrong
    with the first; the model re-solves the problem or falls back into the
    derivation's procedural voice often enough that the second try recovers a
    good share of them.
    """
    prompt = build_integrate_prompt(problem, derivation, answer)
    why = None
    for attempt in range(2):
        ask = prompt if attempt == 0 else (
            prompt + f"\n\nA previous write-up was rejected: {why}. Write up the given "
            f"solution exactly, do not solve the problem anew, and end on \\boxed{{{answer}}}.")
        async with sem:
            try:
                reply = (await generate_reasoning_trace(session, ask, model) or "").strip()
            except Exception as exc:
                return None, f"error: {str(exc)[:120]}"
        # One step per line, as the synthesizer writes them; the model likes
        # to leave a blank line between steps.
        reply = re.sub(r"\n\s*\n+", "\n", reply.strip("`").strip())
        why = accept(reply, answer, dataset)
        if why is None:
            return reply, None
    return None, why


def load(p):
    raw = json.loads(Path(p).read_text(encoding="utf-8"))
    return raw.get("results", raw) if isinstance(raw, dict) else raw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth", required=True)
    ap.add_argument("--adjudicated", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max_votes", type=int, default=3,
                    help="Only override where the consensus had at most this "
                         "many of the k votes behind it")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default=None,
                    help="Model that writes the derivation as steps (default: the "
                         "adjudicated file's model, else gemini-3.1-flash-lite)")
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--base_url", default=None)
    ap.add_argument("--concurrency", type=int, default=10)
    args = ap.parse_args()

    syn = load(_cfg.resolve_input(args.synth))
    adj = {r["sample_id"]: r for r in load(_cfg.resolve_input(args.adjudicated))}

    # Which samples the pass re-answers, by the same rule as before.
    overrides = {}
    for r in syn:
        a = (adj.get(r["sample_id"]) or {}).get("adjudication") or {}
        src = adj.get(r["sample_id"]) or {}
        if a.get("ran") and a.get("picked"):
            labs = [t.get("label") for t in (src.get("traces") or []) if t.get("label")]
            top = sum(1 for l in labs if answers_match(l, a["options"][0], args.dataset))
            if top <= args.max_votes and not answers_match(
                    a["picked"], a["options"][0], args.dataset):
                overrides[r["sample_id"]] = (a, src)

    written = {}
    if overrides:
        import synthesize_trace as _st
        if args.api_key:
            _st.OPENAI_API_KEY = args.api_key
            _st.HEADERS = {**_st.HEADERS,
                           "Authorization": _st._build_auth_header(
                               args.api_key, args.base_url or _st.OPENAI_BASE_URL)}
        if args.base_url:
            _st.OPENAI_BASE_URL = args.base_url
            _st.CHAT_COMPLETIONS_URL = args.base_url.rstrip("/") + "/chat/completions"
        model = args.model or "gemini-3.1-flash-lite"

        async def run_all():
            sem = asyncio.Semaphore(args.concurrency)
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
                jobs = {sid: integrate(session, sem, src.get("problem_text") or "",
                                       a.get("text", ""), a["picked"], model, args.dataset)
                        for sid, (a, src) in overrides.items()}
                results = await asyncio.gather(*jobs.values())
                return dict(zip(jobs.keys(), results))
        written = asyncio.run(run_all())

    changed = 0
    integrated = 0
    out = []
    for r in syn:
        r = dict(r)
        if r["sample_id"] in overrides:
            a, _src = overrides[r["sample_id"]]
            text, why = written.get(r["sample_id"], (None, "off"))
            note = {"picked": a["picked"], "consensus": a["options"][0], "integrated": text is not None}
            if why:
                note["integrate_rejected"] = why
            if text is not None:
                # The derivation the new answer was accepted on becomes the
                # trace; the steps that reached the old answer go, since a
                # trace that argues for one answer and states another is worse
                # than either alone.
                r["synthesized_trace"] = text
                integrated += 1
            else:
                r["synthesized_trace"] = ((r.get("synthesized_trace") or "").rstrip()
                                          + "\n\n[Re-derivation]\n" + a.get("text", "").strip())
            r["adjudication_applied"] = note
            changed += 1
        out.append(r)

    p = Path(_cfg.resolve_output(args.output))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out), encoding="utf-8")
    print(f"  {len(out)} samples, {changed} answers replaced "
          f"({integrated} written as steps, {changed - integrated} appended) -> {p}")


if __name__ == "__main__":
    main()
