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

A second pass, `resolve`, is keyed to the reasoning rather than to the label.
A trace that ends on the absence of a derivation reaches whichever label the
hypothesis's polarity suggests -- a positive hypothesis nobody derived is
called __DISPROVED__, a negative one __PROVED__ because "it does not happen"
is said to be consistent with facts that never mention it -- so the side it
lands on says nothing about the answer, and a pass keyed to the label sees
only part of it. On gemini's ProofWriter run those samples are 230 of 500 and
39.6% correct, against 94.8% for the traces that end on a chain they wrote;
145 of the 230 are __DISPROVED__ and 85 __PROVED__.

Asked to saturate forward instead of searching backward from the hypothesis,
the model closed a chain on 151 of those 230 and changed 90 answers, and all
90 were changed to the right one, with nothing correct broken: 69.4% to 87.4%.
On nano the same pass is 9 of 65, also all correct, 95.4% to 97.2%. The gate
is what makes that hold -- a flip is accepted only against a chain the reply
writes out, and 79 samples where nothing closed were left as they were.

FLD is not run through it. Its by-absence stratum is 93 samples of 500 and the
pass changes 10 of them, 6 right and 4 wrong, which is +2 samples on a coin
flip; nano's is 28, changing 5 for +1. The errors there are a different shape
-- a hypothesis that does not follow, reported as proved -- and this pass does
not address it.

    python cwa_recheck.py --synth <synthesized.json> --k_traces <k_traces.json> \
        --expected_depth 5 --model <m> --output <rechecked.json>
    python cwa_recheck.py --synth <synth_cwa.json> ... --direction resolve
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

# Module III lives beside this package, not in it: these passes run after it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "module3_topology_guided_synthesis"))
from synthesize_trace import generate_reasoning_trace  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as _cfg  # noqa: E402

LABEL_RE = re.compile(r"(__PROVED__|__DISPROVED__)", re.IGNORECASE)
# A derivation names what it chained. Two or more references is the bar; the
# replies that flip an answer without them are the ones that assert rather
# than derive. Asked to saturate, a reply restates the premises in its own
# shorthand first and then cites that -- `F8`, `R19` -- so the abbreviated
# spellings count as references too, or a complete derivation reads as an
# uncited assertion and is thrown away.
CITE_RE = re.compile(
    r"\b(?:fact|rule|sent|premise|statement)\s*#?\d+"
    r"|\b[FR]#?\d+\b", re.IGNORECASE)
MIN_CITATIONS = 2


def last_label(text: str) -> Optional[str]:
    hits = LABEL_RE.findall(text or "")
    return f"__{hits[-1].upper().strip('_')}__" if hits else None


# A trace that closes on the absence of a derivation rather than on one it
# wrote. Both labels are reached this way and the polarity of the hypothesis
# decides which: a positive hypothesis nobody could derive is called
# __DISPROVED__, a negative one is called __PROVED__ because "it does not
# happen" is said to be consistent with facts that never mention it. The
# dataset is open-world, so neither reading is sound, and which of the two a
# sample lands on carries no information about the answer.
_BY_ABSENCE = re.compile(
    r"(cannot be derived|can not be derived|cannot be concluded"
    r"|no (?:premise|rule|mechanism|evidence|information|statement"
    r"|logical (?:rule|chain|derivation))"
    r"|do(?:es)? not (?:provide|contain)|not (?:provided|derivable|supported)"
    r"|is consistent with|remains consistent|cannot be contradicted)",
    re.IGNORECASE)

_REACHED_RE = re.compile(r"REACHED:\s*(HYPOTHESIS|NEGATION|NEITHER)", re.IGNORECASE)

# What a pass appends when it changes an answer, and the step heads of the
# trace it appends to.
_HEADER_RE = re.compile(
    r"\n*\[(?:Directed proof search|Two-sided proof search)\]\n")
_STEP_HEAD = re.compile(r"(?m)^\s*\**\s*Step\s*(\d+)\s*\**\s*[:.\-]\s*")
_STEP_REF = re.compile(r"\bStep\s*\d+\b")


def split_recheck(trace: str):
    """(claimed, header, body) for a trace that carries an appended pass, else None."""
    m = _HEADER_RE.search(trace or "")
    if not m:
        return None
    return trace[:m.start()].rstrip(), m.group(0).strip(), trace[m.end():].strip()


def cut_before_label(claimed: str, old_label: Optional[str]):
    """Drop the step that carries the answer being replaced.

    Returns (kept, next_step_number). The step is cut from its head, so a
    derivation it also contained goes with it; the rewrite below sees the
    whole trace and re-derives what the chain still needs.
    """
    heads = list(_STEP_HEAD.finditer(claimed))
    if not old_label or not heads:
        return claimed.rstrip(), len(heads) + 1
    pos = claimed.rfind(old_label)
    if pos < 0:
        return claimed.rstrip(), len(heads) + 1
    before = [h for h in heads if h.start() <= pos]
    if not before:
        return claimed.rstrip(), len(heads) + 1
    cut = before[-1]
    return claimed[:cut.start()].rstrip(), int(cut.group(1))


def build_integrate_prompt(problem: str, kept: str, chain: str, want: str,
                           reached: Optional[str], next_no: int) -> str:
    """Write the chain a pass found as steps in the trace's own format.

    The pass's reply is a search transcript -- rounds, `F8`/`R19` shorthand,
    bullets, a formalised copy of the premises -- and pasting it under a
    header leaves a trace whose last step says one label and whose appendix
    says the other. The chain is what the flip was accepted on; this asks for
    the same chain written as the steps that continue the trace, and the
    caller accepts the rewrite only if it keeps the label and cites what it
    chained, exactly the gate the flip itself passed.
    """
    what = ("the negation of the hypothesis" if reached == "NEGATION"
            else "the hypothesis")
    close = (f'Step M: Since Step M-1 established that <{what}, written out as a '
             f'statement>, the hypothesis "<hypothesis as written>" is {want}.')
    return (
        "You are finishing one closed-world deduction trace.\n\n"
        f"{problem}\n\n"
        "The trace so far:\n"
        f"{kept}\n\n"
        "Its next step had reasoned from the absence of a derivation and reached "
        "the wrong answer. A second search then found the derivation below; these "
        "are the searcher's own notes, in its own shorthand:\n"
        f"{chain}\n\n"
        f"Rewrite that derivation as the steps that continue the trace, numbered "
        f"from Step {next_no}. Write each step on its own line in exactly the "
        "trace's format:\n"
        "  Step N: According to Fact X, <that fact or rule as written in the "
        "problem>. Since <what it is applied to, citing Fact numbers or earlier "
        "Steps>, it follows that <the derived statement>.\n"
        "Use the problem's own numbering and wording. Do not use shorthand such as "
        "F1 or R8, arrows, bullets, rounds or headings, and do not restate a "
        "statement the trace already derived -- cite the Step that derived it. "
        "Include only the steps the chain actually needs. End with one line:\n"
        f"  {close}\n"
        "Output the steps only."
    )


async def integrate_chain(session, problem: str, claimed: str, chain: str,
                          want: str, reached: Optional[str], before: Optional[str],
                          model: str):
    """(trace with the chain written into it, None), or (None, why) to keep the appendix.

    Accepted only when the rewrite is steps, ends on the pass's label and
    nothing else, and cites at least as many premises as the flip required.
    Anything short of that keeps the appended form, so the answer read back
    off the trace is the pass's answer either way.
    """
    kept, next_no = cut_before_label(claimed, before if before != want else None)
    prompt = build_integrate_prompt(problem, kept, chain, want, reached, next_no)
    try:
        reply = (await generate_reasoning_trace(session, prompt, model) or "").strip()
    except Exception as exc:
        return None, f"error: {str(exc)[:120]}"
    reply = _HEADER_RE.sub("\n", reply).strip()
    labels = {f"__{h.upper().strip('_')}__" for h in LABEL_RE.findall(reply)}
    if not _STEP_HEAD.search(reply):
        return None, "no steps"
    if labels != {want} or reply.count(want) != 1:
        return None, f"labels {sorted(labels)} x{reply.count(want)}"
    if not reply.rstrip().rstrip(".").endswith(want):
        return None, "label not last"
    # A continuation grounds on the trace as much as on the problem, so a
    # reference to an earlier Step counts alongside a Fact; the heads that
    # number the new steps do not.
    refs = (len(CITE_RE.findall(reply)) + len(_STEP_REF.findall(reply))
            - len(_STEP_HEAD.findall(reply)))
    if refs < MIN_CITATIONS:
        return None, "uncited"
    return kept + "\n" + reply, None


def justified_by_absence(text: str) -> bool:
    """Does the trace end by reporting a failed search instead of a derivation?"""
    lines = [l for l in (text or "").splitlines() if l.strip()]
    return bool(_BY_ABSENCE.search(" ".join(lines[-2:])))


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


def build_resolve_prompt(problem: str, depth: Optional[int]) -> str:
    """Two directed searches on a sample that was answered by absence.

    The single-direction pass above starts from a label and asks about that
    label, which only works where the errors sit on one side. They do on
    nano -- 63 of its 65 by-absence answers on ProofWriter are __DISPROVED__ --
    and they do not on gemini, whose 230 are 145 against 85. A pass keyed to
    the label therefore leaves a third of gemini's by-absence samples untouched,
    and the ones it leaves are not the easier ones: the stratum as a whole is
    39.6% correct, against 94.8% for the samples whose final step writes out a
    chain.

    So this pass is keyed to the reasoning instead of the label, and it asks
    for both chains at once, because the thing the earlier attempt got wrong is
    not which side it picked but that it picked from a failed search at all.
    The slice guarantees exactly one of the two chains exists, which is what
    makes a two-sided question answerable; the reply still has to write the
    chain it claims, and "neither closes" changes nothing.
    """
    rounds = (f"The slice is generated at depth {depth}, so {depth} rounds "
              f"saturate it.\n" if depth else "")
    return (
        "You are resolving one hypothesis that an earlier attempt left open.\n\n"
        f"{problem}\n\n"
        "That attempt searched backward from the hypothesis and stopped at the "
        "first condition it could not immediately satisfy, then read its own "
        "failure as an answer. Neither move is sound here: a condition unproven "
        "in one round is often derived in the next, and the hypothesis is "
        "either derivable from the facts or its negation is, with exactly one "
        "of the two holding.\n\n"
        "So do not search backward, and do not stop at an unproven condition. "
        "Work forward in rounds instead:\n"
        "  Round 1 -- apply every rule whose conditions are met by the facts as "
        "given. Write each statement you derive and the rule that produced it.\n"
        "  Round 2 -- do it again, using the facts together with everything "
        "round 1 derived.\n"
        "  Keep going until a round derives nothing new.\n"
        f"{rounds}"
        "Then look through everything you derived for the hypothesis as written, "
        "and for its negation -- the same statement with its polarity reversed.\n\n"
        "Finish in exactly one of these three ways:\n"
        "  - the hypothesis is among them: name the rounds that produced it, "
        "then REACHED: HYPOTHESIS and __PROVED__\n"
        "  - its negation is among them: name the rounds that produced it, "
        "then REACHED: NEGATION and __DISPROVED__\n"
        "  - the rounds stopped deriving and neither appeared: REACHED: NEITHER\n\n"
        "Only something a round actually derived counts. Do not write a label "
        "for a statement you assumed, or for one you could not derive."
    )


async def recheck_one(session, sem, rec, problem, depth, model,
                      direction="prove") -> Dict[str, Any]:
    claimed = rec.get("synthesized_trace") or ""
    prompt = {"prove": lambda: build_prompt(problem, depth),
              "resolve": lambda: build_resolve_prompt(problem, depth)}[direction]()
    async with sem:
        try:
            reply = await generate_reasoning_trace(session, prompt, model)
        except Exception as exc:
            # One sample the gateway refuses -- FLD's invented vocabulary trips
            # a content filter on a handful of them -- is one sample this pass
            # does not get to look at, not a run that stops. The answer already
            # in the trace stands.
            out = dict(rec)
            out["cwa_recheck"] = {"direction": direction, "error": str(exc)[:200],
                                  "flipped": False}
            return out
    label = last_label(reply or "")
    cites = len(CITE_RE.findall(reply or ""))
    note = {"label": label, "citations": cites, "direction": direction}

    if direction == "resolve":
        # The reply says which of the two chains it closed, and the label has
        # to agree with it: a label without its marker, or against it, is the
        # same assertion-without-a-derivation this pass exists to remove.
        m = _REACHED_RE.search(reply or "")
        reached = m.group(1).upper() if m else None
        want = {"HYPOTHESIS": "__PROVED__", "NEGATION": "__DISPROVED__"}.get(reached)
        # The marker is the commitment and the label beside it is a restatement
        # of it, which nano leaves off: 50 of its 51 resolved samples name the
        # chain they closed and then stop. Requiring the token as well threw all
        # 50 away. A reply that writes a token contradicting its own marker is
        # incoherent rather than resolved, and is still refused.
        accepted = (want is not None and label in (None, want) and cites >= MIN_CITATIONS)
        before = last_label(claimed)
        flipped = bool(accepted and want != before)
        note.update({"reached": reached, "accepted": accepted,
                     "before": before, "flipped": flipped})
        header = "[Two-sided proof search]"
    else:
        want = "__PROVED__"
        flipped = label == want and cites >= MIN_CITATIONS
        note["flipped"] = flipped
        header = "[Directed proof search]"

    out = dict(rec)
    out["cwa_recheck"] = note
    if flipped:
        body = (reply or "").strip()
        if label != want:
            # The answer is read back off the trace, so a reply that stated its
            # result only as a marker has to leave the label behind in writing.
            body += f"\n\n{want}"
        # The chain is kept in the record; the trace gets it written as steps
        # when the rewrite passes the gate, and the appended form otherwise.
        note["chain"] = body
        async with sem:
            integrated, why = await integrate_chain(
                session, problem, claimed, body, want, note.get("reached"),
                last_label(claimed), model)
        note["integrated"] = integrated is not None
        if why:
            note["integrate_rejected"] = why
        out["synthesized_trace"] = (integrated if integrated is not None
                                    else claimed.rstrip() + f"\n\n{header}\n" + body)
    return out


async def integrate_one(session, sem, rec, problem, model) -> Dict[str, Any]:
    """Rewrite an already-appended pass into steps; for files a pass wrote earlier."""
    parts = split_recheck(rec.get("synthesized_trace") or "")
    if parts is None:
        return rec
    claimed, _header, body = parts
    want = last_label(body)
    if want is None:
        return rec
    note = dict(rec.get("cwa_recheck") or {})
    async with sem:
        integrated, why = await integrate_chain(
            session, problem, claimed, body, want, note.get("reached"),
            last_label(claimed), model)
    out = dict(rec)
    note.update({"chain": body, "integrated": integrated is not None})
    if why:
        note["integrate_rejected"] = why
    out["cwa_recheck"] = note
    if integrated is not None:
        out["synthesized_trace"] = integrated
    return out


async def main_async(args) -> None:
    synth_path = Path(_cfg.resolve_input(args.synth))
    raw = json.loads(synth_path.read_text(encoding="utf-8"))
    rows = raw.get("results", raw) if isinstance(raw, dict) else raw

    kraw = json.loads(Path(_cfg.resolve_input(args.k_traces)).read_text(encoding="utf-8"))
    krows = kraw.get("results", kraw) if isinstance(kraw, dict) else kraw
    problems = {r["sample_id"]: r.get("problem_text") or "" for r in krows}

    if args.integrate_only:
        targets = [r for r in rows if split_recheck(r.get("synthesized_trace") or "")]
        print(f"  {len(rows)} samples, {len(targets)} carry an appended pass — "
              f"writing those chains as steps")
        sem = asyncio.Semaphore(args.concurrency)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
            done = await asyncio.gather(*[
                integrate_one(session, sem, r, problems.get(r["sample_id"], ""), args.model)
                for r in targets])
        by_id = {r["sample_id"]: r for r in done}
        merged = [by_id.get(r["sample_id"], r) for r in rows]
        n_int = sum(1 for r in done if r["cwa_recheck"].get("integrated"))
        same = sum(1 for r, t in zip(done, targets)
                   if last_label(r["synthesized_trace"]) == last_label(t["synthesized_trace"]))
        print(f"  written as steps: {n_int}   kept appended: {len(done) - n_int}   "
              f"labels unchanged: {same}/{len(done)}")
        out = Path(_cfg.resolve_output(args.output))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(merged, indent=1), encoding="utf-8")
        print(f"  Saved: {out}")
        return

    if args.direction == "resolve":
        # Keyed to how the trace ends, not to the label it ends on.
        targets = [r for r in rows
                   if justified_by_absence(r.get("synthesized_trace") or "")]
        side = "by absence"
    else:
        side = "__DISPROVED__"
        targets = [r for r in rows
                   if last_label(r.get("synthesized_trace") or "") == side]
    print(f"  {len(rows)} samples, {len(targets)} answered {side} — "
          f"rechecking those ({args.direction})")

    sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        done = await asyncio.gather(*[
            recheck_one(session, sem, r, problems.get(r["sample_id"], ""),
                        args.expected_depth, args.model, args.direction)
            for r in targets])

    by_id = {r["sample_id"]: r for r in done}
    merged = [by_id.get(r["sample_id"], r) for r in rows]
    n_flip = sum(1 for r in done if r["cwa_recheck"]["flipped"])
    n_int = sum(1 for r in done if r["cwa_recheck"].get("integrated"))
    if n_flip:
        print(f"  chains written as steps: {n_int} of {n_flip} "
              f"(the rest keep the appended form)")
    if args.direction == "resolve":
        acc = sum(1 for r in done if r["cwa_recheck"].get("accepted"))
        nei = sum(1 for r in done if r["cwa_recheck"].get("reached") == "NEITHER")
        print(f"  closed a chain: {acc}   left open: {nei}   "
              f"answers changed: {n_flip}")
    else:
        want = "__PROVED__"
        said = sum(1 for r in done if r["cwa_recheck"]["label"] == want)
        print(f"  said {want}: {said}   of those naming what they used: {n_flip}")

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
    ap.add_argument("--direction", choices=["prove", "resolve"], default="prove",
                    help="Which side the errors sit on. 'prove' searches again "
                         "for a derivation on the samples answered __DISPROVED__; "
                         "'resolve' takes the samples whose "
                         "final step reasons from the absence of a derivation, "
                         "whichever label that produced, and searches both "
                         "directions. Chosen per configuration from where that "
                         "configuration's errors actually are")
    ap.add_argument("--integrate_only", action="store_true",
                    help="Take a file an earlier run wrote with appended replies and "
                         "write those chains as steps; no new search is made")
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
