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

A third pass, `resolve`, is keyed to the reasoning rather than to the label.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
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


def justified_by_absence(text: str) -> bool:
    """Does the trace end by reporting a failed search instead of a derivation?"""
    lines = [l for l in (text or "").splitlines() if l.strip()]
    return bool(_BY_ABSENCE.search(" ".join(lines[-2:])))


def build_audit_prompt(problem: str, claimed: str) -> str:
    """The mirror pass: check a claimed derivation instead of searching for one.

    Where a model over-produces __DISPROVED__ it is failing to search; where it
    over-produces __PROVED__ it is accepting a chain that does not hold, and on
    FLD gemini does the second — 33 of its 52 errors are a hypothesis that does
    not follow, reported as proved.

    It does not work, and the measurement is the point of keeping it. On those
    500 FLD samples it flips 60 of the 261 answered __PROVED__ and 47 of the 60
    are wrong: 22% precision, against 92% for the search direction on
    ProofWriter. Gating it on a split consensus does not rescue it — 38%
    precision on the 26 non-unanimous flips, still net negative — and the
    unanimous stratum is 9%. Asked to check a chain the model finds a fault
    whether or not one is there.

    So the two directions are not symmetric, and the pipeline only runs the one
    that is safe. That asymmetry is also why the search direction can be trusted
    at all: it accepts a flip only on positive evidence, a chain the reply
    exhibits, and never on a claim that no chain exists.
    """
    return (
        "Check one derivation, step by step. Do not write your own.\n\n"
        f"{problem}\n\n"
        "Claimed derivation:\n"
        f"{claimed}\n\n"
        "Go through it one step at a time. For each step, say which premises it "
        "uses and whether the step follows from exactly those premises. Watch for "
        "the three ways these derivations fail: a step that uses something never "
        "stated, a step that reverses a conditional (concluding A from B and "
        "'if A then B'), and a final step whose conclusion is not the hypothesis "
        "as written.\n\n"
        "Then finish in exactly one of these two ways:\n"
        "  - If every step holds and the chain reaches the hypothesis: __PROVED__\n"
        "  - If some step does not follow: name that step, say which premise it "
        "misuses, then __DISPROVED__\n\n"
        "Do not write __DISPROVED__ unless you can name the step that fails."
    )


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

    The single-direction passes above each start from a label and ask about
    that label, which only works where the errors sit on one side. They do on
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


def build_direct_prompt(problem: str) -> str:
    """One premise at a time, against a hypothesis that follows from one.

    FLD's proofs are short -- `sent4 -> hypothesis` is a third of them, and
    another quarter are two premises -- and its traces are not: the ones that
    get it wrong build a conditional proof or a case analysis and wander. One
    of them derived the right conditional, said so, and then wrote the other
    label. Accuracy on a hypothesis that is itself a negation is 77-82%,
    against 82-95% on a positive one, and the gap is widest exactly where the
    gold proof is one step.

    So this pass asks for the short read instead of a better long one, and
    names the premises rather than the rounds. One premise is a whole proof
    here, so unlike the other directions it accepts a single citation.

    It does not work either, and that is the fourth reading of the same result.
    On 100 of nano's FLD samples it commits on 28 and changes 4, two of them to
    the right answer and two to the wrong one. `resolve` on FLD's by-absence
    stratum is 6 right against 4 wrong, `prove` over everything answered
    __DISPROVED__ is 5 against 8, and `audit` is 13 against 47. Four passes,
    every one of them at or below a coin flip, while the same gate on
    ProofWriter accepted 90 flips and got 90 of them right.

    The gate is what differs, not the model. ProofWriter's rules are Horn
    clauses over concrete relations, so a chain the reply writes out can be
    read and is either there or not. FLD's hypotheses are themselves negations,
    conjunctions and conditionals, and whether a premise entails one is a
    proof-theoretic question that citing the premise does not answer. Asking
    for a citation therefore filters nothing here, and what gets through is
    whatever the model happened to say. Deciding FLD needs a checker that can
    decide entailment, not a second opinion from the same model.
    """
    return (
        "You are checking one thing about a formal-logic problem.\n\n"
        f"{problem}\n\n"
        "An earlier attempt built a long derivation and may have wandered off. "
        "A hypothesis that follows here follows in a few steps, most often from "
        "a single fact read directly against it.\n\n"
        "So go fact by fact, and for each one ask only:\n"
        "  - does this fact on its own entail the hypothesis as written?\n"
        "  - does it entail the negation of the hypothesis?\n"
        "Then try the pairs of facts that share a term. Do not build a "
        "conditional proof, do not argue by cases, and do not assume anything "
        "the facts do not state.\n\n"
        "Watch the polarity. The hypothesis may itself be a negation, a "
        "conjunction or a conditional, and entailing \"not P\" is not the same "
        "as failing to entail \"P\". A conjunction needs every part; a "
        "conditional is about what follows from its antecedent, not about "
        "whether the antecedent holds.\n\n"
        "Finish in exactly one of these three ways:\n"
        "  - a fact or pair entails the hypothesis: name them, then "
        "REACHED: HYPOTHESIS and __PROVED__\n"
        "  - a fact or pair entails its negation: name them, then "
        "REACHED: NEGATION and __DISPROVED__\n"
        "  - neither does: write REACHED: NEITHER and nothing after it\n\n"
        "Name the facts you used. Do not write a label without them."
    )


async def recheck_one(session, sem, rec, problem, depth, model,
                      direction="prove") -> Dict[str, Any]:
    claimed = rec.get("synthesized_trace") or ""
    prompt = {"prove": lambda: build_prompt(problem, depth),
              "audit": lambda: build_audit_prompt(problem, claimed),
              "resolve": lambda: build_resolve_prompt(problem, depth),
              "direct": lambda: build_direct_prompt(problem)}[direction]()
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

    if direction in ("resolve", "direct"):
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
        # A single premise is a whole proof in FLD, so the direct pass takes
        # one citation where the saturating one needs two.
        floor = 1 if direction == "direct" else MIN_CITATIONS
        accepted = (want is not None and label in (None, want) and cites >= floor)
        before = last_label(claimed)
        flipped = bool(accepted and want != before)
        note.update({"reached": reached, "accepted": accepted,
                     "before": before, "flipped": flipped})
        header = ("[Premise-by-premise check]" if direction == "direct"
                  else "[Two-sided proof search]")
    else:
        want = "__PROVED__" if direction == "prove" else "__DISPROVED__"
        flipped = label == want and cites >= MIN_CITATIONS
        note["flipped"] = flipped
        header = ("[Directed proof search]" if direction == "prove"
                  else "[Derivation audit]")

    out = dict(rec)
    out["cwa_recheck"] = note
    if flipped:
        body = (reply or "").strip()
        if label != want:
            # The answer is read back off the trace, so a reply that stated its
            # result only as a marker has to leave the label behind in writing.
            body += f"\n\n{want}"
        out["synthesized_trace"] = claimed.rstrip() + f"\n\n{header}\n" + body
    return out


async def main_async(args) -> None:
    synth_path = Path(_cfg.resolve_input(args.synth))
    raw = json.loads(synth_path.read_text(encoding="utf-8"))
    rows = raw.get("results", raw) if isinstance(raw, dict) else raw

    kraw = json.loads(Path(_cfg.resolve_input(args.k_traces)).read_text(encoding="utf-8"))
    krows = kraw.get("results", kraw) if isinstance(kraw, dict) else kraw
    problems = {r["sample_id"]: r.get("problem_text") or "" for r in krows}

    if args.direction == "direct":
        targets = list(rows)
        side = "every sample"
    elif args.direction == "resolve":
        # Keyed to how the trace ends, not to the label it ends on.
        targets = [r for r in rows
                   if justified_by_absence(r.get("synthesized_trace") or "")]
        side = "by absence"
    else:
        side = "__DISPROVED__" if args.direction == "prove" else "__PROVED__"
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
    if args.direction in ("resolve", "direct"):
        acc = sum(1 for r in done if r["cwa_recheck"].get("accepted"))
        nei = sum(1 for r in done if r["cwa_recheck"].get("reached") == "NEITHER")
        print(f"  closed a chain: {acc}   left open: {nei}   "
              f"answers changed: {n_flip}")
    else:
        want = "__PROVED__" if args.direction == "prove" else "__DISPROVED__"
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
    ap.add_argument("--direction", choices=["prove", "audit", "resolve", "direct"], default="prove",
                    help="Which side the errors sit on. 'prove' searches again "
                         "for a derivation on the samples answered __DISPROVED__; "
                         "'audit' checks the claimed derivation on the samples "
                         "answered __PROVED__; 'resolve' takes the samples whose "
                         "final step reasons from the absence of a derivation, "
                         "whichever label that produced, and searches both "
                         "directions. Chosen per configuration from where that "
                         "configuration's errors actually are")
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
