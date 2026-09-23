#!/usr/bin/env python3
"""What the recheck must do with the chain it found, on the shapes the passes wrote.

Each case is one the appended form got wrong on a real ProofWriter trace: a
`resolve` pass pasted its round-by-round transcript ("F1: Cow eats lion.",
"Round 1", bullets) under a header, so the trace said __DISPROVED__ in its
last step and __PROVED__ in an appendix; 9 of gemini's 50 traces and 101 of
its 500 carried one. The rewrite has to keep the pass's answer as the only
answer, drop the step it replaces and refuse anything that is not steps.

    python test_cwa_recheck.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from framework.domain_optimization import cwa_recheck as cr  # noqa: E402

CLAIMED = (
    "Step 1: According to Fact 16, if someone likes the cow then they are nice. "
    "Since Fact 8 states that the rabbit likes the cow, it follows that the rabbit is nice.\n"
    "Step 2: According to Fact 18, if someone is nice then they visit the cow. "
    "Since Step 1 established that the rabbit is nice, it follows that the rabbit visits the cow.\n"
    "Step 3: No fact or inference establishes that the cow is round. Since the conclusion "
    "cannot be derived from the provided facts, the hypothesis is __DISPROVED__.")
CHAIN = (
    "To determine if the cow is round, we proceed forward. Facts:\n"
    "F1: Cow eats lion. F8: Rabbit likes cow. Round 1:\n"
    "*   From F8 (Rabbit likes cow) and Rule 16: Rabbit is nice.\n"
    "*   From Rabbit is nice and Rule 18: Rabbit visits cow.\n"
    "Round 2:\n*   From Rabbit visits cow and Rule 13: Cow is round.\n"
    "REACHED: HYPOTHESIS and __PROVED__")
APPENDED = CLAIMED + "\n\n[Two-sided proof search]\n" + CHAIN

GOOD = (
    "Step 3: According to Fact 13, if someone visits the cow then the cow is round. "
    "Since Step 2 established that the rabbit visits the cow, it follows that the cow is round.\n"
    'Step 4: Since Step 3 established that the cow is round, the hypothesis "The cow is round" is __PROVED__.')

n_fail = 0


def check(name, got, want):
    global n_fail
    ok = got == want
    n_fail += not ok
    print(f"{'ok ' if ok else 'FAIL'} {name}" + ("" if ok else f"\n     got  {got!r}\n     want {want!r}"))


def run_integrate(reply):
    async def fake(session, prompt, model):
        return reply
    real = cr.generate_reasoning_trace
    cr.generate_reasoning_trace = fake
    try:
        return asyncio.run(cr.integrate_chain(
            None, "problem", CLAIMED, CHAIN, "__PROVED__", "HYPOTHESIS", "__DISPROVED__", "m"))
    finally:
        cr.generate_reasoning_trace = real


# The appended form splits into the trace, the header and the transcript.
claimed, header, body = cr.split_recheck(APPENDED)
check("split: claimed", claimed, CLAIMED)
check("split: header", header, "[Two-sided proof search]")
check("split: body", body, CHAIN)
check("split: none without a header", cr.split_recheck(CLAIMED), None)

# The step carrying the replaced answer goes, and numbering resumes at it.
kept, nxt = cr.cut_before_label(CLAIMED, "__DISPROVED__")
check("cut: kept", kept, CLAIMED.rsplit("\nStep 3", 1)[0])
check("cut: next number", nxt, 3)
check("cut: nothing when the label is absent", cr.cut_before_label(CLAIMED, "__PROVED__"), (CLAIMED, 4))

# A rewrite that is steps, cites premises and ends on the pass's answer is taken.
text, why = run_integrate(GOOD)
check("gate: accepted", why, None)
check("gate: replaced step gone", "cannot be derived" in (text or ""), False)
check("gate: one label", (text or "").count("__PROVED__"), 1)
check("gate: label last", (text or "").rstrip().rstrip(".").endswith("__PROVED__"), True)
check("gate: header not echoed", "[Two-sided" in (text or ""), False)
check("gate: numbering continues", "\nStep 3: According to Fact 13" in (text or ""), True)

# Refusals: the appended form stays and the record says why.
check("gate: refuses prose", run_integrate("The cow is round, so __PROVED__.")[1], "no steps")
check("gate: refuses both labels",
      run_integrate(GOOD.replace("Step 3:", "Step 3: Not __DISPROVED__;"))[1], "labels ['__DISPROVED__', '__PROVED__'] x1")
check("gate: refuses label twice", run_integrate(GOOD + " __PROVED__")[1], "labels ['__PROVED__'] x2")
check("gate: refuses label not last", run_integrate(GOOD + "\nStep 5: Done.")[1], "label not last")
check("gate: refuses uncited",
      run_integrate('Step 3: The cow is round.\nStep 4: Since Step 3 established that the cow is round, '
                    'the hypothesis is __PROVED__.')[1], "uncited")
check("gate: refuses the wrong answer", run_integrate(GOOD.replace("__PROVED__", "__DISPROVED__"))[1],
      "labels ['__DISPROVED__'] x0")

print("\nall passed" if not n_fail else f"\n{n_fail} FAILED")
sys.exit(1 if n_fail else 0)
