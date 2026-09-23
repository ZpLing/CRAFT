#!/usr/bin/env python3
"""What folding an adjudication back into a trace must and must not do.

The published gemini mathematics runs pasted the adjudicator's worked
solution under a "[Re-derivation]" header, so 86 OmniMATH and 60
OlympiadBench traces argued for one answer in their steps and stated another
in an appendix. The rewrite has to be steps, end on the picked answer and
box it once; anything else keeps the appended form so the answer read back
off the trace is the pass's answer either way.

    python test_apply_adjudication.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from framework.domain_optimization import apply_adjudication as aa  # noqa: E402

DERIVATION = ("### Solve\n- The rate is $\\frac{4}{7}$ per day.\n- Over 21 days: $12$ holes.\n"
              "The earlier attempts said 26 and 12. **Answer:** $\\boxed{12}$")
GOOD = ("Step 1: Pearl digs $4$ holes in $7$ days, so her rate is $\\frac{4}{7}$ holes per day.\n"
        "Step 2: Over $21$ days she digs $\\frac{4}{7}\\cdot 21 = 12$ holes, so the answer is $\\boxed{12}$.")
n_fail = 0


def check(name, got, want):
    global n_fail
    ok = got == want
    n_fail += not ok
    print(f"{'ok ' if ok else 'FAIL'} {name}" + ("" if ok else f"\n     got  {got!r}\n     want {want!r}"))


def run(reply, then=None):
    replies = [reply] + ([then] if then is not None else [])
    async def fake(session, prompt, model):
        return replies.pop(0) if len(replies) > 1 else replies[0]
    real = aa.generate_reasoning_trace
    aa.generate_reasoning_trace = fake
    async def go():  # the semaphore has to be born inside the loop on Python 3.9
        return await aa.integrate(None, asyncio.Semaphore(1), "problem", DERIVATION, "12", "m", "OmniMATH")
    try:
        return asyncio.run(go())
    finally:
        aa.generate_reasoning_trace = real


RECIPE = ("Step 1: Calculate the rate: $\\frac{4}{7}$ per day.\nStep 2: Determine the total over $21$ days: $12$.\n"
          "Step 3: Note that this is an integer.\nStep 4: Therefore, over $21$ days Pearl digs $\\boxed{12}$ holes.")
check("gate: refuses a recipe", aa.accept(RECIPE, "12", "OmniMATH"), "written as instructions")
check("gate: a 'Let' opening is not a recipe", aa.accept(GOOD.replace("Step 1: Pearl", "Step 1: Let $r$ be the rate; Pearl"), "12", "OmniMATH"), None)
check("prompt names the answer", "\\boxed{12}" in aa.build_integrate_prompt("p", DERIVATION, "12"), True)
text, why = run(GOOD)
check("gate: accepted", why, None)
check("gate: steps kept", text.count("Step "), 2)
check("gate: refuses prose", run("The answer is $\\boxed{12}$.")[1], "no steps")
check("gate: refuses the wrong answer", run(GOOD.replace("\\boxed{12}", "\\boxed{26}"))[1], "answer '26' != '12'")
check("gate: refuses a bare box", run(GOOD.replace("so the answer is $\\boxed{12}$.", "holes.\nStep 3: \\boxed{12}"))[1], "written as instructions")
check("gate: refuses an instruction", run(GOOD.replace("so the answer is $\\boxed{12}$.", "holes.\nStep 3: State the final count as \\boxed{12}."))[1], "written as instructions")
check("gate: refuses two boxes", run(GOOD.replace("Step 1: Pearl", "Step 1: $\\boxed{4}$ Pearl"))[1], "boxed more than once")
check("retry: the second try can pass", run(RECIPE, GOOD)[1], None)
check("retry: two failures keep the last reason", run("prose only")[1], "no steps")
print("\nall passed" if not n_fail else f"\n{n_fail} FAILED")
sys.exit(1 if n_fail else 0)
