#!/usr/bin/env python3
"""What math_text must find, rewrite and keep whole, from traces it got wrong.

Each case is one the pipeline mishandled on a real trace before the rule
existed: the display block gpt-5.4-nano writes on the lines after "Step N:"
was dropped by a splitter that kept header lines only; a LaTeX line spacing
\\\\[6pt] was rewritten as a display delimiter; an unclosed $$ passed a
delimiter check that only counted dollars.

    python test_math_text.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from framework.domain_optimization.math_text import (  # noqa: E402
    balanced, find_math, normalize_math, split_steps)

# gpt-5.4-nano: header line, then the mathematics on lines of its own, then a
# trailer and the answer. Only the header used to survive.
GPT_MATH = """Step 1: Let the expression be a perfect cube. Set
\\[
n^{3}+2n^{2}+9n+8 = m^{3}
\\]
for some integer \\(m\\).
Step 2: Compare with nearby cubes:
\\[
(n+1)^3 = n^3+3n^2+3n+1.
\\]
So \\(m^3 - (n+1)^3 = -n^2+6n+7\\).
Final Answer: \\boxed{7}"""

# A logical trace that closes with a summary paragraph: the summary is not a
# line of the last step.
LOGIC = """Step 1: From Fact2 and Fact15, the bald eagle is red.
Step 2: From Step 1 and Fact14, the bald eagle eats the bald eagle.
Summary: We derived that the bald eagle is red and eats itself.
Final Conclusion: __PROVED__"""

# A $$ block whose inner line begins with a word the splitter must not read as
# a step header, and an aligned environment inside a step.
DOLLAR_BLOCK = """Step 1: We have
$$
Step = 3 \\cdot x
$$
which fixes $x$.
Step 2: Then
\\begin{aligned} a &= b \\\\ c &= d \\end{aligned}
closes the case."""

# Free prose with no step headers at all: every line is kept.
PROSE = """We want the largest n.
Try n = 7: it works.
\\boxed{7}"""

# A $$ opened and never closed before the next step, whose own $$ would pair
# with it and hide the header inside a "display".
UNCLOSED = """Step 1: We have
$$
a = b
and stop here.
Step 2: Then
$$
c = d
$$
follows."""


def main() -> int:
    failures = 0

    steps = split_steps(GPT_MATH)
    if len(steps) != 2 or "m^{3}" not in steps[0] or "(n+1)^3" not in steps[1]:
        failures += 1
        print("FAIL  a display block on the lines after a header was not kept with its step")
    else:
        print("ok    keeps  the display block gpt-5.4-nano writes after the header")
    if any("Final Answer" in s for s in steps):
        failures += 1
        print("FAIL  the answer trailer was read as part of a step")
    else:
        print("ok    leaves the answer trailer out of the last step")

    steps = split_steps(LOGIC)
    if any("Summary:" in s for s in steps):
        failures += 1
        print("FAIL  the closing summary was attached to the last step")
    else:
        print("ok    leaves the closing summary out of the last step")
    if steps[-1] != "Final Conclusion: __PROVED__":
        failures += 1
        print("FAIL  the Final Conclusion line was not kept as its own step")
    else:
        print("ok    keeps  the Final Conclusion line for the RKG's conclusion node")

    steps = split_steps(DOLLAR_BLOCK)
    if len(steps) != 2 or steps[0].count("$$") != 2:
        failures += 1
        print("FAIL  a $$ block was cut at a line that starts with 'Step'")
    else:
        print("ok    keeps  a $$ block whole across its lines")
    if "\\begin{aligned}" not in steps[1] or "\\end{aligned}" not in steps[1]:
        failures += 1
        print("FAIL  an environment was separated from its step")
    else:
        print("ok    keeps  an environment with its step")
    if not all(balanced(s) for s in steps):
        failures += 1
        print("FAIL  a step came out with an unclosed delimiter")
    else:
        print("ok    closes every delimiter a step opens")

    if split_steps(PROSE) != PROSE.split("\n"):
        failures += 1
        print("FAIL  prose without step headers lost a line")
    else:
        print("ok    keeps  every line of a trace with no step headers")

    steps = split_steps(GPT_MATH, keep_trailers=True)
    if steps[-1] != "Final Answer: \\boxed{7}" or len(steps) != 3:
        failures += 1
        print("FAIL  a scorer asking for the whole text did not get the answer trailer")
    else:
        print("ok    keeps  the answer trailer for a scorer, as its own step")

    steps = split_steps(UNCLOSED)
    if len(steps) != 2 or not steps[1].startswith("Step 2"):
        failures += 1
        print(f"FAIL  an unclosed $$ swallowed the next step header: {len(steps)} steps")
    else:
        print("ok    reads  an unclosed $$ line by line instead of hiding the next header in it")

    kinds = [k for _, _, k in find_math("Set \\[ a \\] and $$ b $$ with \\(x\\), $y$, \\begin{cases} c \\end{cases}.")]
    if kinds != ["display", "display", "inline", "inline", "display"]:
        failures += 1
        print(f"FAIL  find_math read the five spellings as {kinds}")
    else:
        print("ok    finds  all five spellings of mathematics")

    out = normalize_math("a \\\\[6pt] b \\[x\\] c \\(y\\) \\\\(i,j)")
    if "\\\\[6pt]" not in out or "\\\\(i,j)" not in out:
        failures += 1
        print("FAIL  a LaTeX line spacing or \\\\( was rewritten as a delimiter")
    else:
        print("ok    leaves a LaTeX \\\\[6pt] and \\\\(i,j) alone")
    if "$$x$$" not in out or "$y$" not in out:
        failures += 1
        print("FAIL  \\[ \\] or \\( \\) was not written as $$ or $")
    else:
        print("ok    writes \\[ \\] and \\( \\) as the problem's $$ and $")

    if balanced("$$ a=b") or not balanced("$$ a=b $$") or balanced("\\[ a") or balanced("the set $\\{a, b"):
        failures += 1
        print("FAIL  balanced() passed an unclosed delimiter or failed a closed one")
    else:
        print("ok    tells  an unclosed $$, \\[ or brace from a closed one")

    total = 14
    print(f"\n{total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
