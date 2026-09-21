#!/usr/bin/env python3
"""What dedup_trace must and must not drop, from traces it got wrong.

Each case here is one the deduplicator failed on a real trace before the rule
above it existed, which is why they are written down: a lexical test that only
counts shared words removes an enumeration's arithmetic, and one that only
compares order removes a line that reuses its predecessor's shape.

    python test_dedup_trace.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dedup_trace import (_conclusion, _repeats, _tokens,  # noqa: E402
                         dedup_steps, dedup_trace)

_HEAD = re.compile(r"(?m)^\s*Step\s*(\d+)\s*:")

# (later sentence, the earlier one it resembles, is it a restatement, why)
CASES = [
    ("Since $5$ is a prime number, the condition $S(n) \\in \\mathbb{P}$ is satisfied.",
     "Since $2$ is a prime number, the condition $S(n) \\in \\mathbb{P}$ is satisfied.",
     False, "an enumeration checking a second value, not a repeat of the first"),
    ("The coordinates are $A = (0, 0)$, $B = (c, 0)$, $D = (b \\cos \\alpha, b \\sin \\alpha)$.",
     "The coordinates of the vertices are $A = (0, 0)$, $B = (a, 0)$, "
     "$D = (b \\cos \\alpha, b \\sin \\alpha)$.",
     False, "a second placement: c is not a"),
    ("Expanding the squared terms, we get $KM^2 = 10^2 + (20^2)$.",
     "Expanding the squared terms, we get $KM^2 = 100 + (20^2)$.",
     False, "the step before the square is evaluated"),
    ("From **Step 4** and **Fact16**, infer that **the dog chases the dog**.",
     "From Step 3 and Fact16, infer that **the squirrel chases the dog**.",
     False, "a different subject built from the same words"),
    ("Thus, $c_a = c$ for some constant $c \\in \\mathbb{Z}^+$.",
     "Thus, $d_a = c$ for some constant $c \\in \\mathbb{Z}^+$.",
     False, "a second symbol: c_a is not d_a, even where c appears alone"),
    ("Step 3: From Step 1, we know the cow likes the mouse.",
     "Step 2: From Step 1, we know the cow likes the mouse.",
     True, "the same line under a second number"),
    ("Since the bear chases the bald eagle (established in Step 2), it follows "
     "that the bald eagle chases the bear.",
     "Since the bear chases the bald eagle (established in Step 2), it follows "
     "that the bald eagle chases the bear.",
     True, "written out twice, word for word"),
]

# A paragraph whose braces balance across a sentence boundary: dropping the
# repeated half alone would leave the trace one brace short.
BRACES = """Step 1: Consider the set $\\{a, b\\}$ and note that it is finite here.
Step 2: Consider the set $\\{a, b\\}$ and note that it is finite here.
Step 3: Therefore the answer is $2$."""

TRACE = """Step 1: From Fact8 and Fact14, infer that the squirrel is nice.
Step 2: From Step 1 and Fact16, infer that the squirrel chases the dog.
Step 3: From Step 1 and Fact16, infer that the squirrel chases the dog.
Step 4: From Step 3, conclude the hypothesis is __PROVED__."""


# What a step derives, against what it merely quotes to get there. Reading the
# premise as the conclusion made the step that builds on Step 4 look like a
# repeat of it, and dropped the line that carried the proof forward.
CONCLUSIONS = [
    ("From **Step 4** we have **\u201cthe squirrel likes the tiger.\u201d** Using "
     "**Fact18**, with the instantiation **someone = the squirrel**, infer "
     "**\u201cthe tiger visits the dog.\u201d**",
     "tiger visits dog", "the claim it lands on, not the premise it opens with"),
    ("From Step 3 (\u201cthe bear sees the rabbit\u201d) and Fact12, infer that the "
     "bear visits the bald eagle.",
     "bear visits bald eagle", "infer that"),
    ("Adding the two fractions over a common denominator, therefore the "
     "probability is \\boxed{\\frac{2}{9}}.",
     "probability \\boxed{\\frac{2}{9}}", "therefore, with the answer in it"),
    ("We now compute the remaining sum explicitly.", None,
     "no derivation announced, so nothing to compare"),
]

# The same conclusion reached twice, worded differently each time: too far apart
# for the sentence rule, and the second one adds nothing to the proof.
RESTATED = """Step 1: From Fact3 and Fact8, infer that the mouse needs the lion.
Step 2: From Step 1 and Fact10, infer that the mouse needs the rabbit.
Step 3: Applying Fact10 to Step 1 once more, it follows that the mouse needs \
the rabbit.
Step 4: From Step 3, conclude the hypothesis is __PROVED__."""


def main() -> int:
    failures = 0
    for later, earlier, expected, why in CASES:
        got = _repeats(later, earlier, _tokens(later), _tokens(earlier), 0.75)
        if got != expected:
            failures += 1
            print(f"FAIL  expected {expected}, got {got}: {why}")
        else:
            print(f"ok    {'drops' if expected else 'keeps':<5}  {why}")

    # A step that goes must take its header with it, and what cited it must be
    # sent to the step whose line it repeated.
    for step, expected, why in CONCLUSIONS:
        got = _conclusion(step)
        if (got or "") != (expected or ""):
            failures += 1
            print(f"FAIL  read {got!r}, wanted {expected!r}: {why}")
        else:
            print(f"ok    reads  {why}")

    out = dedup_trace(TRACE)
    if out.count("infer that the squirrel chases the dog") != 1:
        failures += 1
        print("FAIL  the repeated step survived")
    elif "From Step 2, conclude" not in out:
        failures += 1
        print("FAIL  the citation of the dropped step was not redirected")
    elif [int(n) for n in _HEAD.findall(out)] != [1, 2, 3]:
        failures += 1
        print(f"FAIL  the surviving steps are numbered {_HEAD.findall(out)}")
    else:
        print("ok    drops  a repeated step, redirects its citation, renumbers")

    out = dedup_trace(RESTATED)
    if out.count("needs the rabbit") != 1:
        failures += 1
        print("FAIL  a conclusion reached twice was written twice")
    elif "From Step 2, conclude" not in out:
        failures += 1
        print("FAIL  the citation was not sent to the step that first derived it")
    else:
        print("ok    drops  a step that re-derives an earlier conclusion")

    # The export scores a list of steps whose "Step N:" headers are gone, but
    # whose citations of them are not, so the list path has to renumber against
    # the numbers the text uses -- which start at one.
    listed = dedup_steps([line.split(": ", 1)[1] for line in RESTATED.splitlines()])
    if len(listed) != 3:
        failures += 1
        print(f"FAIL  the list path kept {len(listed)} steps, wanted 3")
    elif "From Step 1 and Fact10" not in listed[1]:
        failures += 1
        print(f"FAIL  a citation moved that should not have: {listed[1]!r}")
    elif "From Step 2, conclude" not in listed[2]:
        failures += 1
        print(f"FAIL  the citation of the dropped step was not redirected: {listed[2]!r}")
    else:
        print("ok    drops  the same step through the list the export scores")

    braced = dedup_trace(BRACES)
    if braced.count("{") != braced.count("}"):
        failures += 1
        print("FAIL  a dropped sentence left the braces unbalanced")
    else:
        print("ok    keeps  every brace it started with")

    total = len(CASES) + len(CONCLUSIONS) + 4
    print(f"\n{total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
