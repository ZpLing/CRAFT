#!/usr/bin/env python3
"""What dedup_trace must and must not drop, from traces it got wrong.

Each case here is one the deduplicator failed on a real trace before the rule
above it existed, which is why they are written down: a lexical test that only
counts shared words removes an enumeration's arithmetic, and one that only
compares order removes a line that reuses its predecessor's shape.

    python test_dedup_trace.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dedup_trace import _repeats, _tokens, dedup_trace  # noqa: E402

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
    out = dedup_trace(TRACE)
    if "Step 3:" in out:
        failures += 1
        print("FAIL  the repeated step survived")
    elif "From Step 2" not in out:
        failures += 1
        print("FAIL  the citation of the dropped step was not redirected")
    else:
        print("ok    drops  a repeated step, and sends its citation to Step 2")

    braced = dedup_trace(BRACES)
    if braced.count("{") != braced.count("}"):
        failures += 1
        print("FAIL  a dropped sentence left the braces unbalanced")
    else:
        print("ok    keeps  every brace it started with")

    print(f"\n{len(CASES) + 2 - failures}/{len(CASES) + 2} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
