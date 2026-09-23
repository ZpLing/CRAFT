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
from dedup_trace import (_conclusion, _join_label_fragments,  # noqa: E402
                         _repeats, _tokens, _unbox_intermediate, dedup_steps,
                         dedup_trace)

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


# A line written out twice, character for character, inside paragraphs that
# carry displays -- which the sentence rule has to leave whole. One pair like
# this puts ROSCOE's repetition score at its worst value on its own.
VERBATIM = """Step 1: Compute the fifth term for each case.
\\[ a_5 = 5 + 4d \\]
Therefore, the possible values of the fifth term are \\(\\boxed{7,\\,-5}\\).
Step 2: Collect the two cases and state the result.
\\[ a_5 = 5 + 4d \\]
Therefore, the possible values of the fifth term are \\(\\boxed{7,\\,-5}\\)."""

# The same line twice, but the halves a split makes are not balanced on their
# own: removing one leaves the trace short a brace.
UNBALANCED = """Step 1: Consider \\begin{cases} x = 1. \\\\ y = 2. \\end{cases} and note it is finite.
Step 2: Consider \\begin{cases} x = 1. \\\\ y = 2. \\end{cases} and note it is finite."""


# (closing text as written, what it must become, why). The first two came
# apart at the sentence splitter in gpt-5.4-nano's ProofWriter traces and
# scored as one-word sentences; the rest must not be touched.
FRAGMENTS = [
    ('Step 3: Therefore, the hypothesis "The cat is blue." is __PROVED__.',
     'Step 3: Therefore, the hypothesis "The cat is blue" is __PROVED__.',
     "a full stop inside the quoted hypothesis split `is __PROVED__.` off"),
    ("Step 3: Therefore, the hypothesis (“The cow sees the squirrel”) is __PROVED__. __PROVED__",
     "Step 3: Therefore, the hypothesis (“The cow sees the squirrel”) is __PROVED__.",
     "a label echoed after the sentence that already ends on it"),
    ("Step 3: Therefore, the hypothesis “The bear sees the bear” is __PROVED__.\n\n__PROVED__",
     "Step 3: Therefore, the hypothesis “The bear sees the bear” is __PROVED__.",
     "the same echo on a line of its own"),
    ("Step 3: X is __DISPROVED__.\nFinal Conclusion: __DISPROVED__.",
     "Step 3: X is __DISPROVED__.\nFinal Conclusion: __DISPROVED__.",
     "a trailer that names the label is a sentence, not an echo"),
    ("Step 3: the hypothesis “The cat is blue.” It follows.",
     "Step 3: the hypothesis “The cat is blue.” It follows.",
     "a quoted sentence that ends the sentence keeps its stop"),
    ("Step 3: X is __PROVED__. __DISPROVED__",
     "Step 3: X is __PROVED__. __DISPROVED__",
     "a different label after the sentence is not an echo, and stays for the reader"),
]


# (trace, what it must become, why): nano boxes intermediate results.
UNBOX = [
    ("Step 3: Hence $\\boxed{0<c\\le \\sqrt5}$.\nStep 5: So the answer is $\\boxed{(0,1]}$.",
     "Step 3: Hence $0<c\\le \\sqrt5$.\nStep 5: So the answer is $\\boxed{(0,1]}$.",
     "an intermediate box loses the box, the last one keeps it"),
    ("Step 2: We get $\\boxed{\\frac{3}{1009^2}}$.\nStep 4: Thus $\\boxed{773}$.",
     "Step 2: We get $\\frac{3}{1009^2}$.\nStep 4: Thus $\\boxed{773}$.",
     "nested braces inside the intermediate box are kept whole"),
    ("Step 1: Therefore $\\boxed{5}$.", "Step 1: Therefore $\\boxed{5}$.", "a single box is left alone"),
    ("Step 2: The sum is $\\frac{30}{10}$, which simplifies to 3.\n\n\\boxed{3}\nStep 3: Hence $\\boxed{3}$.",
     "Step 2: The sum is $\\frac{30}{10}$, which simplifies to 3.\n\nStep 3: Hence $\\boxed{3}$.",
     "a box alone on its line restates the step's own result and goes with its line"),
    ("Step 1: No value yet.\n$\\boxed{None}$\nStep 2: We get $\\boxed{7}$.",
     "Step 1: No value yet.\nStep 2: We get $\\boxed{7}$.",
     "a bare $\\boxed{None}$ line goes too"),
    ("Step 2: This operation results in $y = 14 - 5$, which simplifies to the final value.\n\\boxed{y=9}\nStep 3: Hence $\\boxed{9}$.",
     "Step 2: This operation results in $y = 14 - 5$, which simplifies to the final value.\nThis gives $y=9$.\nStep 3: Hence $\\boxed{9}$.",
     "a bare box whose value the prose only implies is kept, as a sentence"),
    ("Step 2: Simplifying yields $\\frac{1}{2}$.\n\\boxed{1/2}\nStep 3: So $\\boxed{1/2}$.",
     "Step 2: Simplifying yields $\\frac{1}{2}$.\nStep 3: So $\\boxed{1/2}$.",
     "1/2 and \\frac{1}{2} are the same value, so the line is a restatement"),
    ("Step 2: The terms sum to $\\frac{30}{10}$.\n\\boxed{3}\nStep 3: Hence $\\boxed{3}$.",
     "Step 2: The terms sum to $\\frac{30}{10}$.\nThis gives $3$.\nStep 3: Hence $\\boxed{3}$.",
     "the 3 inside 30/10 is not the value 3, so the boxed value is kept"),
    ("Step 2: The terms sum to $\\frac{30}{10}$, which simplifies to 3.\n\\boxed{3}\nStep 3: Hence $\\boxed{3}$.",
     "Step 2: The terms sum to $\\frac{30}{10}$, which simplifies to 3.\nStep 3: Hence $\\boxed{3}$.",
     "3 written as its own token is the value, so the line is a restatement"),
    ("Step 1: The side is computed from the area.\n\\boxed{5 \\text{ cm}}\nStep 2: Thus $\\boxed{10}$.",
     "Step 1: The side is computed from the area.\nThis gives $5 \\text{ cm}$.\nStep 2: Thus $\\boxed{10}$.",
     "a value with a unit is a value; unstated, it is kept"),
    ("Step 1: The side is $5$ cm.\n\\boxed{5 \\text{ cm}}\nStep 2: Thus $\\boxed{10}$.",
     "Step 1: The side is $5$ cm.\nStep 2: Thus $\\boxed{10}$.",
     "stated with its unit in the prose, the line goes"),
    ("Step 1: No value yet.\n$\\boxed{\\text{None}}$\nStep 2: We get $\\boxed{7}$.",
     "Step 1: No value yet.\nStep 2: We get $\\boxed{7}$.",
     "a placeholder written as \\text{None} goes too"),
    ("Step 1: The recurrence is solved.\n\\boxed{a_n = 2^n.}\nStep 2: Thus $\\boxed{8}$.",
     "Step 1: The recurrence is solved.\nThis gives $a_n = 2^n$.\nStep 2: Thus $\\boxed{8}$.",
     "a period the model left inside the box does not double the sentence's own"),
    ("Step 1: The count is found.\n\\boxed{\\,3^m-2^{m+1}+1\\,}\nStep 2: Thus $\\boxed{8}$.",
     "Step 1: The count is found.\nThis gives $\\,3^m-2^{m+1}+1$.\nStep 2: Thus $\\boxed{8}$.",
     "a trailing thin space \\, is a LaTeX command, not a comma to strip"),
    ("Step 1: The sum is rewritten.\n\\boxed{\\;S=2n-n.\\;}\nStep 2: Thus $\\boxed{8}$.",
     "Step 1: The sum is rewritten.\nThis gives $\\;S=2n-n$.\nStep 2: Thus $\\boxed{8}$.",
     "a period before a trailing thin space is still the box's own end mark"),
    ("Step 1: So $x = \\boxed{4}$ here.\nStep 2: Thus $\\boxed{8}$.",
     "Step 1: So $x = 4$ here.\nStep 2: Thus $\\boxed{8}$.",
     "a box inside a sentence still only loses the box"),
]


# A citation may rest only on an earlier step. The repair used to pick the
# citing step itself (it always contains the claim it is about to make) or a
# later one that repeats the claim.
CITATIONS = """Step 1: According to Fact 17, if someone eats the lion then they like the dog. Since Fact 6 states that the dog eats the lion, it follows that the dog likes the dog.
Step 2: According to Fact 14, if someone is young then they visit the dog. Since Fact 9 states that the dog is young, it follows that the dog visits the dog.
Step 3: According to Fact 12, if someone likes the dog and visits the dog then they visit the squirrel. Since Step 3 established that the dog likes the dog and Step 4 established that the dog visits the dog, it follows that the dog visits the squirrel.
Step 4: According to Fact 13, if the dog visits the squirrel then the squirrel is young. Since Step 3 established that the dog visits the squirrel, it follows that the squirrel is young and the hypothesis is __PROVED__."""


# Step 2 repeats Step 1 and goes; the citations in Step 4 are a run and a
# slash pair, and every number in them has to follow the renumbering.
RUNS = """Step 1: According to Fact 3, the dog is round. Since Fact 3 states it, it follows that the dog is round.
Step 2: According to Fact 3, the dog is round. Since Fact 3 states it, it follows that the dog is round.
Step 3: According to Fact 5, if the dog is round then the dog sees the cat. Since Step 1 established that the dog is round, it follows that the dog sees the cat.
Step 4: From Steps 3, 2, and 1 (see Step2/Step3), we have that the dog sees the cat. Therefore, from Steps 3, 2, and 1 we can infer, with 4 cases and Step 3 - 3 = 0 aside, that the hypothesis is __PROVED__."""


def _forward(text: str) -> list:
    bad = []
    current = None
    for i, part in enumerate(re.split(r"(?m)^\s*Step\s*(\d+)\s*:", text)):
        if i % 2 == 1:
            current = int(part)
            continue
        if current is None:
            continue
        for run in re.finditer(r"\bSteps?\s*\d+(?:\s*(?:,|/|and|&)\s*(?:and\s+)?(?:Steps?\s*)?\d+)*", part):
            for n in re.findall(r"\d+", run.group(0)):
                if int(n) >= current:
                    bad.append(f"Step {current} -> Step {n}")
    return bad


# The problem's own algorithm, stated in steps, then the trace: its "Step 2"
# is not a citation to renumber, and its repeated sentence is not a restatement.
PREAMBLE = ("The value of $y$ does not change after Step 2: Multiply $x$ and $y$. "
            "The value of $y$ does not change. What is the final value of $x$?\n"
            "Step 1: Let $x = 1$ and $y = 2$ at the start.\n"
            "Step 2: Multiply $x$ by 2, so $x = 2$.\n"
            "Step 3: Add $y$ and 1.\n"
            "Step 4: Therefore the final value is $\\boxed{2}$.")

def main() -> int:
    failures = 0
    out = dedup_trace(RUNS)
    bad = _forward(out)
    if bad or "Step 3:" not in out or "Step 4:" in out or "with 4 cases" not in out or " - 3 = 0" not in out:
        failures += 1
        print(f"FAIL  renumbering left a citation pointing at itself or ahead, or touched a number that was not one: {bad} / {out!r}")
    else:
        print("ok    renum  every number in a citation run follows the renumbering")

    out = dedup_trace(CITATIONS)
    step3 = [l for l in out.splitlines() if l.startswith("Step 3:")][0]
    if "Since Step 1 established that the dog likes the dog" not in step3:
        failures += 1
        print(f"FAIL  a self-citation was not sent to the step that derived the line: {step3[:160]!r}")
    elif "Step 2 established that the dog visits the dog" not in step3:
        failures += 1
        print(f"FAIL  a forward citation was not sent back to the earlier step: {step3[:160]!r}")
    elif "Since Step 3 established that the dog visits the squirrel" not in out:
        failures += 1
        print("FAIL  a correct citation was changed")
    else:
        print("ok    cites  only earlier steps: self- and forward citations repaired, correct ones kept")

    for text, expected, why in UNBOX:
        got = _unbox_intermediate(text)
        if got != expected:
            failures += 1
            print(f"FAIL  got {got!r}: {why}")
        else:
            print(f"ok    unbox  {why}")

    for text, expected, why in FRAGMENTS:
        got = _join_label_fragments(text)
        if got != expected:
            failures += 1
            print(f"FAIL  got {got!r}: {why}")
        else:
            print(f"ok    {'joins' if got != text else 'keeps':<5}  {why}")

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

    out = dedup_trace(VERBATIM)
    n_answer = out.count("the possible values of the fifth term")
    if n_answer != 1:
        failures += 1
        print(f"FAIL  the answer line is still written {n_answer} times")
    elif "\\boxed{7,\\,-5}" not in out:
        failures += 1
        print("FAIL  removing the earlier copy took the answer with it")
    else:
        print("ok    drops  a line written twice inside a display paragraph")

    out = dedup_trace(UNBALANCED)
    if out.count("\\begin{cases}") != out.count("\\end{cases}"):
        failures += 1
        print("FAIL  a verbatim drop unbalanced an environment")
    else:
        print("ok    keeps  a repeat whose own delimiters do not balance")

    braced = dedup_trace(BRACES)
    if braced.count("{") != braced.count("}"):
        failures += 1
        print("FAIL  a dropped sentence left the braces unbalanced")
    else:
        print("ok    keeps  every brace it started with")

    # A mathematics trace opens with the problem's own sentences; the problem
    # may describe a procedure in numbered steps and may say the same thing
    # twice, and neither is the trace's doing.
    opened = dedup_trace(PREAMBLE)
    if not opened.startswith(PREAMBLE.split("\nStep 1:")[0]):
        failures += 1
        print(f"FAIL  the text before Step 1 was rewritten: {opened.splitlines()[0]!r}")
    elif "Step 3: Add $y$ and 1." not in opened:
        failures += 1
        print("FAIL  the body's own dedup was lost")
    else:
        print("ok    keeps  the problem's statement before Step 1 as written")

    total = len(CASES) + len(CONCLUSIONS) + 7
    print(f"\n{total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
