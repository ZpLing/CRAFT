#!/usr/bin/env python3
"""Counting the reasoning steps in a trace, the same way for every system.

average steps is a reported comparison and lower is better, so the number has to
mean the same thing on both sides of it. CRAFT counts its own steps with
`extract_reasoning_steps` in module I, and the list it produces is what its
synthesis carries forward and what every later stage reads. Measuring a baseline
by any other rule compares two different quantities and calls them one column.

So the rule here is CRAFT's rule, copied from
framework/module1_generation_filtering/generate_traces.py rather than
reinvented:

    1. A trace that declares its steps — lines beginning "Step 3:" — is those
       lines, plus any conclusion line not already among them. A conclusion is
       a reasoning move and CRAFT keeps it so the RKG can find the conclusion
       node; dropping it here would make the same trace count one step shorter
       on the baseline side than on CRAFT's.
    2. A trace that declares nothing is its non-empty lines, one step each.
    3. Nothing is capped. A long trace counts as long, which is the point of
       reporting the number.

When CRAFT's own `reasoning_steps` list is at hand, that list is the count: it
is this same function's output, already computed upstream and then narrowed by
the anomaly filter, so recomputing it from the rendered text would measure the
trace before filtering instead of after.

`basis()` reports which rule produced a count, so the proportion measured each
way can be stated rather than assumed.
"""

from __future__ import annotations

import re
from typing import List, Optional

# Copied from generate_traces.py. A step is a line that says it is one.
_STEP_PATTERN = re.compile(r"^Step\s*\d+\s*:", re.IGNORECASE)
_FINAL_CONCLUSION_PATTERN = re.compile(
    r"Final\s+Conclusion\s*:\s*(__PROVED__|__DISPROVED__)", re.IGNORECASE)
_LABEL_TOKEN_PATTERN = re.compile(r"__PROVED__|__DISPROVED__", re.IGNORECASE)


def extract_reasoning_steps(text: str) -> List[str]:
    """CRAFT's step segmentation, applied to any trace."""
    if not text:
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    steps = [line for line in lines if _STEP_PATTERN.match(line)]
    if steps:
        conclusion_lines = [l for l in lines
                            if _FINAL_CONCLUSION_PATTERN.search(l)
                            or _LABEL_TOKEN_PATTERN.search(l)]
        for cl in conclusion_lines:
            if cl not in steps:
                steps.append(cl)
        return steps
    return lines


def basis(text: str, reasoning_steps: Optional[List] = None) -> str:
    """Which rule decides this trace's step count: list, markers, or lines."""
    if reasoning_steps and isinstance(reasoning_steps, list):
        return "list"
    if not text:
        return "empty"
    if any(_STEP_PATTERN.match(l.strip()) for l in text.splitlines()):
        return "markers"
    return "lines"


def count_steps(text: str, reasoning_steps: Optional[List] = None) -> int:
    """The number of reasoning steps in a trace. Never capped."""
    if reasoning_steps and isinstance(reasoning_steps, list):
        return len([s for s in reasoning_steps if s and str(s).strip()])
    return len(extract_reasoning_steps(text))


def count_tokens(text: str) -> int:
    """Approximate token count via whitespace split (~0.75x real BPE tokens)."""
    return len(text.split()) if text else 0
