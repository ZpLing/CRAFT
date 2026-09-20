#!/usr/bin/env python3
"""Counting the reasoning steps in a trace, the same way for every system.

average steps is a reported comparison and lower is better, so the number has to
mean the same thing on both sides of it. It did not. CRAFT counts its own steps
exactly — its synthesis emits one step per node of the consensus RKG and keeps
them as a list — while a baseline's trace was measured by counting its non-empty
lines and capping the count at 30. On the runs finished so far that fallback
fired on 91% of traces, and 17% of them hit the cap, where the real length is
unknown and only known to be at least 30. Capping the longest traces at 30
flatters exactly the baselines that ramble, and counting newlines makes a trace's
score depend on how it happens to be formatted.

The rule here is CRAFT's: a step is a discrete reasoning move that the trace
itself delimits.

    1. A trace that declares its steps — "Step 3:", "3.", "3)" — is counted by
       those markers. This is what CRAFT's own reasoning_steps list contains, so
       the list and the markers give the same number and CRAFT is measured by
       the same rule as everything else.
    2. A trace that declares nothing is segmented into sentences. A sentence is
       the smallest thing that can carry a reasoning move; a line is a typesetting
       artifact.
    3. Nothing is capped. A long trace counts as long, which is the point of
       reporting the number.

`basis()` reports which rule produced a count, so the proportion measured each
way can be stated rather than assumed.
"""

from __future__ import annotations

import re
from typing import List, Optional

# "Step 3:", "Step 3.", "步骤3:" at the start of a line.
_STEP_MARKER = re.compile(r"^\s*(?:step|步骤)\s*\d+\s*[:.、]", re.IGNORECASE | re.MULTILINE)
# "3." or "3)" or "(3)" at the start of a line — an enumerated reasoning list.
_ENUM_MARKER = re.compile(r"^\s*\(?\d{1,2}[.)]\s+\S", re.MULTILINE)
# A sentence ends at . ! ? or their full-width forms, or at a newline that
# follows one. Decimal points and common abbreviations are not sentence ends.
_SENT_SPLIT = re.compile(r"(?<![0-9])[.!?。！？]+(?=\s|$)|\n{2,}")

_BOILERPLATE = re.compile(
    r"^\s*(?:__(?:PROVED|DISPROVED)__|\\boxed\{[^}]*\}|answer\s*[:=]|"
    r"final answer\s*[:=]|conclusion\s*[:=])\s*$",
    re.IGNORECASE)


def _sentences(text: str) -> List[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]
    # A fragment with no letters is punctuation or a stray number, not a step.
    return [p for p in parts if re.search(r"[A-Za-z一-鿿]", p) and len(p) > 2]


def basis(text: str, reasoning_steps: Optional[List] = None) -> str:
    """Which rule decides this trace's step count: list, markers, enum, sentences."""
    if reasoning_steps and isinstance(reasoning_steps, list):
        return "list"
    if not text:
        return "empty"
    if _STEP_MARKER.search(text):
        return "markers"
    if len(_ENUM_MARKER.findall(text)) >= 2:
        return "enumerated"
    return "sentences"


def count_steps(text: str, reasoning_steps: Optional[List] = None) -> int:
    """The number of reasoning steps in a trace. Never capped."""
    if reasoning_steps and isinstance(reasoning_steps, list):
        return len([s for s in reasoning_steps if s and str(s).strip()])
    if not text:
        return 0
    marks = _STEP_MARKER.findall(text)
    if marks:
        return len(marks)
    enum = _ENUM_MARKER.findall(text)
    if len(enum) >= 2:
        return len(enum)
    sents = [s for s in _sentences(text) if not _BOILERPLATE.match(s)]
    return len(sents)


def count_tokens(text: str) -> int:
    """Approximate token count via whitespace split (~0.75x real BPE tokens)."""
    return len(text.split()) if text else 0
