#!/usr/bin/env python3
"""Drop the paragraphs a synthesized trace copies back from its earlier steps.

Walking the consensus graph carries each step's conclusion forward, so a late
step restates what earlier ones established before adding anything of its own:
across the eight cells 8% to 20% of sentences repeat an earlier one, a maths
trace names "Step N" seven to twelve times, and the final \\boxed{} answer is
written out up to four times. None of that is reasoning, and all of it counts
against the trace under the ROSCOE metrics defined over pairs of sentences.

What must not move is the answer. extract_label reads it from the *last*
commitment a trace makes -- [-1] on every path it has, for __PROVED__, for
\\boxed{}, for "the answer is" and for the natural-language fallback -- so a
trace is only rewritten when re-reading it returns what it returned before,
and is otherwise kept as it was. `dedup_trace` applies that check itself.

Paragraphs, not sentences, are the unit: a sentence split lands inside
\\begin{cases} ... \\end{cases}, and dropping one fragment of a display leaves
the braces unbalanced, which made the boxed reader run past the end of the
answer and swallow the rest of the trace.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional

_STEP = re.compile(r"(?m)^(\s*Step\s*\d+\s*[:.\-]\s*)")
_PARA = re.compile(r"\n\s*\n")
_WORD = re.compile(r"[A-Za-z\\]+|\d+")

# A paragraph that states an answer is never dropped, wherever it sits.
_ANSWER = re.compile(
    r"\\boxed\{|####|__(?:PROVED|DISPROVED)__|\*\*(?:PROVED|DISPROVED)\*\*"
    r"|(?:final\s+)?answer\s*[:=]|the\s+answer\s+is|final\s+conclusion",
    re.IGNORECASE,
)


def _tokens(text: str) -> frozenset:
    return frozenset(_WORD.findall(text.lower()))


_MATHY = re.compile(r"\\\[|\\\]|\\begin\{|\\end\{|\$\$|\\boxed\{")
_SENT = re.compile(r"(?<=[.!?])\s+")


def _is_prose(text: str) -> bool:
    """A paragraph with no display maths in it can be split a sentence at a time."""
    return not _MATHY.search(text)


def _balanced(text: str) -> bool:
    """Does this paragraph close every delimiter it opens?"""
    if text.count("{") != text.count("}"):
        return False
    if text.count(r"\[") != text.count(r"\]"):
        return False
    if text.count(r"\begin{") != text.count(r"\end{"):
        return False
    return text.count("$") % 2 == 0


def dedup_trace(text: str,
                threshold: float = 0.75,
                extractor: Optional[Callable[[str], object]] = None) -> str:
    """Return the trace with its restatements removed, or unchanged.

    `extractor` is the answer reader this trace will be scored with. When it is
    given and the rewrite would change what it returns, the original is handed
    back instead, so a cell's accuracy cannot move.
    """
    if not text or not text.strip():
        return text

    pieces = _STEP.split(text)
    heads = [p for p in pieces if _STEP.fullmatch(p or "")]
    bodies = [p for p in pieces if not _STEP.fullmatch(p or "")]
    kept = dedup_steps(bodies, threshold)
    if len(kept) != len(bodies):
        return text
    rebuilt = []
    bi = 0
    for piece in pieces:
        if _STEP.fullmatch(piece or ""):
            rebuilt.append(piece)
        else:
            rebuilt.append(kept[bi]); bi += 1
    rewritten = "".join(rebuilt)
    if extractor is not None and extractor(rewritten) != extractor(text):
        return text
    return rewritten


def dedup_steps(steps: List[str], threshold: float = 0.75,
                extractor: Optional[Callable[[str], object]] = None) -> List[str]:
    """The same pass over a trace already split into steps.

    The ROSCOE export joins a trace's steps with a space before scoring, which
    leaves neither the "Step N:" line starts nor the blank lines `dedup_trace`
    reads, so the export deduplicates the list and joins what comes back.
    A step whose every paragraph repeats an earlier one drops out entirely.
    """
    seen: List[frozenset] = []
    out: List[str] = []
    for step in steps:
        kept: List[str] = []
        for para in _PARA.split(step or ""):
            body = para.strip()
            if not body:
                kept.append(para)
                continue
            if _ANSWER.search(body) or not _balanced(para):
                kept.append(para)
                continue
            # Prose splits a sentence at a time, which is what reaches the
            # metric; a paragraph carrying a display is kept whole, because a
            # split inside one leaves its braces unbalanced.
            units = _SENT.split(body) if _is_prose(body) else [body]
            survivors = []
            for unit in units:
                u = unit.strip()
                toks = _tokens(u)
                if len(u) < 40 or len(toks) < 6:
                    survivors.append(unit)
                    continue
                if _ANSWER.search(u):
                    survivors.append(unit)
                    continue
                if any(len(toks & prev) / max(1, len(toks | prev)) > threshold
                       for prev in seen):
                    continue
                seen.append(toks)
                survivors.append(unit)
            joined_para = " ".join(x for x in survivors if x.strip())
            if joined_para.strip():
                kept.append(joined_para)
        out.append("\n\n".join(k for k in kept if k.strip()))
    if not any(o.strip() for o in out):
        return list(steps)
    if extractor is not None and extractor(" ".join(out)) != extractor(" ".join(steps)):
        return list(steps)
    return out
