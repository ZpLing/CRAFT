"""The mathematics in a trace, handled in one place.

Every model writes its mathematics differently. gpt-5.4-nano puts a display
block between \\[ and \\] on lines of its own; gemini keeps $ ... $ inline on
the same line; the problem statements use $ throughout; and a trace can carry
$$ ... $$ or a \\begin{aligned} environment as well. Each stage of the pipeline
used to recognise whatever convention its author had seen. The step splitter
in Module I kept only the "Step N:" lines, so every display block a model
wrote on the line after its header was gone before the TF-IRF terms, the
z-score filter and the RKG ever saw the trace -- an OmniMATH trace from
gpt-5.4-nano kept 19% of its characters. The deduplicator knew \\[ ... \\] but
not $$ ... $$. The ROSCOE adapter rewrote delimiters on its own.

This module is the one definition of what a piece of mathematics looks like
(`DISPLAY_RE`, `INLINE_RE`, `find_math`), how it is written out
(`normalize_math`: the problem's own $ and $$), whether a piece of text closes
what it opens (`balanced`), and how a trace is cut into steps without cutting
through a block (`split_steps`). Everything that reads or rewrites a trace
goes through it.
"""

from __future__ import annotations

import re
from typing import List, Tuple

# ── What mathematics looks like ───────────────────────────────────────────
# A display block: \[ ... \], $$ ... $$, or an environment. Written on lines of
# its own by gpt-5.4-nano, inline by everything else; matched across lines.
_ENVS = r"equation\*?|align\*?|aligned|gather\*?|gathered|cases|array|eqnarray\*?|split|multline\*?"
DISPLAY_RE = re.compile(
    r"(?<!\\)\\\[.*?(?<!\\)\\\]"
    r"|\$\$.*?\$\$"
    r"|\\begin\{(" + _ENVS + r")\}.*?\\end\{\1\}",
    re.DOTALL,
)
# Inline mathematics: \( ... \) or a single-dollar span. A dollar that is part
# of $$ or escaped as \$ does not open one.
INLINE_RE = re.compile(
    r"(?<!\\)\\\(.*?(?<!\\)\\\)"
    r"|(?<![\\$])\$(?!\$)(?:[^$\\]|\\.)+?\$(?!\$)",
    re.DOTALL,
)


def find_math(text: str) -> List[Tuple[int, int, str]]:
    """Every mathematical span in `text` as (start, end, kind), kind in {display, inline}.

    Display blocks are found first so that the $$ of a block is never read as
    two empty inline spans.
    """
    spans: List[Tuple[int, int, str]] = [(m.start(), m.end(), "display") for m in DISPLAY_RE.finditer(text)]
    taken = [(s, e) for s, e, _ in spans]
    for m in INLINE_RE.finditer(text):
        if not any(s <= m.start() < e for s, e in taken):
            spans.append((m.start(), m.end(), "inline"))
    return sorted(spans)


def has_math(text: str) -> bool:
    return bool(DISPLAY_RE.search(text) or INLINE_RE.search(text))


def has_display(text: str) -> bool:
    return bool(DISPLAY_RE.search(text))


# ── How it is written out ─────────────────────────────────────────────────
_OPEN_DISPLAY = re.compile(r"(?<!\\)\\\[")
_CLOSE_DISPLAY = re.compile(r"(?<!\\)\\\]")
_OPEN_INLINE = re.compile(r"(?<!\\)\\\(")
_CLOSE_INLINE = re.compile(r"(?<!\\)\\\)")


def normalize_math(text: str) -> str:
    """Write every delimiter the way the problem statements do: $ inline, $$ display.

    \\[ \\] become $$, \\( \\) become $. Environments are left as they are; a
    $$ around them is not needed for a reader and would double a delimiter.
    """
    text = _OPEN_DISPLAY.sub("$$", text)
    text = _CLOSE_DISPLAY.sub("$$", text)
    text = _OPEN_INLINE.sub("$", text)
    text = _CLOSE_INLINE.sub("$", text)
    return text


# ── Whether a piece of text closes what it opens ──────────────────────────
def balanced(text: str) -> bool:
    """Does this text close every delimiter it opens, in either convention?

    A sentence split can land inside \\begin{cases} or between the braces of
    a set, and dropping one half leaves the rest of the trace short a
    delimiter, which is how a boxed reader once ran past the answer and
    swallowed what followed.
    """
    double = text.count("$$")
    single = text.count("$") - 2 * double
    return (text.count("{") == text.count("}")
            and len(_OPEN_DISPLAY.findall(text)) == len(_CLOSE_DISPLAY.findall(text))
            and len(_OPEN_INLINE.findall(text)) == len(_CLOSE_INLINE.findall(text))
            and text.count(r"\begin{") == text.count(r"\end{")
            and double % 2 == 0 and single % 2 == 0)


# ── How a trace is cut into steps ─────────────────────────────────────────
# The line that opens a step, in every spelling the generators use.
STEP_HEAD_RE = re.compile(r"^\s*\**\s*Step\s*(\d+)\s*\**\s*[:.\-]\s*", re.IGNORECASE)
# A line that closes the reasoning rather than continuing a step: the summary
# some traces append after their last step, and the line naming the answer.
TRAILER_RE = re.compile(
    r"^\s*\**\s*(?:Summary|Final\s+Conclusion|Final\s+Answer|Answer|Conclusion)\s*\**\s*:",
    re.IGNORECASE,
)
FINAL_CONCLUSION_RE = re.compile(r"Final\s+Conclusion\s*:", re.IGNORECASE)
LABEL_RE = re.compile(r"__(?:PROVED|DISPROVED|UNKNOWN)__", re.IGNORECASE)

_NL = "\x00"  # stands in for a newline inside a display block while lines are read


_HEADER_INSIDE = re.compile(r"\n\s*\**\s*Step\s*\d+\s*\**\s*[:.\-]", re.IGNORECASE)


def split_steps(text: str, keep_conclusion_lines: bool = True,
                keep_trailers: bool = False) -> List[str]:
    """Cut a generation into steps, each with every line that belongs to it.

    A step is its "Step N:" line and the lines after it up to the next step
    or a trailer. A display block that spans lines is never cut: its line
    breaks are hidden before the text is read line by line and restored in
    the step that holds it -- unless a "Step N:" line falls inside what the
    delimiters enclose, which is an unclosed block, not a display, and is read
    line by line like the rest. Lines before the first step are not a step
    and are left out.

    A trailer is a line that closes the reasoning rather than continuing a
    step: "Summary: ...", "Final Answer: ...". Module I leaves trailers out,
    keeping only a "Final Conclusion" line or one carrying a
    __PROVED__/__DISPROVED__ token as a step of its own (`keep_conclusion_lines`),
    which is where the RKG reads the conclusion. A scorer that must see the
    whole text as written passes `keep_trailers`, and every trailer stays, as
    its own step, in order.

    A text with no "Step N:" line at all is returned as its non-empty lines,
    display blocks kept whole.
    """
    if not text:
        return []

    def shield(m: "re.Match") -> str:
        block = m.group(0)
        return block if _HEADER_INSIDE.search(block) else block.replace("\n", _NL)

    shielded = DISPLAY_RE.sub(shield, text)
    lines = [ln.strip() for ln in shielded.splitlines() if ln.strip()]
    restore = lambda s: s.replace(_NL, "\n")  # noqa: E731

    if not any(STEP_HEAD_RE.match(ln) for ln in lines):
        return [restore(ln) for ln in lines]

    steps: List[str] = []
    current: List[str] = []
    trailers: List[str] = []
    for ln in lines:
        if STEP_HEAD_RE.match(ln):
            if current:
                steps.append("\n".join(current))
            current = [ln]
        elif TRAILER_RE.match(ln) or LABEL_RE.search(ln):
            if current:
                steps.append("\n".join(current))
                current = []
            if keep_trailers or (keep_conclusion_lines
                                 and (FINAL_CONCLUSION_RE.search(ln) or LABEL_RE.search(ln))):
                trailers.append(ln)
        elif current:
            current.append(ln)
    if current:
        steps.append("\n".join(current))
    for ln in trailers:
        if ln not in steps:
            steps.append(ln)
    return [restore(s) for s in steps]
