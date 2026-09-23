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
from typing import List, Optional, Tuple

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
# A factorial's "!" is a sentence end to every sentence splitter, so "$n!!
# \\mid 2012!!$" comes apart into four "sentences", each a fragment that
# aligns with nothing and repeats the next. The joiner is invisible and
# breaks nothing in LaTeX; it only tells the splitter there is no boundary.
# The "!" is read as a factorial from its neighbours rather than from being
# inside a math span, because the traces that suffer most are the ones whose
# dollars do not pair up: a "!" glued to a letter, digit or closing bracket
# and followed by more mathematics -- another "!", a delimiter, an operator,
# a comma -- or by lower-case text; an exclamation is followed by a capital.
# LaTeX's own "\\!" (a negative thin space, as in "\\sin\\!\\left(") and the
# "62,\\!250" thousands separator split sentences the same way and are caught
# by the backslash before them.
FACTORIAL_JOINER = "\u2060"
_FACTORIAL = re.compile(
    r"(?<=\\)!(?!\u2060)"
    r"|(?<=[A-Za-z0-9)}\]!$\u2060])!(?!\u2060)"
    r"(?=[!$\\()}\],;:=<>+\-*/.|^_]|\s+[a-z\\$(0-9=<>+\-|])")


def protect_factorials(text: str) -> str:
    """Mark every factorial's "!" as not ending a sentence."""
    if "!" not in (text or ""):
        return text
    return _FACTORIAL.sub("!" + FACTORIAL_JOINER, text)


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
    step: "Summary: ...", "Final Answer: ...". It opens a block of its own,
    in place, with the lines that follow it, so that what a model writes after
    "Conclusion: ..." -- the display holding its boxed answer, say -- is
    neither lost nor moved. Module I leaves trailer blocks out, keeping only a
    "Final Conclusion" line or one carrying a __PROVED__/__DISPROVED__ token
    (`keep_conclusion_lines`), which is where the RKG reads the conclusion. A
    scorer that must see the whole text as written passes `keep_trailers`,
    and every block stays where it was.

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

    # (lines, is_trailer_block) in the order written
    blocks: List[Tuple[List[str], bool]] = []
    current: Optional[List[str]] = None
    for ln in lines:
        if STEP_HEAD_RE.match(ln):
            current = [ln]
            blocks.append((current, False))
        elif TRAILER_RE.match(ln) or LABEL_RE.search(ln):
            current = [ln]
            blocks.append((current, True))
        elif current is not None:
            current.append(ln)

    steps: List[str] = []
    for block, is_trailer in blocks:
        if is_trailer and not keep_trailers:
            head = block[0]
            if keep_conclusion_lines and (FINAL_CONCLUSION_RE.search(head) or LABEL_RE.search(head)):
                steps.append(head)
            continue
        steps.append("\n".join(block))
    return [restore(s) for s in steps]


# ── Stating the goal ──────────────────────────────────────────────────────
# A sentence boundary: end punctuation, then space, then something that
# starts a sentence. "e.g." and "i.e." are not boundaries.
# Forum markup some problem statements carry ([i]...[/i], [list], [*]).
_MARKUP_TAG = re.compile(r"\[/?(?:[a-z]+|\*)\]")
_STEP_HEADER = re.compile(r"\s*\**\s*Step\s*\d+\s*\**\s*[:.\-]")
_SENTENCE_END = re.compile(r"(?<!\be\.g\.)(?<!\bi\.e\.)(?<=[.?!])\s+(?=[A-Z\"\u201c$(\\])")


def goal_sentences(problem: str, lo: int = 6, hi: int = 40, max_math: int = 2) -> str:
    """The problem's own sentences that can open a trace, verbatim, or "".

    A mathematics trace that begins by restating what is given and what is
    asked is the trace a solver writes; the synthesizer's first step tends to
    jump into a construction instead. This picks the sentences of the problem
    statement that read as sentences on their own -- a whole sentence with its
    end mark, no display block, at most `max_math` pieces of inline
    mathematics, and between `lo` and `hi` words counting each piece of
    mathematics as one -- and returns them in order, joined by a space. A
    sentence that is mostly a formula, or a definition that runs for a
    paragraph, is not a statement of the goal and stays out.
    """
    keep = []
    for sentence in _SENTENCE_END.split(problem.strip()):
        sentence = sentence.strip()
        if not sentence or sentence[-1] not in ".?!":
            continue
        if (has_display(sentence) or "$$" in sentence or "\\begin" in sentence
                or not balanced(sentence) or _MARKUP_TAG.search(sentence)
                or _STEP_HEADER.match(sentence)):
            # A problem that lists its own procedure as "Step 1: ..." would
            # open the trace with what reads as the trace's first step.
            continue
        spans = find_math(sentence)
        if len(spans) > max_math:
            continue
        prose = sentence
        for start, end, _ in sorted(spans, reverse=True):
            prose = prose[:start] + " MATH " + prose[end:]
        if not lo <= len(prose.split()) <= hi:
            continue
        keep.append(sentence)
    return " ".join(keep)


def prepend_goal(trace: str, problem: str) -> str:
    """The trace with the problem's goal sentences as its first line.

    The line goes before "Step 1" and is not a step: it is what the steps are
    about. Nothing is added when the problem has no sentence that qualifies
    or when the trace already opens with the line.
    """
    goal = goal_sentences(problem)
    trace = trace.lstrip("\n")
    if not goal or trace.startswith(goal):
        return trace
    return goal + "\n" + trace

