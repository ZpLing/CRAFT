#!/usr/bin/env python3
"""Drop the paragraphs a synthesized trace copies back from its earlier steps.

Walking the consensus graph carries each step's conclusion forward, so a late
step restates what earlier ones established before adding anything of its own:
across the eight cells 8% to 20% of sentences repeat an earlier one, a maths
trace names "Step N" seven to twelve times, and the final \\boxed{} answer is
written out up to four times. None of that is reasoning, and all of it counts
against the trace under the ROSCOE metrics defined over pairs of sentences,
every one of which takes a maximum over those pairs.

Three things are preserved.

The answer. extract_label reads it from the *last* commitment a trace makes --
[-1] on every path it has, for __PROVED__, for \\boxed{}, for "the answer is"
and for the natural-language fallback -- so a paragraph that states one is
never dropped, and `dedup_trace` checks the rewrite against the reader it will
be scored with before keeping it.

The maths. Paragraphs, not sentences, are the unit wherever a display appears:
a sentence split lands inside \\begin{cases} ... \\end{cases}, and dropping one
fragment leaves the braces unbalanced, which makes the boxed reader run past
the answer and swallow the rest of the trace. Prose, which carries most of the
repetition, still splits a sentence at a time.

The references. A step that says nothing new drops out with its header, since
"Step 3: Step 4:" with nothing between them is worse than the repetition it
removes. Half the traces that lose a step cite it by number from a later one,
so those citations are redirected to the step whose content was repeated --
"From Step 5 (the squirrel chases the dog)" becomes "From Step 2", which is
where that line was first derived.
"""

from __future__ import annotations

import difflib
import re
from typing import Callable, Dict, List, Optional, Tuple

_STEP = re.compile(r"(?m)^(\s*Step\s*\d+\s*[:.\-]\s*)")
_STEP_NO = re.compile(r"(?m)^\s*Step\s*(\d+)\s*[:.\-]")
_STEP_REF = re.compile(r"(?i)(\bSteps?\s*)(\d+)")
_PARA = re.compile(r"\n\s*\n")
_SENT = re.compile(r"(?<=[.!?])\s+")
# A subscripted symbol is one name: split c_a into "c" and "a" and it stops
# being distinguishable from d_a whenever the sentence also mentions c on
# its own, which is how "Thus, $c_a = c$" was read as a repeat of
# "Thus, $d_a = c$".
_WORD = re.compile(r"[A-Za-z\\]+(?:_\{?[A-Za-z0-9]+\}?)?|\d+")
_MATHY = re.compile(r"\\\[|\\\]|\\begin\{|\\end\{|\$\$|\\boxed\{")

# A display is balanced by construction, so one can be lifted out of a
# paragraph the paragraph rule has to keep whole. That is where most of what
# survives sentence deduplication sits: a trace that derives a_n once quotes
# the formula again in every step that uses it -- four copies of the same two
# lines, each wrapped in different prose, so no two paragraphs match. The
# prose already says which step it came from, so the second copy onwards is
# dropped and the citation left standing.
_DISPLAY = re.compile(r"\\\[.*?\\\]", re.DOTALL)

# A paragraph that states an answer is never dropped, wherever it sits.
_ANSWER = re.compile(
    r"\\boxed\{|####|__(?:PROVED|DISPROVED)__|\*\*(?:PROVED|DISPROVED)\*\*"
    r"|(?:final\s+)?answer\s*[:=]|the\s+answer\s+is|final\s+conclusion",
    re.IGNORECASE,
)


def _tokens(text: str) -> frozenset:
    return frozenset(_WORD.findall(text.lower()))


# Token overlap alone cannot tell "the dog chases the dog" from "the squirrel
# chases the dog": the second holds every word of the first, so the two score
# 0.85 and the later one looks like a restatement when it states something new.
# A near-verbatim repeat also matches in order, which a sequence ratio sees and
# a bag of words does not, so both have to agree before anything is dropped.
_VERBATIM = 0.95

# Neither test survives work that reuses a sentence's shape. A maths trace
# checking n = 2, then n = 5, then n = 7 writes "Since $5$ is a prime number,
# the condition is satisfied" against "Since $2$ ..."; one placing a
# parallelogram writes "$B = (c, 0)$" against "$B = (a, 0)$". Both score 0.93
# or better by sequence and share every word but one, and both are doing
# something their source did not.
#
# What separates them is that a restatement says nothing new: every word of it
# was already in the sentence it repeats. A line that carries a value or a
# symbol its source lacks -- 5 against 2, c against a -- is new work, whatever
# the two score. Step citations are renumbering, not content, so they come out
# before the comparison.
_STEP_CITE = re.compile(r"(?i)\bSteps?\s*\d+")


def _content(text: str) -> frozenset:
    return frozenset(_WORD.findall(_STEP_CITE.sub(" ", text).lower()))


def _drop_repeated_displays(paragraph: str, seen: set) -> str:
    """Remove a display this trace has already written out once."""
    def maybe(m: re.Match) -> str:
        body = m.group(0)
        if _ANSWER.search(body):
            return body
        key = " ".join(body.split())
        if key in seen:
            return ""
        seen.add(key)
        return body
    out = _DISPLAY.sub(maybe, paragraph)
    return re.sub(r"\n{3,}", "\n\n", out)


def _repeats(candidate: str, source: str, toks: frozenset,
             prev: frozenset, threshold: float) -> bool:
    if len(toks & prev) / max(1, len(toks | prev)) <= threshold:
        return False
    if _content(candidate) - _content(source):
        return False
    return difflib.SequenceMatcher(None, candidate, source).ratio() > _VERBATIM


def _balanced(text: str) -> bool:
    """Does this paragraph close every delimiter it opens?"""
    return (text.count("{") == text.count("}")
            and text.count(r"\[") == text.count(r"\]")
            and text.count(r"\begin{") == text.count(r"\end{")
            and text.count("$") % 2 == 0)


def _is_prose(text: str) -> bool:
    """A paragraph with no display maths in it can be split a sentence at a time."""
    return not _MATHY.search(text)


def _trim(bodies: List[str], owners: List[Optional[str]],
          threshold: float) -> Tuple[List[str], Dict[str, str]]:
    """Trim each body against what the earlier ones already said.

    `owners[i]` is the step number body i belongs to, or None for text before
    the first step. Returns the trimmed bodies and, for each step that ends up
    empty, the step its content repeated.
    """
    seen: List[Tuple[frozenset, Optional[str], str]] = []
    displays: set = set()
    trimmed: List[str] = []
    sources: Dict[str, List[str]] = {}
    for body, owner in zip(bodies, owners):
        kept: List[str] = []
        for para in _PARA.split(body or ""):
            text = para.strip()
            if not text:
                continue
            if _ANSWER.search(text) or not _balanced(para):
                kept.append(para)
                continue
            if not _is_prose(text):
                text = _drop_repeated_displays(text, displays)
                if not text.strip():
                    continue
                kept.append(text)
                continue
            units = _SENT.split(text)
            survivors: List[str] = []
            for unit in units:
                piece = unit.strip()
                toks = _tokens(piece)
                # The paragraph balances, but a sentence inside it need not:
                # "the set $\\{a, b\\}$" splits after a full stop that falls
                # between the braces, and dropping that half leaves the rest
                # of the trace one brace short.
                if (len(piece) < 40 or len(toks) < 6 or _ANSWER.search(piece)
                        or not _balanced(unit)):
                    survivors.append(unit)
                    continue
                match = next((src for prev, src, prev_text in seen
                              if _repeats(piece, prev_text, toks, prev, threshold)),
                             "")
                if match != "":
                    if owner is not None and match is not None:
                        sources.setdefault(owner, []).append(match)
                    continue
                seen.append((toks, owner, piece))
                survivors.append(unit)
            joined = " ".join(x for x in survivors if x.strip())
            if joined.strip():
                kept.append(joined)
        trimmed.append("\n\n".join(k for k in kept if k.strip()))

    alias: Dict[str, str] = {}
    for body, owner in zip(trimmed, owners):
        if owner is not None and not body.strip() and sources.get(owner):
            picks = sources[owner]
            alias[owner] = max(set(picks), key=picks.count)
    return trimmed, alias


# What a step derives, however it announces the derivation. A step that reaches
# a conclusion an earlier one already reached has added nothing to the proof,
# and the sentence rule above cannot see it: the two are worded differently, so
# they never reach the verbatim ratio, and where the conclusion is the final
# answer the sentence carrying it is protected outright. That is how a maths
# trace comes to write \boxed{26} in step 5 and again in step 7.
# Only words that announce a derivation. "we have" is not one of them: in these
# traces it introduces the premise a step starts from -- "From Step 4 we have
# X, so by Fact18 infer Y" -- and reading it as the conclusion made Y's step
# look like a repeat of X's, which is the step it builds on. "that" is optional
# after infer and conclude, because the traces write both "infer that X" and
# "infer **X**".
_DERIVES = re.compile(
    r"(?is)\b(?:infer(?:\s+that)?|conclude(?:\s+that)?|it follows that|"
    r"therefore|thus|hence|which gives|this gives|yields?)\b"
    r"[,:\s]*(.+?)(?=[.;]|$)")
# Wording that carries no claim, so two steps sharing it are not the same step.
_FILLER = re.compile(r"\b(the|a|an|that|this|is|are|be|we|it|to|of|and|then|"
                     r"so|now|next|finally|therefore|thus|hence)\b")


def _conclusion(body: str) -> Optional[str]:
    """The last thing a step asserts, normalised for comparison."""
    found = _DERIVES.findall(body or "")
    if not found:
        return None
    text = _STEP_CITE.sub(" ", found[-1].lower())
    text = re.sub(r"[^a-z0-9\\{}^_]+", " ", text)
    text = _FILLER.sub(" ", text)
    text = " ".join(text.split())
    # Too short to be distinctive: "it holds", "this is true". Steps sharing
    # nothing but a turn of phrase would otherwise collapse into each other.
    # A stated answer is exempt from the word count: "\\boxed{\\frac{2}{9}}"
    # is two words and identifies the claim exactly, and a trace that derives
    # the same boxed value twice is the repetition this is here to remove.
    if "\\boxed" in text or "__proved__" in text or "__disproved__" in text:
        return text if len(text) >= 8 else None
    return text if len(text) >= 12 and len(text.split()) >= 3 else None


def _repeat_conclusions(bodies: List[str],
                        owners: List[Optional[str]]) -> Dict[str, str]:
    """Map each step that re-derives an earlier conclusion to that earlier step.

    The last step is never mapped. It is where extract_label reads the answer
    from, and a trace whose final step restates the one before it is still a
    trace that ends by stating its answer -- which is what the reader needs.
    """
    last = next((o for o, b in zip(reversed(owners), reversed(bodies))
                 if o is not None and b.strip()), None)
    first_seen: Dict[str, str] = {}
    repeats: Dict[str, str] = {}
    for body, owner in zip(bodies, owners):
        if owner is None or not body.strip():
            continue
        claim = _conclusion(body)
        if claim is None:
            continue
        if claim in first_seen:
            if owner != last:
                repeats[owner] = first_seen[claim]
        else:
            first_seen[claim] = owner
    return repeats


def _resolve(alias: Dict[str, str]) -> Dict[str, str]:
    """Follow a chain of dropped steps back to the one that still exists."""
    out: Dict[str, str] = {}
    for key in alias:
        seen = {key}
        target = alias[key]
        while target in alias and target not in seen:
            seen.add(target)
            target = alias[target]
        if target not in alias:
            out[key] = target
    return out


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
    heads: List[Optional[str]] = []
    bodies: List[str] = []
    pending: Optional[str] = None
    seen_head = False
    for piece in pieces:
        if _STEP.fullmatch(piece or ""):
            pending = piece.strip()
            seen_head = True
            continue
        heads.append(pending if seen_head else None)
        bodies.append((piece or "").strip())
        pending = None

    owners = [_STEP_NO.match(h).group(1) if h and _STEP_NO.match(h) else None
              for h in heads]
    trimmed, alias = _trim(bodies, owners, threshold)

    # A step whose conclusion an earlier step already reached goes out whole,
    # citations and all, rather than sentence by sentence: what makes it a
    # repeat is the claim it lands on, not the words it gets there with.
    for owner, source in _repeat_conclusions(trimmed, owners).items():
        alias.setdefault(owner, source)
        for i, o in enumerate(owners):
            if o == owner:
                trimmed[i] = ""
    alias = _resolve(alias)

    # A step that emptied out but has nowhere to send its citations keeps the
    # body it had: dropping it would leave a later "from Step N" pointing at
    # nothing, which is worse than the repetition.
    for i, (owner, body) in enumerate(zip(owners, trimmed)):
        if owner is not None and not body.strip() and owner not in alias:
            trimmed[i] = bodies[i]

    # What survives is renumbered from one. Leaving the original numbers behind
    # would open a gap wherever a step went out -- "Step 1, Step 2, Step 4" --
    # and every citation of a dropped step would point at a number the trace no
    # longer has. Both are the same map: a surviving step takes its new
    # position, and a dropped one takes the new position of the step whose
    # conclusion it repeated.
    renumber: Dict[str, str] = {}
    position = 0
    for owner, body in zip(owners, trimmed):
        if owner is not None and body.strip():
            position += 1
            renumber[owner] = str(position)
    for dropped, source in alias.items():
        if source in renumber:
            renumber[dropped] = renumber[source]

    parts: List[str] = []
    for head, owner, body in zip(heads, owners, trimmed):
        if not body.strip():
            continue
        if head and owner in renumber:
            head = f"Step {renumber[owner]}:"
        parts.append(f"{head} {body}" if head else body)
    rewritten = "\n".join(parts)

    if any(old_no != new_no for old_no, new_no in renumber.items()):
        def redirect(m: re.Match) -> str:
            return m.group(1) + renumber.get(m.group(2), m.group(2))
        rebuilt = [piece if _STEP.fullmatch(piece or "")
                   else _STEP_REF.sub(redirect, piece or "")
                   for piece in _STEP.split(rewritten)]
        rewritten = "".join(rebuilt)
        # Two citations that now point at the same step read as "Step 2 and
        # Step 2"; say it once.
        rewritten = re.sub(r"(?i)\b(Steps?\s*(\d+))(\s*(?:and|,)\s*Steps?\s*\2\b)+",
                           r"\1", rewritten)

    if extractor is not None and extractor(rewritten) != extractor(text):
        return text
    return rewritten


def dedup_steps(steps: List[str], threshold: float = 0.75,
                extractor: Optional[Callable[[str], object]] = None) -> List[str]:
    """The same pass over a trace already split into steps.

    The ROSCOE export joins a trace's steps with a space before scoring, which
    leaves neither the "Step N:" line starts nor the blank lines dedup_trace
    reads, so the export deduplicates the list and joins what comes back.
    """
    # Numbered from one, because a citation inside a step reads "From Step 1"
    # and the redirect has to match it. Numbering these from zero sent every
    # citation one step forward, so the first step came to cite the second.
    owners = [str(i + 1) for i in range(len(steps))]
    trimmed, alias = _trim(list(steps), owners, threshold)
    for owner, source in _repeat_conclusions(trimmed, owners).items():
        alias.setdefault(owner, source)
        trimmed[int(owner) - 1] = ""
    alias = _resolve(alias)
    out = [b for b in trimmed if b.strip()]
    if not out:
        return list(steps)
    renumber: Dict[str, str] = {}
    position = 0
    for owner, body in zip(owners, trimmed):
        if body.strip():
            position += 1
            renumber[owner] = str(position)
    for dropped, source in alias.items():
        if source in renumber:
            renumber[dropped] = renumber[source]
    if any(old_no != new_no for old_no, new_no in renumber.items()):
        def redirect(m: re.Match) -> str:
            return m.group(1) + renumber.get(m.group(2), m.group(2))
        out = [_STEP_REF.sub(redirect, b) for b in out]
    if extractor is not None and extractor(" ".join(out)) != extractor(" ".join(steps)):
        return list(steps)
    return out
