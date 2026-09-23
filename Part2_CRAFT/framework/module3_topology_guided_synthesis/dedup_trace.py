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
# A reference to one of this trace's own steps, which renumbering may rewrite.
# Requiring the wording that makes it a reference keeps renumbering off the
# problem's own text: an OlympiadBench question describes an algorithm whose
# "Step 1" is Ada adding x and y, and a trace that quotes it would otherwise
# have that step silently renumbered along with the citations.
_STEP_REF = re.compile(
    r"(?i)(?:"
    # a reference word before it: "from Step 3", "established in Step 4"
    r"(?P<pre>\b(?:from|in|into|at|by|per|see|using|used|use|with|of|to|and|or|"
    r"than|versus|vs\.?|since|because|,|;|\()\s*(?:the\s+)?"
    r"(?:result|results|conclusion|premise|premises|prerequisite|prerequisites)?"
    r"\s*(?:established|derived|shown|obtained|stated)?\s*(?:in|at|by)?\s*"
    r"\bSteps?\s*)(?P<n1>\d+)"
    r"|"
    # a reference verb after it: "Step 3 establishes that ...", or the start
    # of a run that ends in one: "Step 5, Step 2, and Step 4 establish that".
    # The first step of such a run has only a comma after it, and a citation
    # left unmatched keeps its old number -- which, once the steps around it
    # are renumbered, can be the number of the step citing it. The run has to
    # end in the verb: "Ada performs Step 1, Step 2, and Step 3" is the
    # problem's own text and names no step of the trace.
    r"(?P<pre2>\bSteps?\s*)(?P<n2>\d+)"
    r"(?=(?:\s*,?\s*(?:and\s+)?Steps?\s*\d+)*\s*(?:establish(?:es|ed)?|show(?:s|ed)?|"
    r"give(?:s)?|gave|derive(?:s|d)?|state(?:s|d)?|tell(?:s)?|told|yield(?:s|ed)?|"
    r"provide(?:s|d)?)\b)"
    r")")


def _ref_parts(m: "re.Match"):
    """(prefix, number) for whichever alternative matched."""
    if m.group("n1") is not None:
        return m.group("pre"), m.group("n1")
    return m.group("pre2"), m.group("n2")
# The looser form, for counting and for the "Step 2 and Step 2" cleanup.
_STEP_REF_ANY = re.compile(r"(?i)(\bSteps?\s*)(\d+)")
_PARA = re.compile(r"\n\s*\n")
_SENT = re.compile(r"(?<=[.!?])\s+")
# A subscripted symbol is one name: split c_a into "c" and "a" and it stops
# being distinguishable from d_a whenever the sentence also mentions c on
# its own, which is how "Thus, $c_a = c$" was read as a repeat of
# "Thus, $d_a = c$".
_WORD = re.compile(r"[A-Za-z\\]+(?:_\{?[A-Za-z0-9]+\}?)?|\d+")

# A display is balanced by construction, so one can be lifted out of a
# paragraph the paragraph rule has to keep whole. That is where most of what
# survives sentence deduplication sits: a trace that derives a_n once quotes
# the formula again in every step that uses it -- four copies of the same two
# lines, each wrapped in different prose, so no two paragraphs match. The
# prose already says which step it came from, so the second copy onwards is
# dropped and the citation left standing. What counts as a display, in either
# convention a model writes, is math_text's to say.
import sys as _sys_mt
import pathlib as _pl_mt
_sys_mt.path.insert(0, str(_pl_mt.Path(__file__).resolve().parents[2]))
from framework.domain_optimization.math_text import (  # noqa: E402
    DISPLAY_RE as _DISPLAY, balanced as _balanced, has_display as _has_display)

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

# Below this a repeated line is a connective -- "Therefore:", "So we have" --
# that two steps may legitimately share.
_VERBATIM_MIN = 25

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


# Markdown emphasis is not part of a sentence: "**if someone is blue then they
# eat the cow**" and "if someone is blue then they eat the cow" are the same
# line, and the asterisks kept them apart under both the verbatim and the
# sequence tests.
def _plain(text: str) -> str:
    return text.replace("*", "")


# A rule stated twice under the same name. The traces restate a fact where
# they apply it -- "According to Fact 3, the nitrobacteria is quadrupedal, and
# the nitrobacteria does not vary hackee ..." in one step and "... and it does
# not vary hackee ..." in a later one -- and the pronoun keeps the pair under
# the sequence ratio while adding nothing. The same fact number and no new
# content word is what makes the second a restatement, whatever the ratio.
_RULE_OPEN = re.compile(r"(?i)^\s*(?:according to|by|per|from|using)\s+(Fact\s*\d+)\b")


def _same_rule(candidate: str, source: str) -> bool:
    a, b = _RULE_OPEN.match(candidate), _RULE_OPEN.match(source)
    if not a or not b or a.group(1).replace(" ", "").lower() != b.group(1).replace(" ", "").lower():
        return False
    return not (_content(candidate) - _content(source) - _RECAP_WORDS)


def _repeats(candidate: str, source: str, toks: frozenset,
             prev: frozenset, threshold: float) -> bool:
    if len(toks & prev) / max(1, len(toks | prev)) <= threshold:
        return False
    if _content(candidate) - _content(source):
        return False
    return difflib.SequenceMatcher(None, _plain(candidate), _plain(source)).ratio() > _VERBATIM


def _is_prose(text: str) -> bool:
    """A paragraph with no display maths in it can be split a sentence at a time."""
    return not (_has_display(text) or "\\boxed{" in text)


def _drop_verbatim(bodies: List[str], owners: List[Optional[str]]
                   ) -> Tuple[List[str], Dict[str, str]]:
    """Remove sentences the trace writes twice, character for character.

    ROSCOE's repetition scores are (1 - max over every pair of sentences), so a
    single pair of identical sentences puts the trace at the worst score the
    metric has and no amount of other deduplication moves it. One trace in five
    has such a pair, and among the maths cells it is one in three.

    Nothing about this pass is lossy -- what it removes is a character-for-
    character copy of a line the trace still has -- so it runs over every
    paragraph, including the ones carrying displays that _trim has to leave
    whole. What keeps it safe there is that a sentence whose own delimiters do
    not balance is never removed: those are the fragments a split lands inside
    \\begin{cases}, and dropping one leaves the rest of the trace short a brace.

    The first copy stays, since that is where the line was derived. A copy
    stating the answer is the exception: the last one stays, so the trace still
    ends on its answer and extract_label reads what it always read.

    A step left empty by this is reported with the step its line came from, so
    that what cited it can be sent there instead of being restored.
    """
    counts: Dict[str, int] = {}
    for body in bodies:
        for piece in _SENT.split(body or ""):
            key = " ".join(_plain(piece).split())
            if len(key) >= _VERBATIM_MIN and _balanced(piece):
                counts[key] = counts.get(key, 0) + 1
    repeated = {k for k, n in counts.items() if n > 1}
    if not repeated:
        return list(bodies), {}

    kept_once: Dict[str, Optional[str]] = {}
    seen_so_far: Dict[str, int] = {}
    alias: Dict[str, str] = {}
    out: List[str] = []
    for body, owner in zip(bodies, owners):
        pieces = _SENT.split(body or "")
        survivors: List[str] = []
        for piece in pieces:
            key = " ".join(_plain(piece).split())
            if key not in repeated:
                survivors.append(piece)
                continue
            seen_so_far[key] = seen_so_far.get(key, 0) + 1
            if _ANSWER.search(piece):
                # keep the last
                if seen_so_far[key] == counts[key]:
                    survivors.append(piece)
            else:
                # keep the first
                if key not in kept_once:
                    kept_once[key] = owner
                    survivors.append(piece)
                elif owner is not None and kept_once[key] is not None:
                    alias.setdefault(owner, kept_once[key])
        out.append(" ".join(x for x in survivors if x.strip()))
    # Only a step this emptied needs redirecting; one that kept a line of its
    # own is still there to be cited.
    alias = {o: src for o, src in alias.items()
             if not out[owners.index(o)].strip()}
    return out, alias


# A condition the trace has already quoted, quoted again in a parenthetical:
# "By the problem's condition (“draw a convex n-gon whose vertices ...”)". The
# synthesis prompt asks each step to name the condition it uses in the
# problem's words, and a step that uses the same condition as the one before
# it writes the same words again -- one OmniMATH trace carries the same
# forty-word quotation in six of its eight steps. The first quotation is
# where the condition enters the proof and stays; a later one in a
# parenthetical is an aside by construction, so the sentence reads without
# it, and the noun it hangs on ("Fact 18", "the problem's condition") still
# says which condition is meant. Only the parenthetical form is touched: a
# quotation that is the object of its sentence -- "the hypothesis “X” is
# __DISPROVED__" -- is the claim, not a reminder of it.
_QUOTE = re.compile(r'[“"]([^”"]{20,}?)[”"]')
_PAREN_QUOTE = re.compile(r'\s*\(\s*[“"]([^”"]{20,}?)[”"]\s*\)')
# The same quotation in apposition to the noun that names it: "Using the
# given condition “the sum of the squares ...” from the setup", "by Fact 17
# (“If someone is big ...”)". The noun stays and the quotation goes, and only
# where the sentence carries on after it -- a quotation that ends its sentence
# would leave "According to Fact 12." standing on its own. A quotation that is
# what the sentence is about ("the hypothesis “X” is equivalent to “Y”") or
# is predicated of the noun ("the condition is “X”") is left alone: there the
# words are the claim. So is one that the sentence goes on to predicate
# something of -- "According to Fact 12, “X” reduces to ..." -- where the
# quotation is the subject of the verb that follows it; what may follow is a
# preposition carrying on the same phrase ("... “X” from the setup"), and
# nothing else.
_APPOS_QUOTE = re.compile(
    r'(?P<head>\b(?:Fact\s*\d+|(?:the\s+)?(?:problem[\'’]s\s+)?(?:given\s+|stated\s+)?'
    r'(?:winning\s+)?(?:condition|premise|rule|principle|constraint|definition)))'
    r'\s*[“"](?P<q>[^”"]{20,}?)[”"]'
    r'(?=\s+(?:from|together|with|for|to|in|on|at|by|of|under|over|across)\b)')


def _drop_requoted(bodies: List[str]) -> List[str]:
    """Remove a repeat quotation of a span an earlier step quoted."""
    quoted: set = set()
    out: List[str] = []
    for body in bodies:
        def paren(m: re.Match) -> str:
            key = " ".join(m.group(1).split())
            return "" if key in quoted else m.group(0)

        def appos(m: re.Match) -> str:
            key = " ".join(m.group("q").split())
            return m.group("head") if key in quoted else m.group(0)
        # Only a quotation that entered before this body counts as earlier:
        # collected after the substitution so a body's own first quotation
        # is not taken for a repeat of itself.
        new = _PAREN_QUOTE.sub(paren, body or "")
        new = _APPOS_QUOTE.sub(appos, new)
        quoted |= {" ".join(q.split()) for q in _QUOTE.findall(body or "")}
        out.append(new)
    return out


# Two steps in a row that open with the same clause: "Using the results
# $S_1 = 52$ from Step 3 and $S_2 = 12$ from Step 4, we substitute ..." and
# then "Using the results $S_1 = 52$ from Step 3 and $S_2 = 12$ from Step 4,
# we perform the subtraction ...". The second names its premises in the words
# the first just used, and the two sentences are the most alike pair in the
# trace. The opening comes off the second step, which then begins with what
# it does; the step before it still says where the values came from.
_LEAD = re.compile(r"^\s*([^,\n]{20,}?),\s+(?=[a-z])")

# The same condition named in full at the head of step after step: "Using the
# given condition that $n$ is a positive integer and $n^3+2n^2+9n+8$ is the
# cube of an integer, ..." opens six of an OmniMATH trace's eight steps. The
# first naming is where the condition enters; a later one is replaced by a
# back-reference, which says the same thing in the words the trace already
# used once. Only a clause that names a condition is touched, and only when
# the same words opened an earlier step.
_CONDITION_OPEN = re.compile(
    r"^\s*(?P<lead>(?:Using|By|From|Applying|Given|Under|With)\s+"
    r"(?:the\s+)?(?:problem[\'’]s\s+)?(?:given\s+|stated\s+)?condition(?:s)?\s+"
    r"(?:that\s+|[“\"]))(?P<cond>[^,\n]{15,}?)(?P<close>[”\"]?)\s*,\s+(?=[a-z])",
    re.IGNORECASE)


def _drop_repeated_condition(bodies: List[str], owners: List[Optional[str]]) -> List[str]:
    out: List[str] = []
    first_named: Dict[str, Optional[str]] = {}
    for body, owner in zip(bodies, owners):
        text = body or ""
        m = _CONDITION_OPEN.match(text)
        if m:
            key = " ".join(m.group("cond").split()).lower().rstrip(".,;")
            source = first_named.get(key)
            if source is not None:
                verb = m.group("lead").split()[0]
                text = f"{verb} the condition stated in Step {source}, " + text[m.end():]
            elif key not in first_named and owner is not None:
                first_named[key] = owner
        out.append(text)
    return out


def _drop_repeated_lead(bodies: List[str]) -> List[str]:
    out: List[str] = []
    prev_lead: Optional[str] = None
    for body in bodies:
        text = body or ""
        m = _LEAD.match(text)
        lead = " ".join(m.group(1).split()) if m else None
        if (lead is not None and lead == prev_lead
                and len(lead.split()) >= 6 and _balanced(m.group(1))):
            rest = text[m.end():]
            if len(rest.split()) >= 5:
                text = rest[0].upper() + rest[1:]
        prev_lead = lead
        out.append(text)
    return out


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
                if _is_recap(piece, [text for _, _, text in seen]):
                    continue
                match = next((src for prev, src, prev_text in seen
                              if _repeats(piece, prev_text, toks, prev, threshold)
                              or _same_rule(piece, prev_text)),
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
# "derive", "deduce", "establish" and "show" announce a derivation as plainly
# as "infer" does, and the traces use them interchangeably: "we derive that
# the cow sees the squirrel" in one step and "we infer that the cow sees the
# squirrel" in the next were two derivations of one line until the first
# verb was read.
_DERIVES = re.compile(
    r"(?is)\b(?:infer(?:\s+that)?|conclude(?:\s+that)?|it follows that|"
    r"we (?:can\s+)?(?:derive|deduce|establish|show)(?:\s+that)?|"
    r"(?:this|which|that)\s+(?:implies|means|shows|establishes)(?:\s+that)?|"
    r"therefore|thus|hence|which gives(?:\s+us)?|this gives|yields?|"
    r"we (?:get|obtain|find)|to get|resulting in|so that)\b"
    r"[,:\s]*(.+?)(?=[.;]|$)")

# A sentence that only says again what an earlier step established. The
# synthesis prompt forbids it in so many words and the traces write it anyway,
# in the shape "We established in Step 1 that X" or "As shown in Step 3, X".
# It repeats a step rather than the problem, so it earns nothing under the
# faithfulness scores and costs under the repetition ones.
_RECAP_OPEN = re.compile(
    r"(?i)^\s*(?:as\s+)?(?:we\s+)?(?:have\s+)?"
    r"(?:established|shown|derived|determined|found|noted|recalled|seen)\s+"
    r"(?:above\s+|earlier\s+|previously\s+)?"
    r"(?:in|from|at)\s+Steps?\s*\d+|"
    r"(?i)^\s*(?:as|from|per)\s+(?:established\s+in\s+|shown\s+in\s+|"
    r"derived\s+in\s+)?Steps?\s*\d+\s*,?\s*we\s+"
    r"(?:have|know|established|derived|obtained)\b")


# The wording a recap is made of. Left in, "established" and "in" count as
# content the earlier step lacked, and every recap looks like new work.
_RECAP_WORDS = frozenset("""
as we have has had been being in from at of to that this it its and or but so
then now here there also again both each all any some no not is are was were
be by with which what where when who whom whose for on onto into over under
established shown derived determined found noted recalled seen show give gives
given know known step steps fact facts rule rules above earlier previously
therefore thus hence since because recall note
""".split())


def _substance(text: str) -> frozenset:
    """The content words of a sentence, with the connective wording removed."""
    return frozenset(w for w in _content(text) if w not in _RECAP_WORDS)


def _is_recap(piece: str, earlier: List[str]) -> bool:
    """Does this sentence only repeat what an earlier step already said?

    A sentence opening "as established in Step 3" and then going on to do
    something is not a recap; one that opens that way and adds nothing is. The
    difference is whether any of its substance is new, so that is what is
    checked -- against each earlier sentence on its own, since a recap names
    one step.
    """
    if not _RECAP_OPEN.search(piece) or _ANSWER.search(piece):
        return False
    body = _substance(piece)
    if len(body) < 2:
        return False
    return any(not (body - _substance(prev)) for prev in earlier)
# Wording that carries no claim, so two steps sharing it are not the same step.
# The value a maths step boxes, taken apart from the words around it.
_BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
# A modal or an intensifier changes how firmly a conclusion is put, not what
# it says: "the diffuser must be antipollution" and "the diffuser is
# antipollution" are one conclusion, and were two until "must" was filler.
_FILLER = re.compile(r"\b(the|a|an|that|this|is|are|be|we|it|to|of|and|then|"
                     r"so|now|next|finally|therefore|thus|hence|"
                     r"must|should|will|would|can|could|may|might|shall|"
                     r"indeed|necessarily|clearly|certainly|also|still|already|"
                     r"in fact|as well|such|answer|final|namely|exactly|precisely)\b")


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
    # Two words is the floor, not three: the connectives are already gone, so
    # what is left is substance, and a logical conclusion is routinely two
    # words of it -- "infer that the woof is scarred" reduces to "woof
    # scarred". At three, conclusions went unread on half the steps of an FLD
    # trace, and two steps applying the same rule to the same premise could
    # not be seen to agree.
    # A stated answer is exempt from the word count: "\\boxed{\\frac{2}{9}}"
    # is two words and identifies the claim exactly, and a trace that derives
    # the same boxed value twice is the repetition this is here to remove.
    if "\\boxed" in text or "__proved__" in text or "__disproved__" in text:
        return text if len(text) >= 8 else None
    return text if len(text) >= 10 and len(text.split()) >= 2 else None


def _repeat_conclusions(bodies: List[str],
                        owners: List[Optional[str]]) -> Dict[str, str]:
    """Map each step that re-derives an earlier conclusion to that earlier step.

    The final step is mapped like any other. Sparing it looked like the careful
    choice -- it is where extract_label reads the answer -- but the traces put
    their repeats at the end: a step reaching the same conclusion off a
    different premise, "From Step 2 ... infer the woof is scarred" against
    "From Step 3 ... infer the woof is scarred". Sparing the last one left
    every one of those standing, and they are what ROSCOE's repetition score,
    a maximum over sentence pairs, ends up measuring. Since the two conclusions
    are the same string, dropping the later one leaves the step before it
    saying what it said, and the reader reads the same answer; dedup_trace's
    guard checks exactly that before keeping the rewrite.
    """
    first_seen: Dict[str, str] = {}
    boxed_seen: Dict[str, Tuple[str, str]] = {}
    repeats: Dict[str, str] = {}
    for body, owner in zip(bodies, owners):
        if owner is None or not body.strip():
            continue
        claim = _conclusion(body)
        if claim is None:
            continue
        hit = first_seen.get(claim)
        # Two steps can word the same landing differently -- "the final value of
        # the expression is \\boxed{46}" against "the value is \\boxed{46}" --
        # and the strings then miss each other. Matching on the boxed value
        # alone catches those, but it also catches a step that merely mentions
        # the same value while doing something else: deriving tan from an
        # equation the step before rearranged, or checking the result against
        # the problem's constraints. So the value has to agree AND the later
        # step has to say nothing the earlier one did not.
        boxed = _BOXED.findall(claim)
        if hit is None and boxed:
            key = "\\boxed{" + boxed[-1].strip() + "}"
            cand = boxed_seen.get(key)
            if cand is not None and not (_substance(claim) - _substance(cand[1])):
                hit = cand[0]
        if hit is not None:
            repeats[owner] = hit
        else:
            first_seen.setdefault(claim, owner)
            if boxed:
                boxed_seen.setdefault(
                    "\\boxed{" + boxed[-1].strip() + "}", (owner, claim))
    return repeats


# A citation that says what the step it names established: "Step 4 establishes
# that the tiger sees the tiger". The claim is checkable against that step, and
# in these traces it is wrong about one time in twenty -- the trace cites the
# step before the one that actually derived the line, and reads as rigorous
# while resting on nothing.
_CITE_CLAIM = re.compile(
    r"(?i)\b(Steps?\s*)(\d+)(\s*(?:establish(?:es|ed)?|show(?:s|ed)?|"
    r"give(?:s)?|gave|derive(?:s|d)?|state(?:s|d)?|as the premise that|"
    r"we (?:have|know|established) that)\s*(?:that\s+)?)([^,.;]{10,140})")


def _fix_citations(text: str, bodies: List[str],
                   owners: List[Optional[str]],
                   here: Optional[str] = None) -> str:
    """Point a citation at the step that actually carries what it claims.

    Only the number is touched, and only when the step named does not contain
    the claim and exactly one earlier step does -- an ambiguous case is left
    alone, since guessing between two candidates would be inventing a
    derivation rather than repairing a reference.
    """
    by_owner = {o: b for o, b in zip(owners, bodies) if o is not None}
    if not by_owner:
        return text

    def repair(m: re.Match) -> str:
        named, claim = m.group(2), m.group(4)
        want = _substance(claim)
        if len(want) < 2:
            return m.group(0)
        holds = [o for o, b in by_owner.items()
                 if not (want - _substance(b))]
        # A step cannot stand on itself, so a self-citation is repaired even
        # when the step does contain the claim -- it is citing the line it is
        # about to write. Otherwise the named step has to be wrong about the
        # claim before anything moves.
        if here is not None and named == here:
            holds = [o for o in holds if o != here]
        elif named in holds:
            return m.group(0)
        if len(holds) != 1:
            return m.group(0)
        return m.group(1) + holds[0] + m.group(3) + m.group(4)

    return _CITE_CLAIM.sub(repair, text)


# A run of citations, as a step writes them: "From Step 6, Step 1, and Step 1".
# Redirecting can send two of them to the same step, and a list that names one
# step twice is both wrong and ungrammatical -- the CoLA model that scores these
# traces puts "From Step6, Step1, and Step1, we have ..." at 0.15 where the
# trace averages 0.77.
_CITE_RUN = re.compile(
    # "Step 6, Step 1, and Step 1" and the plural form that names the word
    # once, "Steps 5 and 7" -- both come out of redirecting, and both can end
    # up naming one step twice.
    r"(?i)\bSteps?\s*\d+(?:\s*(?:,|and|,\s*and)\s*(?:Steps?\s*)?\d+)+")
# Every number in such a run, whether or not "Step" precedes it: the plural
# form writes the word once and then bare numbers, "Steps 5 and 7".
_CITE_ONE = re.compile(r"\d+")


# A step whose body opens by naming itself again: "Step 8: Step 8 (final): From
# Steps 5 and 7 ...". The header is already there, so the repeat is noise, and
# the CoLA model that scores these traces puts such a sentence at 0.04 where the
# trace averages 0.79.
_ECHOED_HEAD = re.compile(r"(?i)^\s*\**\s*Steps?\s*\d+\s*(?:\([^)]{0,20}\))?\s*\**\s*[:.\-]\s*")


def _drop_echoed_header(body: str) -> str:
    """Remove a step number the body repeats after its own header."""
    return _ECHOED_HEAD.sub("", body, count=1)


def _tidy_citation_lists(text: str) -> str:
    """Drop repeats from a run of step citations and space them properly.

    A run naming distinct steps is returned exactly as written. Rewriting one
    risks dropping a citation, and there is nothing there to fix.
    """
    def fix(m: re.Match) -> str:
        found = _CITE_ONE.findall(m.group(0))
        seen: List[str] = []
        for n in found:
            if n not in seen:
                seen.append(n)
        if len(seen) == len(found):
            return m.group(0)
        parts = [f"Step {n}" for n in seen]
        if len(parts) == 1:
            return parts[0]
        return ", ".join(parts[:-1]) + (" and " if len(parts) == 2 else ", and ") + parts[-1]
    return _CITE_RUN.sub(fix, text)


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
    # A citation is repaired against the trace as it came in, before anything
    # is dropped or renumbered, so the numbers it is checked against are the
    # ones it was written with.
    source_bodies = list(bodies)
    bodies = [_fix_citations(b, source_bodies, owners, o)
              for b, o in zip(bodies, owners)]

    # Before the sentence rules, so that two sentences differing only in a
    # re-quoted parenthetical meet them as the repeat they are.
    bodies = _drop_requoted(bodies)
    bodies = _drop_repeated_condition(bodies, owners)
    bodies = _drop_repeated_lead(bodies)
    bodies, verbatim_alias = _drop_verbatim(bodies, owners)
    trimmed, alias = _trim(bodies, owners, threshold)
    for owner, source in verbatim_alias.items():
        alias.setdefault(owner, source)

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
        if head:
            if owner in renumber:
                head = f"Step {renumber[owner]}:"
            body = _drop_echoed_header(body)
        parts.append(f"{head} {body}" if head else body)
    rewritten = "\n".join(parts)

    if any(old_no != new_no for old_no, new_no in renumber.items()):
        # Renumbering can land a citation on the step that carries it: a step
        # citing one that was dropped is sent to the step it repeated, and that
        # step may be this one under its new number. A step cannot stand on
        # itself, so such a citation keeps the number it had.
        live = {n for n in renumber.values()}

        def redirect_for(current: Optional[str]):
            def redirect(m: re.Match) -> str:
                pre, old_no = _ref_parts(m)
                new_no = renumber.get(old_no, old_no)
                if current is None or new_no != current:
                    return pre + new_no
                # The citation would name the step carrying it. Keeping the
                # number it had only works if that number still exists; where
                # it does not, the nearest surviving earlier step is the one
                # this step was built on.
                if old_no in live:
                    return m.group(0)
                earlier = [n for n in live if n.isdigit() and int(n) < int(current)]
                if not earlier:
                    return m.group(0)
                return pre + max(earlier, key=int)
            return redirect

        rebuilt = []
        current: Optional[str] = None
        for piece in _STEP.split(rewritten):
            if _STEP.fullmatch(piece or ""):
                found = _STEP_NO.match(piece or "")
                current = found.group(1) if found else None
                rebuilt.append(piece)
            else:
                rebuilt.append(_STEP_REF.sub(redirect_for(current), piece or ""))
        rewritten = "".join(rebuilt)
        pass

    # A run of citations can name one step twice whether or not anything was
    # renumbered, so this runs on every trace.
    rewritten = _tidy_citation_lists(rewritten)

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
    stripped, verbatim_alias = _drop_verbatim(list(steps), owners)
    trimmed, alias = _trim(stripped, owners, threshold)
    for owner, source in verbatim_alias.items():
        alias.setdefault(owner, source)
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
            pre, old_no = _ref_parts(m)
            return pre + renumber.get(old_no, old_no)
        out = [_STEP_REF.sub(redirect, b) for b in out]
    if extractor is not None and extractor(" ".join(out)) != extractor(" ".join(steps)):
        return list(steps)
    return out
