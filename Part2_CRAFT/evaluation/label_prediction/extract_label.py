#!/usr/bin/env python3
"""Reading the answer a trace states — the one copy of it.

Both domains live here because both are the same job: find where a trace
commits to an answer and return it as written. A logical trace commits with
__PROVED__ / __DISPROVED__, a maths trace with \\boxed{}; nothing else about
them differs, and splitting them into separate modules is what let three
copies of this code drift apart in the first place.

There were three. This module, the pipeline's own evaluation and the baseline
runner each carried their own extract_label and extract_math_answer, and on
6000 saved traces they disagreed: 8 logical traces got different labels — one
pair opposite, __PROVED__ against __DISPROVED__ — and 5 maths traces gave an
answer under one copy and none under the other. CRAFT and the baselines were
being read by different rules and compared in the same table.

The rules kept here are the fuller ones: a label is also read from "Final
Conclusion: PROVED" without the underscores, the natural-language fallback
sees five lines rather than three, and it falls back once more to the whole
trace, taking whichever of proved/disproved appears last.

What is deliberately *not* here is normalise_math_answer. It rewrote an answer
before anything compared it, and the rewriting lost mathematics: it stripped
every LaTeX command, so \\lceil n/2 \\rceil + 1 became n/2 + 1 and 3\\pi/4
became 0.75, and it flattened \\sqrt2 to 2, which then matched a gold answer
of 2 and scored a wrong answer correct. An answer is returned as the trace
wrote it and answers_match decides equivalence, which it does symbolically and
without discarding operators.

_normalise_single survives only as a predicate: extract_math_answer uses it to
ask whether a bare line looks like an answer at all, and returns the line
itself, not the normalised form.
"""

from __future__ import annotations

import re
from typing import List, Optional

VALID_LABELS = {"__PROVED__", "__DISPROVED__"}

_LABEL_RE      = re.compile(r"(__PROVED__|__DISPROVED__)", re.IGNORECASE)
_NL_DISPROVED  = re.compile(r"\b(disproved|false|incorrect|refuted)\b", re.IGNORECASE)
_NL_PROVED     = re.compile(r"\b(proved|proven|true|correct|holds)\b", re.IGNORECASE)


def _extract_boxed_content(text: str) -> list:
    """Extract \\boxed{...} contents handling nested braces."""
    results = []
    i = 0
    while i < len(text):
        idx = text.find('\\boxed{', i)
        if idx == -1:
            break
        start = idx + 7
        depth, j = 1, start
        while j < len(text) and depth > 0:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1
        if depth == 0:
            results.append(text[start:j - 1].strip())
        i = j
    return results


_HASH_ANS_RE   = re.compile(r"####\s*(.+)")
_FINAL_ANS_RE  = re.compile(
    r"(?:the\s+answer\s+is|final\s+answer\s*[:\=]|answer\s*[:\=])\s*([^\n\.]+)", re.IGNORECASE
)


def normalise_label(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    label = label.strip().upper()
    if label in VALID_LABELS:
        return label
    if "DISPROVED" in label:
        return "__DISPROVED__"
    if "PROVED" in label:
        return "__PROVED__"
    return {"TRUE": "__PROVED__", "FALSE": "__DISPROVED__"}.get(label)


def extract_label(text: str) -> Optional[str]:
    """Extract __PROVED__ or __DISPROVED__ from text."""
    if not text:
        return None
    # Primary: exact __PROVED__ or __DISPROVED__ format (last occurrence)
    matches = _LABEL_RE.findall(text)
    if matches:
        m = matches[-1].upper()
        return m if m in ("__PROVED__", "__DISPROVED__") else None
    # Secondary: markdown bold without underscores: **PROVED**, **DISPROVED**
    bold_matches = re.findall(r"\*\*(PROVED|DISPROVED)\*\*", text, re.I)
    if bold_matches:
        b = bold_matches[-1].upper()
        return {"PROVED": "__PROVED__", "DISPROVED": "__DISPROVED__"}.get(b)
    # Tertiary: look for "Final Conclusion: PROVED/DISPROVED" (without underscores)
    concl_m = re.search(r"(?:final\s+conclusion|conclusion)\s*[:\-]\s*(PROVED|DISPROVED)", text, re.I)
    if concl_m:
        return {"PROVED": "__PROVED__", "DISPROVED": "__DISPROVED__"}.get(concl_m.group(1).upper())
    # Fallback: search last 5 lines for natural language signals
    lines = [l.strip() for l in text.splitlines() if l.strip() and "[Step" not in l]
    tail = " ".join(lines[-5:]) if len(lines) >= 5 else " ".join(lines)
    if _NL_DISPROVED.search(tail):
        return "__DISPROVED__"
    if _NL_PROVED.search(tail):
        return "__PROVED__"
    # Last resort: scan the full text for natural language signals (last occurrence)
    all_disproved = list(re.finditer(r"\b(disproved|false|incorrect|refuted|cannot be proved|not proved|hypothesis is not)\b", text, re.I))
    all_proved = list(re.finditer(r"\b(proved|proven|true|correct|holds|hypothesis is proved|hypothesis holds)\b", text, re.I))
    if all_disproved and all_proved:
        # Take whichever appears last
        if all_disproved[-1].start() > all_proved[-1].start():
            return "__DISPROVED__"
        return "__PROVED__"
    if all_disproved:
        return "__DISPROVED__"
    if all_proved:
        return "__PROVED__"
    return None


def majority_vote(labels: List[Optional[str]]) -> Optional[str]:
    valid = [l for l in labels if l in VALID_LABELS]
    return Counter(valid).most_common(1)[0][0] if valid else None


def extract_math_answer(text: str) -> Optional[str]:
    """Extract numeric answer: \\boxed{} → #### N → 'the answer is N'.
    Commas are preserved (multi-answer support for OlympiadBench).
    """
    if not text:
        return None
    m = _extract_boxed_content(text)
    if m:
        raw = m[-1].strip()
        # Remove thousands-separator commas only (e.g. 1,000 → 1000)
        raw = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', raw)
        return raw
    m2 = _HASH_ANS_RE.findall(text)
    if m2:
        return m2[-1].strip()
    m3 = _FINAL_ANS_RE.findall(text)
    if m3:
        return m3[-1].strip()
    # Bare-number / short-answer fallback (USC selector, bare responses, OlympiadBench format)
    for candidate in [text.strip(), text.strip().splitlines()[-1].strip() if text.strip() else ""]:
        if candidate and len(candidate) <= 60:
            norm = _normalise_single(candidate)
            if norm is not None:
                return candidate
    return None


def _eval_latex_frac(text: str) -> str:
    """Evaluate \\frac / \\tfrac / \\dfrac / \\cfrac{a}{b} to a decimal string."""
    def _replace(m: re.Match) -> str:
        try:
            n, d = float(m.group(1).strip()), float(m.group(2).strip())
            if d == 0:
                return m.group(0)
            r = n / d
            return str(int(r)) if r == int(r) else str(round(r, 8)).rstrip("0").rstrip(".")
        except (ValueError, TypeError):
            return m.group(0)
    return re.sub(r'\\(?:[tdc])?frac\{([^}]+)\}\{([^}]+)\}', _replace, text)


def _normalise_single(s: str) -> Optional[str]:
    """Normalize one numeric/expression token."""
    s = s.strip().strip("$").strip()
    if not s:
        return None
    # Normalize Unicode minus/dash variants → ASCII minus
    s = s.replace('\u2013', '-').replace('\u2014', '-').replace('\u2212', '-')  # en-dash, em-dash, minus sign
    # Strip leading LaTeX spacing artifacts (e.g. \; or \, at start after multi-answer split)
    s = re.sub(r'^[;\\]+\s*', '', s)
    s = _eval_latex_frac(s)
    s = re.sub(r'\^?\{?\\circ\}?|°|\\degree', '', s)   # degree symbols
    s = re.sub(r'\\(?:text|mathrm|mbox)\{[^}]*\}', '', s)
    s = re.sub(
        r'\s*\b(?:km|cm|mm|m|kg|g|mg|l|ml|hours?|hrs?|minutes?|mins?|seconds?|secs?|'
        r'days?|weeks?|months?|years?|dollars?|cents?|degrees?)\b.*$',
        '', s, flags=re.IGNORECASE,
    )
    s = re.sub(r'\\[a-zA-Z]+\*?', '', s)
    s = re.sub(r'[{}]', '', s)
    s = s.strip().lstrip("$\\").strip(";")
    if s.endswith('%'):
        s = s[:-1]
    s = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', s)   # thousands separators
    # Normalize internal whitespace in algebraic expressions (e.g. "2 n" → "2n", "2rc + r + c" → "2rc+r+c")
    s = re.sub(r'\s+', '', s)
    s = s.strip().lower()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, OverflowError):
        pass
    # Evaluate plain a/b fractions (e.g. "505/8076" → decimal)
    _frac_m = re.match(r'^(-?\d+\.?\d*)\s*/\s*(-?\d+\.?\d*)$', s)
    if _frac_m:
        try:
            _n, _d = float(_frac_m.group(1)), float(_frac_m.group(2))
            if _d != 0:
                _r = _n / _d
                return str(int(_r)) if _r == int(_r) else str(round(_r, 8)).rstrip("0").rstrip(".")
        except (ValueError, ZeroDivisionError):
            pass
    # Only return non-numeric strings that look like math expressions.
    # Reject English prose (e.g. step titles like "step 8: state the result.")
    if s and not re.search(r'[a-z]{4,}', s):
        return s
    return None



def extract_pred(text: str, domain: str = "logical") -> Optional[str]:
    """The answer the trace states, as written. Comparison happens later."""
    if domain == "math":
        return extract_math_answer(text)
    return extract_label(text)


# ---------------------------------------------------------------------------
# Grouping, not scoring
# ---------------------------------------------------------------------------
# normalise_math_answer collapses an answer to a canonical string so that two
# spellings of one answer land in the same bucket. It is lossy on purpose and
# must never decide whether an answer is right: it strips LaTeX commands, so
# \\sqrt2 becomes 2 and \\lceil n/2 \\rceil becomes n/2. Use it to group or to
# display; use answers_match to compare.

def normalise_math_answer(ans: Optional[str]) -> Optional[str]:
    """Normalize a math answer.
    Handles GSM8K (plain integers), OlympiadBench (fractions, degrees,
    multi-answer tuples), and general LaTeX formatting.
    """
    if ans is None:
        return None
    ans = str(ans).strip()
    # Strip matching outer dollar signs
    if ans.startswith('$') and ans.endswith('$') and len(ans) > 2:
        inner = ans[1:-1].strip()
        if not inner.startswith('$'):
            ans = inner
    # Normalize Unicode minus/dash variants → ASCII minus (before split)
    ans = ans.replace('\u2013', '-').replace('\u2014', '-').replace('\u2212', '-')
    # Normalize LaTeX spacing commands used as multi-answer separators: \, \; \quad → comma
    ans = re.sub(r'\\[,;]', ',', ans)
    ans = re.sub(r'\\quad\s*', ',', ans)
    ans = re.sub(r',+', ',', ans)          # collapse duplicate commas
    ans = ans.strip(',')
    # Strip outer parentheses or brackets enclosing a tuple: (1,2,3) → 1,2,3
    if ((ans.startswith('(') and ans.endswith(')')) or
            (ans.startswith('[') and ans.endswith(']'))):
        inner = ans[1:-1].strip()
        if inner:
            ans = inner
    # Comma-separated multi-answer (e.g. OlympiadBench "-1,-9" or "0.5,3.56")
    if ',' in ans:
        parts = [p.strip() for p in ans.split(',') if p.strip()]
        if len(parts) >= 2:
            norm_parts = [_normalise_single(p) for p in parts]
            # Partial match OK: if most parts normalize, sort and compare those
            valid_parts = [p for p in norm_parts if p is not None]
            if valid_parts and len(valid_parts) == len(norm_parts):
                try:
                    sorted_parts = sorted(valid_parts, key=float)
                except (ValueError, TypeError):
                    sorted_parts = sorted(valid_parts)
                return ','.join(sorted_parts)
    return _normalise_single(ans)
