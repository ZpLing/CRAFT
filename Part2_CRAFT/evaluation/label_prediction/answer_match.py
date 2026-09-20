#!/usr/bin/env python3
"""Deciding whether a predicted answer is the gold answer.

Two things were wrong with doing this by string comparison, which is what both
scorers did before this module existed.

The first is that it cannot be right. `\\frac{1}{2}`, `1/2` and `0.5` are the same
answer, and no amount of string normalisation makes them the same string without
also collapsing answers that differ. The old code chased this by rewriting the
text until it looked canonical, and on OlympiadBench's gold answers it either
gave up (`\\frac{1}{40}` normalised to None in Part 2's copy — 79 of the 424
Numerical answers were unparseable, so a correct prediction was scored wrong) or
produced something that was not the answer (`\\frac{2}{7}\\sqrt{53}` became
"0.28571429 53" in the baselines' copy, the square root silently dropped).

The second is that there were two copies of it, and they had drifted: on 127 of
OlympiadBench's 500 gold answers they disagreed, so a baseline's prediction and
CRAFT's prediction were not being measured by the same rule.

This module is the one copy, and it compares meanings rather than spellings: the
two answers are parsed into SymPy objects and tested for symbolic equivalence.
Set-valued, tuple-valued and interval answers are compared as the objects they
are, so `(3,2),(-3,2)` matches the same pair of points written in the other
order, and does not match `(3,2),(3,-2)`.

Datasets differ in what an answer is, so the entry point dispatches on the
dataset: GSM8K's answers are integers and are compared as integers, FLD's and
FOLIO's are labels and are compared as labels, and OlympiadBench's carry an
`answer_type` that says which of the four shapes to expect.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any, List, Optional, Sequence, Tuple

import sympy
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

# Numeric answers are compared with a tolerance: gold is often a rounded decimal
# ("0.286") where the exact value is a fraction, and demanding exact equality
# would mark the exact answer wrong.
_REL_TOL = 1e-6
_ABS_TOL = 1e-9

VALID_LABELS = {"__PROVED__", "__DISPROVED__"}


# ---------------------------------------------------------------------------
# LaTeX -> something SymPy can parse
# ---------------------------------------------------------------------------

_GREEK = ("alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota "
          "kappa lambda mu nu xi rho sigma tau upsilon phi varphi chi psi omega "
          "Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega").split()


def _strip_wrappers(s: str) -> str:
    """Remove the LaTeX that carries no arithmetic: $, \\left, spacing, text."""
    s = s.strip()
    while s.startswith("$") and s.endswith("$") and len(s) > 2:
        s = s[1:-1].strip()
    s = s.replace("$", " ")
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\[,;:!]", " ", s)
    s = re.sub(r"\\quad|\\qquad", " ", s)
    s = re.sub(r"\\(?:text|mathrm|mbox|textbf|mathbf|operatorname)\s*\{([^{}]*)\}", r" \1 ", s)
    s = re.sub(r"\\displaystyle|\\limits", " ", s)
    # Degrees and percent are units on the number, not operations on it. Both
    # sides get the same treatment, so dropping them keeps the comparison fair.
    s = re.sub(r"\^?\s*\{?\s*\\circ\s*\}?|°|\\degree", " ", s)
    s = re.sub(r"\\cup\b", " @U@ ", s)
    s = s.replace("\\%", " ").replace("%", " ")
    s = s.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    s = s.replace("\u00d7", "*").replace("\u00b7", "*")
    return s.strip()


def _latex_to_expr_text(s: str) -> str:
    """Rewrite the LaTeX constructs that actually change the value."""
    s = _strip_wrappers(s)

    # \frac{a}{b}, \dfrac, \tfrac — innermost first, so nesting resolves.
    frac = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
    for _ in range(12):
        new = frac.sub(r"((\1)/(\2))", s)
        if new == s:
            break
        s = new
    # \frac12 — the two-argument form without braces.
    s = re.sub(r"\\[dt]?frac\s*(\d)\s*(\d)", r"((\1)/(\2))", s)

    # Roots: \sqrt[n]{x} before \sqrt{x}, or SymPy sees a stray bracket.
    s = re.sub(r"\\sqrt\s*\[([^\]]*)\]\s*\{([^{}]*)\}", r"((\2)**(1/(\1)))", s)
    for _ in range(6):
        new = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", s)
        if new == s:
            break
        s = new
    s = re.sub(r"\\sqrt\s*(\w)", r"sqrt(\1)", s)

    # \log_{10} 250 is a base, not a subscript to discard: dropping it turns
    # log base 10 into the natural log and the answer silently changes.
    s = re.sub(r"\\(log|ln)\s*_\s*\{([^{}]*)\}\s*", r"LOGBASE(\2)@", s)
    s = re.sub(r"\\(log|ln)\s*_\s*(\w+)\s*", r"LOGBASE(\2)@", s)
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    # Floor and ceiling are operations, not decoration. Left to the generic
    # \command stripper they vanish, and then n/2+1 matches ceil(n/2)+1 and
    # floor matches ceiling.
    for _ in range(6):
        new_s = re.sub(r"\\lceil(.+?)\\rceil", r"ceiling(\1)", s)
        new_s = re.sub(r"\\lfloor(.+?)\\rfloor", r"floor(\1)", new_s)
        if new_s == s:
            break
        s = new_s
    s = s.replace("\u2308", "ceiling(").replace("\u2309", ")")
    s = s.replace("\u230a", "floor(").replace("\u230b", ")")
    s = re.sub(r"\\d?binom\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"binomial(\1,\2)", s)
    s = re.sub(r"\\(ln|log|exp|sin|cos|tan|cot|sec|csc|arcsin|arccos|arctan|sinh|cosh|tanh)\b",
               r"\1", s)
    s = s.replace("\\pi", "pi").replace("\\infty", "oo")
    for g in _GREEK:
        s = s.replace("\\" + g, g)
    s = re.sub(r"\\!|\\ ", " ", s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)          # anything left over is decoration

    s = _resolve_logbase(s)
    s = s.replace("^", "**")
    s = s.replace("{", "(").replace("}", ")")
    # Thousands separators, but not the commas that separate answers: only
    # between digits with exactly three following.
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)
    return s.strip()


_FUNCS = {
    "sqrt", "log", "ln", "exp", "sin", "cos", "tan", "cot", "sec", "csc",
    "arcsin", "arccos", "arctan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
    "binomial", "factorial", "abs", "floor", "ceiling", "gcd", "lcm", "pi", "oo",
    "Abs", "Max", "Min", "LOGBASE", "ceiling", "floor",
}
_IDENT_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\b")


def _symbol_dict(text: str) -> dict:
    """Bind every identifier that is not a function to a plain Symbol.

    Without this, `\frac{N}{2}` fails to parse: SymPy's `N` is numerical
    evaluation, so the expression becomes a function divided by two. The same
    holds for E, I, S, O, beta, gamma and zeta, all of which appear in this
    dataset as ordinary variables.
    """
    return {name: sympy.Symbol(name)
            for name in set(_IDENT_RE.findall(text)) if name not in _FUNCS}


def _resolve_logbase(s: str) -> str:
    """LOGBASE(b)@ arg  ->  log(arg, b)."""
    out = s
    for _ in range(6):
        m = re.search(r"LOGBASE\(([^()]*)\)@\s*", out)
        if not m:
            break
        rest = out[m.end():]
        # The argument is the next bracketed group, or the next atom.
        am = re.match(r"\s*\(([^()]*)\)", rest)
        if am:
            arg, after = am.group(1), rest[am.end():]
        else:
            am = re.match(r"\s*([0-9.]+|[A-Za-z][A-Za-z0-9_]*)", rest)
            if not am:
                out = out[:m.start()] + "log" + rest
                continue
            arg, after = am.group(1), rest[am.end():]
        out = out[:m.start()] + f"log({arg},{m.group(1)})" + after
    return out.replace("LOGBASE(", "log(").replace(")@", ")")


def _to_sympy(text: str) -> Optional[Any]:
    """Parse one scalar answer. None when it is not a mathematical object."""
    t = _latex_to_expr_text(text)
    if not t:
        return None
    # "f(x) = x - 1", "k = n+1": the answer is the right-hand side. Keep the
    # last one so "a = b = 3" resolves to 3.
    if "=" in t and not re.search(r"[<>!]=|=[<>]", t):
        parts = [p for p in t.split("=") if p.strip()]
        if len(parts) >= 2:
            t = parts[-1]
    t = t.strip()
    if not t:
        return None
    try:
        expr = parse_expr(t, transformations=_TRANSFORMS, evaluate=True,
                          local_dict=_symbol_dict(t))
    except Exception:
        return None
    if isinstance(expr, sympy.logic.boolalg.BooleanAtom):
        return None
    return expr


# ---------------------------------------------------------------------------
# Splitting an answer into its parts
# ---------------------------------------------------------------------------

def _split_top_level(s: str, seps: Sequence[str] = (",",)) -> List[str]:
    """Split on separators that are not inside brackets."""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch in seps and depth <= 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [p.strip() for p in out if p.strip()]


_TUPLE_RE = re.compile(r"\(([^()]*)\)")


def _parse_tuples(s: str) -> Optional[List[Tuple]]:
    """`(3,2),(-3,2)` -> [(3,2), (-3,2)]. None when it is not tuples."""
    body = _strip_wrappers(s)
    groups = _TUPLE_RE.findall(body)
    if not groups:
        return None
    # Everything outside the parentheses must be separators, or this is an
    # expression that merely contains brackets, like n(n-1).
    if _TUPLE_RE.sub("", body).strip(" ,;and") != "":
        return None
    tuples = []
    for g in groups:
        parts = _split_top_level(g)
        if len(parts) < 2:
            return None
        vals = [_to_sympy(p) for p in parts]
        if any(v is None for v in vals):
            return None
        tuples.append(tuple(vals))
    return tuples or None


_INTERVAL_RE = re.compile(r"^([\[\(])\s*(.+?)\s*,\s*(.+?)\s*([\]\)])$")


def _parse_interval(s: str) -> Optional[Tuple]:
    """`[0, 1)` -> (lo, hi, lo_closed, hi_closed). None when not an interval.

    A leading variable name — "t(0,4]", the t being what the interval
    constrains — is dropped: the answer is the interval.
    """
    body = _strip_wrappers(s)
    body = re.sub(r"^[A-Za-z][A-Za-z0-9_]*\s*(?=[\[\(])", "", body).strip()
    m = _INTERVAL_RE.match(body)
    if not m:
        return None
    lo, hi = _to_sympy(m.group(2)), _to_sympy(m.group(3))
    if lo is None or hi is None:
        return None
    try:                                   # an interval's ends are comparable
        if lo.is_number and hi.is_number and float(lo) > float(hi):
            return None
    except (TypeError, ValueError):
        return None
    return (lo, hi, m.group(1) == "[", m.group(4) == "]")


# ---------------------------------------------------------------------------
# Equivalence
# ---------------------------------------------------------------------------

def _num(expr) -> Optional[float]:
    try:
        if expr is None or not expr.is_number:
            return None
        return float(expr.evalf())
    except (TypeError, ValueError, AttributeError):
        return None


_DECIMAL_RE = re.compile(r"(?<![\d.])\d*\.(\d+)(?![\d.])")


def _gold_decimals(text: Optional[str]) -> Optional[int]:
    """How many decimal places the gold answer was written to.

    Gold is sometimes a rounded decimal where the exact value is not — "0.286"
    for 2/7 — and the exact answer has to count as correct. The precision to
    forgive is the one gold itself states, and no more: reading it off the text
    is what keeps 9/25 (0.36) from being accepted as 2/5 (0.4), which a blanket
    one-decimal tolerance does.
    """
    if not text:
        return None
    places = [len(m) for m in _DECIMAL_RE.findall(str(text))]
    if not places:
        return None
    d = max(places)
    # One or two decimal places is almost always an exact answer, not a rounded
    # one, and forgiving to that width would accept 0.36 as 0.4. Three or more
    # is the shape of a deliberately truncated irrational ("0.286" for 2/7).
    return d if 3 <= d <= 8 else None


def _scalar_equal(a, b, gold_text: Optional[str] = None) -> bool:
    """Same number, or the same expression written differently."""
    if a is None or b is None:
        return False
    fa, fb = _num(a), _num(b)
    if fa is not None and fb is not None:
        if fa == fb:
            return True
        scale = max(abs(fa), abs(fb), 1.0)
        if abs(fa - fb) <= max(_ABS_TOL, _REL_TOL * scale):
            return True
        # Only forgive the rounding gold itself declares.
        d = _gold_decimals(gold_text)
        if d is not None and round(fa, d) == round(fb, d):
            return True
        return False
    if fa is None and fb is None:
        try:
            if a == b:
                return True
            d = sympy.simplify(a - b)
            if d == 0:
                return True
            return bool(sympy.simplify(sympy.Eq(a, b)) is sympy.true)
        except Exception:
            return False
    return False


def _set_equal(xs, ys) -> bool:
    """Unordered comparison: every element on one side has a partner on the other."""
    if xs is None or ys is None or len(xs) != len(ys):
        return False
    remaining = list(ys)
    for x in xs:
        for i, y in enumerate(remaining):
            if _elem_equal(x, y):
                del remaining[i]
                break
        else:
            return False
    return True


def _interval_equal(a, b) -> bool:
    return (_scalar_equal(a[0], b[0]) and _scalar_equal(a[1], b[1])
            and a[2] == b[2] and a[3] == b[3])


def _elem_equal(x, y) -> bool:
    if isinstance(x, tuple) and isinstance(y, tuple):
        return len(x) == len(y) and all(_scalar_equal(a, b) for a, b in zip(x, y))
    if isinstance(x, tuple) or isinstance(y, tuple):
        return False
    return _scalar_equal(x, y)


# ---------------------------------------------------------------------------
# Per-dataset adapters
# ---------------------------------------------------------------------------

def _parse_any(text: str, answer_type: Optional[str] = None):
    """Parse an answer into the object it denotes, whatever shape that is.

    Returns ("tuples", [...]) / ("interval", ...) / ("set", [...]) /
    ("scalar", expr), or None when nothing could be made of it.
    """
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None

    want = (answer_type or "").strip().lower()

    if "@U@" in _strip_wrappers(text):
        pieces = [q for q in _strip_wrappers(text).split("@U@") if q.strip()]
        parsed = [_parse_interval(q) for q in pieces]
        if all(v is not None for v in parsed) and len(parsed) > 1:
            return ("intervals", parsed)

    if want != "numerical":
        tup = _parse_tuples(text)
        if tup is not None:
            return ("tuples", tup)
        itv = _parse_interval(text)
        if itv is not None:
            return ("interval", itv)

    # A comma-separated list of scalars is a set of answers, not one answer.
    body = _strip_wrappers(text)
    parts = _split_top_level(body, (",", ";"))
    if len(parts) >= 2:
        vals = [_to_sympy(p) for p in parts]
        if all(v is not None for v in vals):
            return ("set", vals)

    expr = _to_sympy(text)
    if expr is None:
        return None
    return ("scalar", expr)


def match_olympiadbench(pred: str, gold: str, answer_type: Optional[str] = None) -> bool:
    """OlympiadBench: four answer shapes, each compared as what it is.

    Numerical    a number, possibly exact (`\\frac{2}{7}\\sqrt{53}`); compared
                 numerically, with a tolerance because gold is often rounded.
    Expression   an algebraic form (`n(n-1)`, `f(x)=x-1`); compared symbolically,
                 so `n**2-n` matches, and an `f(x)=` prefix is dropped.
    Tuple        one or more points; compared as an unordered set of points, so
                 the four sign combinations may be listed in any order.
    Interval     compared by its endpoints and whether each end is closed.

    This is where the 77 samples the loader used to skip come back: they were
    skipped because the scorer could only handle the Numerical ones.
    """
    p, g = _parse_any(pred, answer_type), _parse_any(gold, answer_type)
    if p is None or g is None:
        return False
    kp, vp = p
    kg, vg = g
    if kp != kg:
        # A single answer and a one-element set are the same answer.
        if {kp, kg} == {"scalar", "set"}:
            sv = vp if kp == "set" else vg
            other = vg if kp == "set" else vp
            return len(sv) == 1 and _scalar_equal(sv[0], other)
        return False
    if kp == "tuples":
        return _set_equal(vp, vg)
    if kp == "set":
        return _set_equal(vp, vg)
    if kp == "interval":
        return _interval_equal(vp, vg)
    if kp == "scalar":
        return _scalar_equal(vp, vg, gold_text=gold)
    if kp == "intervals":
        if len(vp) != len(vg):
            return False
        rest = list(vg)
        for iv in vp:
            for i, jv in enumerate(rest):
                if _interval_equal(iv, jv):
                    del rest[i]
                    break
            else:
                return False
        return True
    return _scalar_equal(vp, vg)


def match_gsm8k(pred: str, gold: str, answer_type: Optional[str] = None) -> bool:
    """GSM8K: the answer is one number. Compared as a number, not as text."""
    p, g = _to_sympy(pred or ""), _to_sympy(gold or "")
    return _scalar_equal(p, g, gold_text=gold)


def normalise_label(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    up = str(label).strip().upper()
    if up in VALID_LABELS:
        return up
    if "DISPROVED" in up:
        return "__DISPROVED__"
    if "PROVED" in up:
        return "__PROVED__"
    return None


def match_label(pred: str, gold: str, answer_type: Optional[str] = None) -> bool:
    """FLD / FOLIO: the answer is a label."""
    p, g = normalise_label(pred), normalise_label(gold)
    return p is not None and p == g


ADAPTERS = {
    "OlympiadBench": match_olympiadbench,
    "GSM8K":         match_gsm8k,
    "FLD":           match_label,
    "FOLIO":         match_label,
}


def answers_match(pred, gold, dataset: Optional[str] = None,
                  answer_type: Optional[str] = None, domain: Optional[str] = None) -> bool:
    """Is `pred` the same answer as `gold`?

    `dataset` picks the adapter. Without one, the domain decides: a math answer
    is compared as OlympiadBench's are, which is the most permissive of the
    numeric adapters and reduces to a numeric comparison for a plain number.
    """
    if pred is None or gold is None or str(pred).strip() == "" or str(gold).strip() == "":
        return False
    fn = ADAPTERS.get(dataset or "")
    if fn is None:
        fn = match_label if domain == "logical" else match_olympiadbench
    try:
        return bool(fn(str(pred), str(gold), answer_type))
    except Exception:
        return False


def canonical(text, dataset: Optional[str] = None,
              answer_type: Optional[str] = None) -> Optional[str]:
    """A stable spelling of an answer, for grouping and for reading a report.

    Equivalence is decided by `answers_match`, never by comparing two of these:
    a canonical form cannot represent `n(n-1)` and `n**2-n` as one string
    without a normal form that does not exist for the general case.
    """
    if dataset in ("FLD", "FOLIO"):
        return normalise_label(text)
    parsed = _parse_any(text, answer_type) if text is not None else None
    if parsed is None:
        return None
    kind, val = parsed
    try:
        if kind == "scalar":
            f = _num(val)
            if f is not None:
                return str(int(f)) if f == int(f) else str(round(f, 8)).rstrip("0").rstrip(".")
            return str(val)
        if kind == "set":
            return ",".join(sorted(str(v) for v in val))
        if kind == "tuples":
            return ";".join(sorted("(" + ",".join(str(x) for x in t) + ")" for t in val))
        if kind == "interval":
            lo, hi, lc, hc = val
            return f"{'[' if lc else '('}{lo},{hi}{']' if hc else ')'}"
        if kind == "intervals":
            return " U ".join(sorted(
                f"{'[' if lc else '('}{lo},{hi}{']' if hc else ')'}"
                for lo, hi, lc, hc in val))
    except Exception:
        return None
    return None
