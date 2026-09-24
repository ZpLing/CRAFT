#!/usr/bin/env python3
"""
tfirf_terms.py  (Module I — TF-IRF Consensus Terms T_Con)
---------------------------------------------------------------------------
Extract important logical terms and domain-specific vocabulary from reasoning traces.

Called internally by steps_filter.py (Module I) and synthesize_trace.py (Module III);
run directly it just dumps the term table for inspection.
It implements TF-IRF term extraction: terms that appear frequently within a single
sample's traces but rarely across other samples' traces are considered important.

Key features:
- TF-IRF based extraction: high within-sample TF, low across-sample IDF
- Averages term frequencies across k traces per sample before extraction
- Supports both logical domain (LOGICAL_KEYWORDS) and math domain (LaTeX formula tokens)
- Outputs important term lists per sample
"""

from __future__ import annotations

import argparse
import io
import json
import keyword
import re
import tokenize
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import sympy
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations, implicit_multiplication, convert_xor,
    )
    # implicit_multiplication, NOT implicit_multiplication_application. The "application"
    # half rewrites "sin x" as "sin(x)", inserting a call that was never in the text -- so
    # "chr 97" evaluated to 'a' and "open 1" opened file descriptor 1 and closed stdout,
    # neither of which any check on the token stream can see. Only 3p -> 3*p is wanted here.
    # convert_xor: without it SymPy reads a^b as bitwise XOR, so "3^0 = 1" (3 xor 0 = 3) and
    # "5^2 = 25" (7) came back as false arithmetic and the math check deleted correct steps.
    _SYMPY_TRANSFORMS = standard_transformations + (implicit_multiplication, convert_xor)
    # A namespace holding SymPy's names and nothing else. eval() inserts __builtins__ when
    # the globals lack it, so it is set empty rather than left out, and a bare builtin name
    # resolves to a Symbol instead of the function object.
    _SYMPY_NAMESPACE: dict = {}
    exec('from sympy import *', _SYMPY_NAMESPACE)
    _SYMPY_NAMESPACE['__builtins__'] = {}
    _SYMPY_OK = True
except Exception:          # SymPy is optional; without it equations keep their written form
    _SYMPY_OK = False

import importlib.util as _ilu
_cfg_path = Path(__file__).resolve().parents[2] / "config.py"
_spec = _ilu.spec_from_file_location("_part_config", _cfg_path)
_cfg  = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_cfg)

# Predefined list of logical words (used for identification, not necessarily prioritized)
LOGICAL_KEYWORDS = {
    # Step-related
    "step", "steps", "step1", "step2", "step3", "step4", "step5",
    "first", "second", "third", "next", "then", "finally",

    # Conditional logic
    "if", "else", "elif", "when", "whenever", "unless", "provided",

    # Causal relations
    "because", "since", "as", "due", "therefore", "thus", "hence",
    "so", "consequently", "accordingly",

    # Logical connectives
    "and", "or", "but", "however", "moreover", "furthermore",
    "additionally", "also", "besides",

    # Inference words
    "implies", "imply", "implies", "conclude", "conclusion",
    "infer", "inference", "deduce", "deduction",

    # Proof-related
    "prove", "proof", "proven", "disprove", "disproven",
    "contradiction", "contradictory", "assume", "assumption",

    # Fact references
    "fact", "facts", "given", "premise", "premises",
    "hypothesis", "statement", "claim",

    # Logical quantifiers
    "all", "any", "some", "none", "every", "each",
    "not", "no", "never", "always",

    # Comparisons and relations
    "equal", "equals", "equivalent", "same", "different",
    "greater", "less", "than", "from",
}

# Overly common logical words (should be filtered out — they appear in too many samples and carry little semantic signal)
COMMON_LOGICAL_WORDS = {
    "because", "also", "so", "and", "or", "but", "if", "then",
    "therefore", "thus", "since", "as", "when", "not", "no",
    "all", "any", "some", "each", "every", "fact", "facts",
    "step", "steps", "prove", "proof", "conclude", "conclusion",
    "approve", "disprove", "given", "statement", "claim",
    "implies", "however",  # Added: common logical connectives
}

# Stopwords (common but semantically empty words)
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "can", "this", "that", "these", "those",
    "it", "its", "they", "them", "their", "there", "here", "where", "what",
    "which", "who", "whom", "whose", "how", "why", "when", "to", "of", "in",
    "on", "at", "by", "for", "with", "from", "as", "into", "onto", "up",
    "down", "out", "off", "over", "under", "above", "below", "between",
    "among", "through", "during", "before", "after", "while", "about",
    "against", "within", "without", "throughout", "across", "around",
    "near", "far", "inside", "outside", "beside", "besides", "except",
    "including", "excluding", "concerning", "regarding", "according",
    "i", "you", "he", "she", "we", "us", "our", "your", "my", "me",
    "him", "her", "his", "hers", "mine", "yours", "ours", "theirs",
}


# ── Math domain: mathematical operation words (discriminative across problem types, should not be filtered) ──
MATH_OPERATION_WORDS = {
    # Algebraic operations
    "simplify", "simplifies", "simplified",
    "substitute", "substituting", "substitution",
    "expand", "expanding", "factor", "factoring", "factorize",
    "solve", "solving", "cancel", "cancels", "eliminate", "eliminating",
    "multiply", "multiplying", "divide", "dividing", "subtract", "subtracting",
    "add", "adding", "rearrange", "rearranging", "isolate", "isolating",
    # Equations/functions
    "equation", "equations", "inequality", "expression", "formula",
    "polynomial", "quadratic", "linear", "coefficient", "variable",
    "numerator", "denominator", "fraction", "exponent", "root", "radical",
    # Geometry/trigonometry
    "midpoint", "distance", "radius", "diameter", "area", "perimeter",
    "angle", "sine", "cosine", "tangent", "hypotenuse", "perpendicular",
    "parallel", "congruent", "similar", "vertex", "vertices",
    # Number theory/combinatorics
    "remainder", "divisible", "prime", "modulo", "factorial",
    "permutation", "combination", "probability",
    # Calculus
    "derivative", "integral", "maximum", "minimum", "critical",
    # Verification/checking
    "verify", "verifying", "check", "checking", "confirm", "plug",
    "substitute", "validate",
    # Derivation step words (meaningful in math context)
    "therefore", "thus", "hence", "conclude", "obtain", "get", "find",
}

# Words that are too generic in the math domain and lack discriminative power (filter out)
MATH_COMMON_WORDS = {
    "step", "steps", "let", "so", "now", "also", "then", "first",
    "second", "third", "next", "finally", "note", "notice", "use",
    "using", "since", "because", "have", "need", "want", "know",
    "see", "can", "we", "this", "that", "which", "problem", "question",
    "answer", "solution", "result", "value", "number", "equal", "equals",
    "side", "left", "right", "both", "each", "all", "any",
}

# LaTeX command words (meaningless when appearing alone; only meaningful as part of a full formula)
_LATEX_COMMANDS = re.compile(
    r'\\(?:frac|sqrt|cdot|times|div|pm|mp|leq|geq|neq|approx'
    r'|sum|prod|int|lim|infty|partial|nabla|Delta|Sigma|Pi'
    r'|alpha|beta|gamma|delta|epsilon|theta|lambda|mu|pi|sigma|phi|omega'
    r'|left|right|text|mathrm|mathbf|overline|hat|vec|bar)\b'
)


_OPERATOR_SPELLINGS = {
    '\u00d7': '*', '\u00f7': '/', '\u00b7': '*', '\u2212': '-',
}
_LATEX_OPERATORS = re.compile(r'\\(?:cdot|times)\b')
_LATEX_DIVIDE    = re.compile(r'\\div\b')


def fold_math_operators(text: str) -> str:
    """Write every spelling of an operator the one way the parser reads.

    An operator the equation pattern does not recognise is read as another operand, and two
    operands running together end the match, so the pattern restarts in the middle of the
    expression. "2 \\cdot 3 = 6" was cut to "3 = 6" and judged an arithmetic error -- that
    alone accounted for all 18 remaining force-deletions on GSM8K.
    """
    text = text.translate(str.maketrans(_OPERATOR_SPELLINGS))
    text = _LATEX_OPERATORS.sub('*', text)
    return _LATEX_DIVIDE.sub('/', text)


# A match that begins straight after one of these began in the middle of an expression:
# the character is part of the equation the step wrote, but not part of what was matched.
# Widening the operand class one symbol at a time never ends -- $ and % took their turn
# after \cdot and \u00d7 -- so the boundary is checked instead of enumerated.
_MIDEXPR_PRECEDERS = set('+-*/^$%\u00b7\u00d7\u00f7\u2212')


def iter_equations(text: str):
    """Yield the equations in a step, skipping any the pattern only caught part of.

    Delimited formulas come first and their spans are then masked, because a $ means one
    thing opening a formula and another in front of a price. Once the formulas are out of
    the way a $ can only be currency, and an equation starting right after one started in
    the middle of what the step wrote.
    """
    text = fold_math_operators(text)

    for match in list(_LATEX_INLINE.finditer(text)) + list(_LATEX_DISPLAY.finditer(text)):
        body = match.group(1) or match.group(2) or ''
        if '=' in body:
            yield body

    rest = _LATEX_DISPLAY.sub(' ', _LATEX_INLINE.sub(' ', text))
    for match in _BARE_EQUATION.finditer(rest):
        before = rest[:match.start()].rstrip()
        if before and before[-1] in _MIDEXPR_PRECEDERS:
            continue
        yield match.group(1)


# parse_expr() compiles and eval()s what it is given, which SymPy documents as unsafe on
# untrusted input. These strings come from model-written traces, and the LaTeX path hands
# over whatever sat between the dollar signs, so a step reading
# $open('f','w').write('x')=1$ actually ran. Only expression characters reach the parser,
# Filtering characters is not enough: letters, dots and parentheses are all a call needs,
# and a string argument can be built at run time without ever writing a quote, so
# eval(chr(95)+chr(95)+...) once handed back __import__ with no dunder in the source at all.
# What closes that is the namespace, not the syntax. _SYMPY_NAMESPACE holds SymPy's bindings
# and an empty __builtins__, so a name it does not know can never be called: with only
# implicit_multiplication applied, "chr(97)" parses as 97*chr and "open(1)" as open. Written
# calls are therefore allowed, but only for names on this list.
#
# Emptying __builtins__ is not on its own enough. SymPy exports plenty that is not
# mathematics -- plot() drew a figure, and preview() shells out to LaTeX and opens a viewer
# -- so the callable names are enumerated rather than excluded. Everything here is an
# ordinary analytic function of bounded cost; factorial and binomial are deliberately absent,
# since factorial(10**9) is short to write and long to finish.
#
# Refusing the rest is also what keeps equations honest. An unknown name before a paren is
# read as multiplication, so f(x) became f*x and matched a step that wrote f*x, while
# f(x+1) became f*(x+1) and distributed across an argument. Those are different statements,
# and refusing the expression leaves it in its written form, which matches nothing it should
# not.
# Intersected with the namespace, because a listed name SymPy does not define is not a call
# at all: lowercase max is not bound, so max(1) would parse as the symbol max times (1) and
# reintroduce exactly the multiplication-in-disguise this list exists to prevent.
_CALLABLE_NAMES = frozenset(name for name in {
    'sqrt', 'cbrt', 'root', 'exp', 'log', 'ln', 'Abs', 'abs', 'sign',
    'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
    'asin', 'acos', 'atan', 'acot', 'asec', 'acsc',
    'sinh', 'cosh', 'tanh', 'coth', 'asinh', 'acosh', 'atanh',
    'floor', 'ceiling', 'ceil', 'gcd', 'lcm', 'Min', 'Max', 'min', 'max',
} if _SYMPY_OK and name in _SYMPY_NAMESPACE)

# The token check also refuses what the namespace cannot: "." reaches attributes of objects
# SymPy does hold, strings and keywords open other grammar, and a length cap bounds the work
# a single expression can ask for.
_MAX_EXPR_LEN = 200
# A power is evaluated, so its size must be bounded: "9^9^9^9" or "10**10**6" would ask SymPy
# for a number with billions of digits. An exponent must be a plain integer literal of at most
# _MAX_EXPONENT (optionally negated), a power may not be raised again, and a parenthesized
# group may not be raised at all. Anything else is
# refused and the equation goes unjudged, which leaves the step to the z-score.
_MAX_EXPONENT = 64
_POWER_OPS = frozenset({'**', '^'})
_ARITHMETIC_OPS = frozenset({'+', '-', '*', '/', '**', '^', '(', ')'})
_SKIPPABLE_TOKENS = frozenset({
    tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER, tokenize.INDENT, tokenize.DEDENT,
})


def _is_pure_arithmetic(expr: str) -> bool:
    """True when the text can only combine names and numbers with arithmetic operators."""
    if len(expr) > _MAX_EXPR_LEN:
        return False
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(expr).readline))
    except Exception:
        return False
    significant = [t for t in tokens if t.type not in _SKIPPABLE_TOKENS]
    for i, token in enumerate(significant):
        if token.type == tokenize.OP and token.string in _POWER_OPS:
            if i > 0 and significant[i - 1].string == ')':
                return False          # (…)^n: nested groups would build a tower again
            j = i + 1
            if j < len(significant) and significant[j].string in ('-', '+'):
                j += 1
            if j >= len(significant) or significant[j].type != tokenize.NUMBER:
                return False          # exponent is an expression, a name or missing
            try:
                if abs(float(significant[j].string)) > _MAX_EXPONENT:
                    return False
            except ValueError:
                return False
            k = j + 1
            if k < len(significant) and significant[k].string in _POWER_OPS:
                return False          # a tower: a**b**c
    previous_name = None
    for token in tokens:
        if token.type in _SKIPPABLE_TOKENS:
            continue
        if token.type == tokenize.NUMBER:
            previous_name = None
        elif token.type == tokenize.NAME:
            if keyword.iskeyword(token.string):
                return False
            previous_name = token.string
        elif token.type == tokenize.OP:
            if token.string not in _ARITHMETIC_OPS:
                return False          # "." , "," , "[" , ":" ... all refused
            if token.string == '(' and previous_name is not None:
                if previous_name not in _CALLABLE_NAMES:
                    return False      # an unlisted call, or multiplication in disguise
            previous_name = None
        else:
            return False              # strings, f-strings, error tokens
    return True


def safe_parse_expr(expr: str, evaluate: bool = True):
    """parse_expr() restricted to things that can only be arithmetic. None if refused."""
    if not _SYMPY_OK or not expr or not _is_pure_arithmetic(expr):
        return None
    try:
        return parse_expr(expr, transformations=_SYMPY_TRANSFORMS, evaluate=evaluate,
                          global_dict=_SYMPY_NAMESPACE)
    except Exception:
        return None


def _sort_commutative(expr):
    """Order the operands of + and * so that writing order stops mattering."""
    if not expr.args:
        return expr
    args = [_sort_commutative(a) for a in expr.args]
    if isinstance(expr, (sympy.Add, sympy.Mul)):
        args = sorted(args, key=sympy.srepr)
    # Rebuild only what Add/Mul accept. Handing them a set or a relation is deprecated
    # rather than an error, so it warns instead of raising and try/except would miss it.
    if not all(isinstance(a, sympy.Expr) for a in args):
        return expr
    try:
        return expr.func(*args, evaluate=False)
    except Exception:
        return expr          # not rebuildable; leave it as parsed


@lru_cache(maxsize=50000)
def _canonicalise_equation(expr: str) -> Optional[str]:
    """Reduce an equation to a key that does not depend on how it was written.

    Two traces deriving the same thing rarely type it the same way -- 3p+e=1.24,
    e+3p=1.24 and 1.24=3p+e are one equation and three different strings, so consensus
    counted them as three unrelated terms. Each side is parsed WITHOUT evaluation and its
    commutative operands sorted, then the two sides are stored as an unordered pair.

    Not evaluating is the point. Letting SymPy fold the arithmetic would collapse 2+2=4
    and 3+1=4 onto one key, and reducing to lhs-rhs would collapse every true numeric
    equation onto zero, merging steps that computed entirely different things. Returns
    None when the equation cannot be parsed, and the caller keeps the written form.
    """
    if not _SYMPY_OK or expr.count('=') != 1:
        return None
    lhs, rhs = expr.split('=')
    if not lhs.strip() or not rhs.strip():
        return None
    parsed = [safe_parse_expr(side, evaluate=False) for side in (lhs, rhs)]
    if any(p is None for p in parsed):
        return None
    try:
        sides = [sympy.srepr(_sort_commutative(p)) for p in parsed]
    except Exception:
        return None
    return "EQN:" + "|".join(sorted(sides))


# The command names themselves, for filtering them out of prose word counts.
_LATEX_COMMAND_WORDS = frozenset(re.findall(r'[a-zA-Z]+', _LATEX_COMMANDS.pattern)) - {'re'}


def _normalise_latex_token(expr: str) -> str:
    """Normalise a LaTeX expression into a comparable token string.

    Strategy:
    - Strip whitespace
    - Preserve structure (variable names, numbers, operators)
    - Unify common equivalent forms (e.g. \\cdot → *)
    - Discard tokens shorter than 3 chars after stripping
    """
    expr = expr.strip()
    # Spell the unicode operators the way the parser and the rest of the pipeline do, so
    # "3 \u00d7 4" and "3 * 4" are not two different terms.
    expr = fold_math_operators(expr)
    # Remove bare LaTeX commands (e.g. \\frac itself; keep its arguments)
    expr = _LATEX_COMMANDS.sub('', expr).strip()
    # Collapse whitespace
    expr = re.sub(r'\s+', ' ', expr)
    return expr


# Regex for extracting inline/display LaTeX formulas
_LATEX_INLINE  = re.compile(r'\$\$(.+?)\$\$|\$(.+?)\$', re.DOTALL)
_LATEX_DISPLAY = re.compile(r'\\\[(.+?)\\\]|\\\((.+?)\\\)', re.DOTALL)
# Bare equations: an expression, =, an expression, with no $ needed. Whitespace is allowed
# only around operators, never between two operands. Letting \s float free made any sentence
# containing "=" match from its first word onward, so 40% of EQ: tokens were prose --
# "EQ:Add all strawberries to get the total number used for jam. Using Betty = 16" -- which
# is unique to its trace and therefore pure noise in the frequency counts.
# \u00d7 \u00f7 \u00b7 \u2212 are operators too. Leaving them out made the pattern start after
# them, so "0.5 \u00d7 12 = (50/100) \u00d7 12" was cut down to the fragment "12 = (50/100)" and the
# two sides being compared were no longer the two sides the step had written.
_EQ_OPERAND = r'[A-Za-z0-9_\.\\{}\^\(\)]+'
_EQ_OPS     = r'[\+\-\*/\^\u00d7\u00f7\u00b7\u2212]'
_EQ_SIDE    = _EQ_OPERAND + r'(?:\s*' + _EQ_OPS + r'\s*' + _EQ_OPERAND + r')*'
_BARE_EQUATION = re.compile(
    r'(?<![A-Za-z0-9])(' + _EQ_SIDE + r'\s*=\s*' + _EQ_SIDE + r')(?![A-Za-z0-9])'
)


# Logical predicate applications: Pred(arg), optionally negated. FLD and ProofWriter carry a
# step's content in these -- 33% of logical steps contain at least one -- and the word
# tokenizer below either breaks the binding (can_read(Mike) becomes can_read plus mike, so
# it collides with took_bar(Mike)) or loses the term outright when the predicate or the
# argument is a single letter, since V(B) yields v and b and both fail the len > 1 rule.
# The name binds tight to the paren and the argument holds no whitespace. Prose writes its
# asides with a space -- "is a fellow (Step 2)", "a student (Fact 3)" -- and allowing that
# shape turned 55% of these tokens into citations while erasing the very words they cite
# from the word stream below.
_LOGIC_PREDICATE = re.compile(
    r'([\u00ac~]\s*)?([A-Za-z][A-Za-z0-9_]*)\(([^()\s]{1,30})\)'
)


_NEG_MARK = '\u00ac'


def _normalise_logic_token(neg: str, pred: str, arg: str) -> str:
    """Render one predicate application as a single comparable token."""
    arg = re.sub(r'\s+', ' ', arg.strip().lower())
    mark = _NEG_MARK if neg else ''
    return "LOGIC:" + mark + pred.lower() + "(" + arg + ")"


def tokenize_math_text(text: str) -> List[str]:
    """Math-domain-specific tokenizer.

    Extracts two categories of tokens:
    1. Complete mathematical expressions (LaTeX formulas, bare equations) as atomic units
    2. Discriminative mathematical operation words (from MATH_OPERATION_WORDS)

    Design principles:
    - $3p+e=1.24$ is treated as one token, not split into 3, p, e, 1, 24
    - Formulas repeated across traces of the same problem → high TF, low IDF (within-problem consensus)
    - Different problems use different variables/values → high IDF (cross-problem discriminative power)
    """
    if not text:
        return []

    tokens: List[str] = []

    # ── 1. Extract LaTeX formulas ($...$ and \[...\]) ─────────────────────────────
    for m in list(_LATEX_INLINE.finditer(text)) + list(_LATEX_DISPLAY.finditer(text)):
        expr = m.group(1) or m.group(2) or ''
        norm = _normalise_latex_token(expr)
        if len(norm) >= 3:
            tokens.append('MATH:' + (_canonicalise_equation(norm) or norm))

    # ── 2. Extract bare equations (outside $, but containing =) ────────────────
    # Mask already-processed LaTeX regions to avoid double-counting
    text_no_latex = _LATEX_INLINE.sub('', text)
    text_no_latex = _LATEX_DISPLAY.sub('', text_no_latex)

    for equation in iter_equations(text_no_latex):
        norm = _normalise_latex_token(equation.strip())
        if len(norm) >= 3 and '=' in norm:
            tokens.append('EQ:' + (_canonicalise_equation(norm) or norm))

    # ── 3. Extract mathematical operation words (discriminative verbs/nouns) ────────────────────────────
    words = re.findall(r'\b[a-z]+\b', text.lower())
    for w in words:
        if w in MATH_OPERATION_WORDS and w not in MATH_COMMON_WORDS:
            tokens.append(w)

    # ── 4. Content words ───────────────────────────────────────────────────────
    # Read from the masked text, so a word inside a formula is not counted again beside the
    # MATH:/EQ: token that already carries it, and drop bare LaTeX command names, which are
    # markup rather than content and frequent enough to behave like stopwords.
    words = re.findall(r'\b[a-z]+\b', text_no_latex.lower())
    # A step can carry its whole argument in prose -- "Convert the number of can payments
    # into dollars using the given value per payment" -- and steps 1-3 return nothing at all
    # for it. That left 1088 of 2798 GSM8K steps with no tokens, so they contributed nothing
    # to the consensus and could not be scored against it.
    for w in words:
        if w in MATH_OPERATION_WORDS or w in MATH_COMMON_WORDS or w in STOPWORDS:
            continue
        if w in _LATEX_COMMAND_WORDS:
            continue
        if len(w) > 2:
            tokens.append(w)

    return tokens


def tokenize_text(text: str, domain: str = "logical") -> List[str]:
    """Tokenize text, retaining logical words and important terms.

    Args:
        domain: "logical" (default, original behaviour) or "math" (uses math-formula-aware tokenizer)
    """
    if domain == "math":
        return tokenize_math_text(text)

    if not text:
        return []

    # Predicate applications first, as atomic units, then mask their spans so the word
    # tokenizer does not also emit their pieces and count the same content twice. This
    # mirrors how tokenize_math_text() masks LaTeX before reading operation words.
    tokens: List[str] = []
    for m in _LOGIC_PREDICATE.finditer(text):
        tokens.append(_normalise_logic_token(m.group(1), m.group(2), m.group(3)))
    text = _LOGIC_PREDICATE.sub(' ', text)

    # Convert to lowercase
    text = text.lower()

    # Extract words (letters, digits, hyphens)
    words = re.findall(r'\b[a-z0-9]+(?:\-[a-z0-9]+)*\b', text)

    # Filter out stopwords
    tokens.extend(t for t in words if t not in STOPWORDS and len(t) > 1)

    return tokens


def extract_reasoning_text(traces: List[Dict[str, Any]]) -> List[str]:
    """Extract reasoning text from traces."""
    texts = []
    for trace in traces:
        # Prefer reasoning_text; fall back to raw_response
        text = trace.get("reasoning_text") or trace.get("raw_response", "")
        if text:
            texts.append(text)
    return texts


def calculate_tf(tokens: List[str]) -> Dict[str, float]:
    """Calculate Term Frequency (TF)."""
    if not tokens:
        return {}

    token_counts = Counter(tokens)
    total_tokens = len(tokens)

    tf = {word: count / total_tokens for word, count in token_counts.items()}
    return tf


def calculate_idf(all_documents: List[List[str]], term: str) -> float:
    """Calculate Inverse Document Frequency (IDF)."""
    # Count how many documents contain the term
    docs_containing_term = sum(1 for doc in all_documents if term in doc)

    if docs_containing_term == 0:
        return 0.0

    # IDF = log(total documents / documents containing the term)
    total_docs = len(all_documents)
    idf = np.log(total_docs / docs_containing_term)

    return idf


class DocFreqTable:
    """Document frequencies for a fixed corpus, so IDF can be scored without re-scanning it.

    This is what makes the two IRF settings comparable. Under ``idf_scope="sample"`` the
    corpus is one sample's own steps (a fresh table per sample, matching what
    calculate_idf() computes on the fly); under ``idf_scope="global"`` it is every step of
    every sample, built once and shared. Only the corpus differs — the scoring is identical.
    """

    def __init__(self, n_docs: int = 0, df: Optional[Dict[str, int]] = None,
                 normalize: bool = False, domain: Optional[str] = None):
        self.n_docs = n_docs
        self.df: Counter = Counter(df or {})
        # log(N/df) ranges over [0, log(N)], so its scale follows corpus size: the same raw
        # score means something different under a 44-document sample corpus and a 17k-document
        # global one. Dividing by log(N) maps it to [0, 1] so absolute thresholds such as
        # min_tfidf carry the same meaning whichever corpus is in play.
        self.normalize = normalize
        # Tokenisation differs per domain -- math emits MATH:/EQ: formula tokens where
        # logical emits words -- so frequencies from one domain say nothing about the
        # other. The domain is recorded here so a table cannot be loaded into the wrong run.
        self.domain = domain

    def add_document(self, tokens: List[str]) -> None:
        self.n_docs += 1
        self.df.update(set(tokens))

    @classmethod
    def from_documents(cls, documents: List[List[str]],
                       normalize: bool = False,
                       domain: Optional[str] = None) -> "DocFreqTable":
        table = cls(normalize=normalize, domain=domain)
        for tokens in documents:
            table.add_document(tokens)
        return table

    def idf(self, term: str) -> float:
        """IDF of a term against this corpus; mirrors calculate_idf()'s log(N/df)."""
        if self.n_docs == 0:
            return 0.0
        # A term the corpus has never seen is maximally rare, not maximally common, so
        # fall back to df=1 rather than returning 0.0 and letting min_idf delete it. Only
        # reachable when the table was built on a corpus other than the steps being scored
        # (e.g. a frozen background table); for a table built over these steps, df >= 1.
        docs_containing_term = self.df.get(term, 0) or 1
        idf = float(np.log(self.n_docs / docs_containing_term))
        if self.normalize:
            scale = float(np.log(self.n_docs))
            # A single-document corpus has no room for the ratio to vary; leave it raw
            # rather than dividing by zero.
            if scale > 0:
                idf /= scale
        return idf

    def to_dict(self) -> Dict[str, Any]:
        return {"n_docs": self.n_docs, "df": dict(self.df), "normalize": self.normalize,
                "domain": self.domain}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocFreqTable":
        return cls(n_docs=data.get("n_docs", 0), df=data.get("df", {}),
                   normalize=data.get("normalize", False), domain=data.get("domain"))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "DocFreqTable":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def __len__(self) -> int:
        return len(self.df)


def check_df_table(table: "DocFreqTable", domain: str, normalize: bool) -> None:
    """Reject a saved DF table whose corpus does not match the run loading it.

    A table holds frequencies for one tokenisation of one corpus on one IDF scale. Scoring
    against a table built for the other domain, or the other scale, yields numbers that look
    perfectly ordinary and mean nothing, so a mismatch is raised rather than absorbed. A
    table saved before the domain was recorded carries None and is accepted.
    """
    if table.normalize != normalize:
        raise ValueError(
            f"--df_table was built with idf_norm="
            f"{'log_n' if table.normalize else 'raw'}, but this run asks for "
            f"{'log_n' if normalize else 'raw'}; rebuild the table or match the flag"
        )
    if table.domain is not None and table.domain != domain:
        raise ValueError(
            f"--df_table was built for domain={table.domain!r}, but this run is "
            f"domain={domain!r}; each domain needs its own table"
        )


def resolve_df_table_path(path) -> Path:
    """Resolve a --df_table path identically in every module that touches it.

    resolve_input() leaves a relative path that does not exist yet pointing at the CWD, so
    the module that saves the table and the module that later loads it would disagree
    whenever they run from different directories, and the loader would silently rebuild a
    different corpus instead of reusing the saved one. Sending a not-yet-existing path
    through resolve_output() instead pins both sides to RESULTS_ROOT.
    """
    existing = _cfg.resolve_input(path)
    return existing if existing.exists() else _cfg.resolve_output(path)


class FlatDocFreqTable(DocFreqTable):
    """A table with no IRF signal at all: every term scores idf = 1.0.

    This is the ablation that isolates what the IRF factor buys, leaving TF-IRF == TF.
    Because 1.0 clears the usual min_idf floor, no term is dropped for being common
    either, so the only thing removed relative to the other settings is the weighting.
    """

    def idf(self, term: str) -> float:
        return 1.0


def calculate_tfidf_for_sample(
    sample_traces: List[Dict[str, Any]],
    all_trace_documents: List[List[str]],
    min_tf: float = 0.001,
    min_idf: float = 0.1,
    domain: str = "logical",
) -> Dict[str, float]:
    """
    Compute TF-IRF scores for a single sample.

    Procedure:
    1. Compute term frequency (TF) for each trace individually
    2. Average TF values across k traces to get the sample-level average TF
    3. Compute IDF for each term (relative to all traces, not just this sample)
    4. TF-IRF = average_TF * IDF

    Args:
        sample_traces: All traces for this sample
        all_trace_documents: Document list for all traces (one document per trace, used for IDF)
        min_tf: Minimum TF threshold
        min_idf: Minimum IDF threshold

    Returns:
        Dictionary mapping terms to their TF-IRF scores
    """
    # Extract text from all traces of this sample
    sample_texts = extract_reasoning_text(sample_traces)

    if not sample_texts:
        return {}

    # Step 1: Compute TF for each trace individually
    trace_tfs = []
    for text in sample_texts:
        tokens = tokenize_text(text, domain=domain)
        if tokens:
            tf = calculate_tf(tokens)
            trace_tfs.append(tf)

    if not trace_tfs:
        return {}

    # Step 2: Average TF values across k traces
    all_terms = set()
    for tf_dict in trace_tfs:
        all_terms.update(tf_dict.keys())

    avg_tf = {}
    for term in all_terms:
        term_frequencies = [tf_dict.get(term, 0.0) for tf_dict in trace_tfs]
        avg_tf[term] = np.mean(term_frequencies)

    # Step 3: Compute IDF and filter domain-specific common words
    common_filter = MATH_COMMON_WORDS if domain == "math" else COMMON_LOGICAL_WORDS

    tfidf_scores = {}
    for term, avg_tf_value in avg_tf.items():
        if avg_tf_value < min_tf:
            continue

        # math domain: tokens with MATH: / EQ: prefix skip the common word filter
        if domain != "math" or not (term.startswith("MATH:") or term.startswith("EQ:")):
            if term in common_filter:
                continue

        idf = calculate_idf(all_trace_documents, term)
        if idf < min_idf:
            continue

        tfidf_scores[term] = avg_tf_value * idf

    return tfidf_scores


def extract_important_terms(
    data: List[Dict[str, Any]],
    top_k: Optional[int] = None,
    min_tf: float = 0.001,
    min_idf: float = 0.1,
    min_tfidf: float = 0.0,
    prioritize_logical: bool = True,
    domain: str = "logical",
) -> Dict[str, Any]:
    """
    Extract important terms from reasoning traces data.

    Args:
        data: JSON data containing traces (list format)
        top_k: Number of top important terms to extract per sample
        min_tf: Minimum TF threshold
        min_idf: Minimum IDF threshold
        min_tfidf: Minimum TF-IRF score threshold (only terms above this are considered semantically rich)
        prioritize_logical: Whether to prioritize logical keywords

    Returns:
        Dictionary containing important terms
    """
    # Handle two data formats: list or dict with a "results" key
    if isinstance(data, dict):
        results = data.get("results", [])
    else:
        results = data

    if not results:
        return {"error": "No results found in data"}

    # Step 1: Prepare all traces as documents (for IDF computation)
    all_trace_documents = []
    sample_traces_list = []

    for sample in results:
        traces = sample.get("traces", [])
        if not traces:
            sample_traces_list.append([])
            continue

        for trace in traces:
            text = trace.get("reasoning_text") or trace.get("raw_response", "")
            if text:
                tokens = tokenize_text(text, domain=domain)
                all_trace_documents.append(tokens)

        sample_traces_list.append(traces)

    # Step 2: Compute TF-IRF for each sample
    all_important_terms = []

    for idx, (sample, traces) in enumerate(zip(results, sample_traces_list)):
        if not traces:
            all_important_terms.append({
                "sample_id": sample.get("sample_id", f"sample_{idx}"),
                "important_terms": [],
                "logical_terms": [],
                "tfidf_scores": {},
            })
            continue

        tfidf_scores = calculate_tfidf_for_sample(
            traces,
            all_trace_documents,
            min_tf=min_tf,
            min_idf=min_idf,
            domain=domain,
        )

        # math domain: MATH:/EQ: prefix tokens no longer filtered by COMMON_LOGICAL_WORDS
        if domain == "math":
            filtered_tfidf_scores = {
                term: score for term, score in tfidf_scores.items()
                if score >= min_tfidf
            }
        else:
            filtered_tfidf_scores = {
                term: score for term, score in tfidf_scores.items()
                if term not in COMMON_LOGICAL_WORDS and score >= min_tfidf
            }

        # Separate semantically rich words from regular terms (domain-aware)
        semantic_logical_terms = {}
        regular_terms = {}

        for term, score in filtered_tfidf_scores.items():
            if domain == "math":
                # math: MATH:/EQ: formula tokens are core features; math operation words are supplementary
                if term.startswith("MATH:") or term.startswith("EQ:"):
                    regular_terms[term] = score
                elif term in MATH_OPERATION_WORDS:
                    semantic_logical_terms[term] = score
                else:
                    regular_terms[term] = score
            else:
                if term in LOGICAL_KEYWORDS and term not in COMMON_LOGICAL_WORDS:
                    semantic_logical_terms[term] = score
                else:
                    regular_terms[term] = score

        sorted_semantic_logical = sorted(semantic_logical_terms.items(), key=lambda x: x[1], reverse=True)
        sorted_regular = sorted(regular_terms.items(), key=lambda x: x[1], reverse=True)

        if top_k is None:
            # No limit — return all terms that meet the criteria
            if prioritize_logical:
                # Add semantically rich logical words first, then domain terms
                selected_terms = sorted_semantic_logical + sorted_regular
            else:
                # Sort purely by TF-IRF score (common logical words already filtered)
                selected_terms = sorted(filtered_tfidf_scores.items(), key=lambda x: x[1], reverse=True)
        else:
            # Limit to top-k
            if prioritize_logical:
                # Prioritize semantically rich logical words, then fill with domain terms
                semantic_ratio = 0.3  # 30% semantically rich logical words, 70% domain terms
                num_semantic = max(1, int(top_k * semantic_ratio))
                selected_terms = sorted_semantic_logical[:num_semantic] + sorted_regular[:top_k - num_semantic]
            else:
                # Sort purely by TF-IRF score (common logical words already filtered)
                all_sorted = sorted(filtered_tfidf_scores.items(), key=lambda x: x[1], reverse=True)
                selected_terms = all_sorted[:top_k]

            # Enforce count limit
            selected_terms = selected_terms[:top_k]

        important_terms = [term for term, _ in selected_terms]
        # logical: record semantic logical words; math: record math formula tokens
        if domain == "math":
            logical_terms_list = [t for t in important_terms if t.startswith("MATH:") or t.startswith("EQ:")]
        else:
            logical_terms_list = [
                term for term in important_terms
                if term in LOGICAL_KEYWORDS and term not in COMMON_LOGICAL_WORDS
            ]

        all_important_terms.append({
            "sample_id": sample.get("sample_id", f"sample_{idx}"),
            "source_dataset": sample.get("source_dataset"),
            "source_index": sample.get("source_index"),
            "target_answer": sample.get("target_answer"),
            "important_terms": important_terms,
            "logical_terms": logical_terms_list,
            "num_traces": len(traces),
            "tfidf_scores": {term: score for term, score in selected_terms},
        })

    return {
        "metadata": {
            "total_samples": len(results),
            "top_k": top_k if top_k is not None else "unlimited",
            "min_tf": min_tf,
            "min_idf": min_idf,
            "domain": domain,
            "prioritize_logical": prioritize_logical,
            "logical_keywords_count": len(LOGICAL_KEYWORDS),
            "filtered_common_words": len(COMMON_LOGICAL_WORDS),
            "note": (
                "math domain: extracts LaTeX formulas and bare equations as complete tokens, supplemented by math operation words"
                if domain == "math" else
                "Filtered overly common logical words (e.g. because, also, approve, disprove); prioritizes semantically rich vocabulary"
            ),
        },
        "results": all_important_terms,
    }


def main():
    parser = argparse.ArgumentParser(
        description="TF-IRF (Term Frequency-Inverse Reasoning Frequency): dump the consensus terms of reasoning traces for inspection"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON file path (containing reasoning traces)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="important_terms.json",
        help="Output JSON file path (relative paths resolve under the results root)",
    )
    parser.add_argument(
        "--top_k",
        type=lambda x: None if x is None else int(x),
        default=None,
        nargs="?",
        help="Number of top important terms to extract per sample (default None: no limit, extract all terms meeting criteria)",
    )
    parser.add_argument(
        "--min_tf",
        type=float,
        default=0.001,
        help="Minimum term frequency threshold (default 0.001)",
    )
    parser.add_argument(
        "--min_idf",
        type=float,
        default=0.1,
        help="Minimum inverse document frequency threshold (default 0.1)",
    )
    parser.add_argument(
        "--min_tfidf",
        type=float,
        default=0.0,
        help="TF-IRF importance floor; 0.0 (default) applies no floor, which is how the "
             "reported runs scored terms here. The paper's alpha=0.01 is applied at "
             "synthesis (Module III --min_tfidf)",
    )
    parser.add_argument(
        "--no_prioritize_logical",
        action="store_true",
        help="Do not prioritize logical words; sort purely by TF-IRF score",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="logical",
        choices=["logical", "math"],
        help="Reasoning domain: 'logical' (default) or 'math' (uses LaTeX-formula-aware tokenizer)",
    )

    args = parser.parse_args()

    # Read input file
    input_path = _cfg.resolve_input(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Reading file: {input_path}")
    with input_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Handle data format: if dict with metadata, extract results; otherwise use directly
    if isinstance(raw_data, dict) and "results" in raw_data:
        data = raw_data["results"]
    else:
        data = raw_data

    print(f"Extracting important terms...")
    print(f"  - Top K: {args.top_k if args.top_k else 'unlimited (extract all terms meeting criteria)'}")
    print(f"  - Min TF: {args.min_tf}")
    print(f"  - Min IDF: {args.min_idf}")
    print(f"  - Min TF-IRF: {args.min_tfidf} (only terms above this threshold are considered semantically rich)")
    print(f"  - Prioritize logical words: {not args.no_prioritize_logical}")

    # Extract important terms
    result = extract_important_terms(
        data,
        top_k=args.top_k,
        min_tf=args.min_tf,
        min_idf=args.min_idf,
        min_tfidf=args.min_tfidf,
        prioritize_logical=not args.no_prioritize_logical,
        domain=args.domain,
    )

    # Save results
    output_path = _cfg.resolve_output(args.output)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Print statistics
    if "results" in result:
        results = result["results"]
        total_samples = len(results)
        samples_with_terms = sum(1 for r in results if r.get("important_terms"))
        avg_terms_per_sample = np.mean([len(r.get("important_terms", [])) for r in results])
        avg_logical_terms = np.mean([len(r.get("logical_terms", [])) for r in results])

        print(f"\nDone!")
        print(f"Statistics:")
        print(f"  - Total samples: {total_samples}")
        print(f"  - Samples with terms: {samples_with_terms}")
        print(f"  - Average terms per sample: {avg_terms_per_sample:.2f}")
        print(f"  - Average logical terms per sample: {avg_logical_terms:.2f}")
        print(f"\nResults saved to: {output_path}")

        # Show sample important terms for first few entries
        print(f"\nImportant terms preview:")
        for i, sample_result in enumerate(results[:3]):
            if sample_result.get("important_terms"):
                print(f"\n  Sample {i+1} ({sample_result.get('sample_id')}):")
                terms = sample_result["important_terms"][:10]
                logical = sample_result.get("logical_terms", [])[:5]
                print(f"    Important terms: {', '.join(terms)}")
                if logical:
                    print(f"    Logical terms: {', '.join(logical)}")


if __name__ == "__main__":
    main()
