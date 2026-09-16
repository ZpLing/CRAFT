#!/usr/bin/env python3
"""
step2_extract_terms.py  (CRAFT Pipeline — Step 2: TF-IDF Term Extraction)
---------------------------------------------------------------------------
Extract important logical terms and domain-specific vocabulary from reasoning traces.

This module is used internally by Step 3.1 (anomaly filter) and Step 5 (synthesis).
It implements TF-IDF term extraction: terms that appear frequently within a single
sample's traces but rarely across other samples' traces are considered important.

Key features:
- TF-IDF based extraction: high within-sample TF, low across-sample IDF
- Averages term frequencies across k traces per sample before extraction
- Supports both logical domain (LOGICAL_KEYWORDS) and math domain (LaTeX formula tokens)
- Outputs important term lists per sample
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

import importlib.util as _ilu
_cfg_path = Path(__file__).resolve().parents[1] / "config.py"
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


def _normalise_latex_token(expr: str) -> str:
    """Normalise a LaTeX expression into a comparable token string.

    Strategy:
    - Strip whitespace
    - Preserve structure (variable names, numbers, operators)
    - Unify common equivalent forms (e.g. \\cdot → *)
    - Discard tokens shorter than 3 chars after stripping
    """
    expr = expr.strip()
    # Remove bare LaTeX commands (e.g. \\frac itself; keep its arguments)
    expr = _LATEX_COMMANDS.sub('', expr).strip()
    # Collapse whitespace
    expr = re.sub(r'\s+', ' ', expr)
    return expr


# Regex for extracting inline/display LaTeX formulas
_LATEX_INLINE  = re.compile(r'\$\$(.+?)\$\$|\$(.+?)\$', re.DOTALL)
_LATEX_DISPLAY = re.compile(r'\\\[(.+?)\\\]|\\\((.+?)\\\)', re.DOTALL)
# Bare equations: composed of letters/digits/basic operators, containing =, no $ needed
_BARE_EQUATION = re.compile(
    r'(?<![a-zA-Z])([a-zA-Z0-9\+\-\*/\^\(\)\.\{\}\\,\s]{2,}?'
    r'\s*=\s*'
    r'[a-zA-Z0-9\+\-\*/\^\(\)\.\{\}\\,\s]{1,})(?=[,\.\n\)\s]|$)'
)


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
    for m in _LATEX_INLINE.finditer(text):
        expr = m.group(1) or m.group(2) or ''
        norm = _normalise_latex_token(expr)
        if len(norm) >= 3:
            tokens.append('MATH:' + norm)

    for m in _LATEX_DISPLAY.finditer(text):
        expr = m.group(1) or m.group(2) or ''
        norm = _normalise_latex_token(expr)
        if len(norm) >= 3:
            tokens.append('MATH:' + norm)

    # ── 2. Extract bare equations (outside $, but containing =) ────────────────
    # Mask already-processed LaTeX regions to avoid double-counting
    text_no_latex = _LATEX_INLINE.sub('', text)
    text_no_latex = _LATEX_DISPLAY.sub('', text_no_latex)

    for m in _BARE_EQUATION.finditer(text_no_latex):
        expr = m.group(1).strip()
        norm = _normalise_latex_token(expr)
        if len(norm) >= 3 and '=' in norm:
            tokens.append('EQ:' + norm)

    # ── 3. Extract mathematical operation words (discriminative verbs/nouns) ────────────────────────────
    words = re.findall(r'\b[a-z]+\b', text.lower())
    for w in words:
        if w in MATH_OPERATION_WORDS and w not in MATH_COMMON_WORDS:
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

    # Convert to lowercase
    text = text.lower()

    # Extract words (letters, digits, hyphens)
    tokens = re.findall(r'\b[a-z0-9]+(?:\-[a-z0-9]+)*\b', text)

    # Filter out stopwords
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

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


def calculate_tfidf_for_sample(
    sample_traces: List[Dict[str, Any]],
    all_trace_documents: List[List[str]],
    min_tf: float = 0.001,
    min_idf: float = 0.1,
    domain: str = "logical",
) -> Dict[str, float]:
    """
    Compute TF-IDF scores for a single sample.

    Procedure:
    1. Compute term frequency (TF) for each trace individually
    2. Average TF values across k traces to get the sample-level average TF
    3. Compute IDF for each term (relative to all traces, not just this sample)
    4. TF-IDF = average_TF * IDF

    Args:
        sample_traces: All traces for this sample
        all_trace_documents: Document list for all traces (one document per trace, used for IDF)
        min_tf: Minimum TF threshold
        min_idf: Minimum IDF threshold

    Returns:
        Dictionary mapping terms to their TF-IDF scores
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
        min_tfidf: Minimum TF-IDF score threshold (only terms above this are considered semantically rich)
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

    # Step 2: Compute TF-IDF for each sample
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
                # Sort purely by TF-IDF score (common logical words already filtered)
                selected_terms = sorted(filtered_tfidf_scores.items(), key=lambda x: x[1], reverse=True)
        else:
            # Limit to top-k
            if prioritize_logical:
                # Prioritize semantically rich logical words, then fill with domain terms
                semantic_ratio = 0.3  # 30% semantically rich logical words, 70% domain terms
                num_semantic = max(1, int(top_k * semantic_ratio))
                selected_terms = sorted_semantic_logical[:num_semantic] + sorted_regular[:top_k - num_semantic]
            else:
                # Sort purely by TF-IDF score (common logical words already filtered)
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
        description="Extract important logical words and terms from reasoning traces"
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
        help="Minimum TF-IDF score threshold; only terms above this are considered semantically rich (default 0.0, i.e. no filtering)",
    )
    parser.add_argument(
        "--no_prioritize_logical",
        action="store_true",
        help="Do not prioritize logical words; sort purely by TF-IDF score",
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
    print(f"  - Min TF-IDF: {args.min_tfidf} (only terms above this threshold are considered semantically rich)")
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
