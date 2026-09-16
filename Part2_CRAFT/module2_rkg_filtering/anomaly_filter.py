#!/usr/bin/env python3
"""
step3_1_anomaly_filter.py  (CRAFT Pipeline — Step 3.1: Group Relative Anomaly Filtering)
------------------------------------------------------------------------------------------
Detect and remove anomalous reasoning steps using GRPO-inspired z-score filtering.

Two detection methods are provided:

Method 1: supervised (same-position comparison)
- Compares steps at the same position across traces (requires matching step counts)
- Stricter, suitable when trace lengths are consistent

Method 2: unsupervised (cross-position comparison, GRPO-style)
- Cross-position comparison across traces (no alignment required)
- More flexible for variable-length traces
- Uses group relative policy optimization (GRPO) style relative comparison

Core functionality:
1. Extract important terms per step (via step2_extract_terms.py)
2. Compare steps across traces within the same sample to detect anomalies
3. Flag steps whose term composition deviates significantly (z-score < threshold)
4. Remove anomalous steps; report before/after step counts

Also supports method="rkg" (RKG structural + edge-frequency anomaly detection),
which requires output from step3_2_build_rkg.py.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import importlib.util as _ilu
_cfg_path = _Path(__file__).resolve().parents[1] / "config.py"
_spec = _ilu.spec_from_file_location("_part_config", _cfg_path)
_cfg  = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_cfg)

# Reuse functions from step2_extract_terms.py (Step 2: TF-IDF Term Extraction)
from module1_trace_generation.extract_terms import (
    tokenize_text,
    calculate_tf,
    calculate_idf,
    DocFreqTable,
    FlatDocFreqTable,
    resolve_df_table_path,
    check_df_table,
    iter_equations,
    safe_parse_expr,
    COMMON_LOGICAL_WORDS,
    LOGICAL_KEYWORDS,
    MATH_COMMON_WORDS,
)

# SymPy (optional, only used for math domain)
try:
    import sympy
    # Parsing goes through safe_parse_expr(); only simplify() is needed here.
    _SYMPY_AVAILABLE = True
except ImportError:
    _SYMPY_AVAILABLE = False

STEP_PATTERN = re.compile(r"^Step\s*\d+\s*:", re.IGNORECASE | re.MULTILINE)


def parse_steps_from_trace(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse individual steps from a trace."""
    reasoning_steps = trace.get("reasoning_steps", [])
    reasoning_text = trace.get("reasoning_text", "") or trace.get("raw_response", "")

    parsed_steps = []

    if reasoning_steps and isinstance(reasoning_steps, list):
        # If steps are already parsed
        for idx, step_text in enumerate(reasoning_steps):
            if step_text and isinstance(step_text, str):
                parsed_steps.append({
                    "step_number": idx + 1,
                    "step_text": step_text.strip(),
                })
    elif reasoning_text:
        # Parse steps from text
        lines = reasoning_text.split('\n')
        current_step = None
        current_text = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this is the start of a step
            match = STEP_PATTERN.match(line)
            if match:
                # Save the previous step
                if current_step is not None:
                    parsed_steps.append({
                        "step_number": current_step,
                        "step_text": '\n'.join(current_text).strip(),
                    })

                # Start a new step
                step_num_match = re.search(r'\d+', line)
                if step_num_match:
                    current_step = int(step_num_match.group())
                    current_text = [line]
                else:
                    current_step = len(parsed_steps) + 1
                    current_text = [line]
            else:
                if current_step is not None:
                    current_text.append(line)

        # Save the last step
        if current_step is not None:
            parsed_steps.append({
                "step_number": current_step,
                "step_text": '\n'.join(current_text).strip(),
            })

    return parsed_steps


def build_global_df_table(
    samples: List[Dict[str, Any]],
    domain: str = "logical",
    normalize: bool = False,
) -> DocFreqTable:
    """Build one DF table over every step of every sample — the global IRF setting.

    The document unit stays a single step, exactly as in the within-sample setting, so the
    only thing that changes is the corpus: a term's IDF now reflects how common it is
    across problems rather than how widely it spreads inside one problem. Traces are read
    with the same precedence process_sample() uses, so the corpus covers the steps that
    actually get scored.
    """
    table = DocFreqTable(normalize=normalize, domain=domain)
    for sample in samples:
        traces = sample.get("traces", []) or sample.get("cleaned_traces", [])
        for trace in traces:
            for step_info in parse_steps_from_trace(trace):
                step_text = step_info["step_text"]
                if step_text:
                    table.add_document(tokenize_text(step_text, domain=domain))
    return table


def extract_step_terms(
    step_text: str,
    all_step_documents: List[List[str]],
    min_tf: float = 0.001,
    min_idf: float = 0.1,
    min_tfidf: float = 0.0,
    domain: str = "logical",
    df_table: Optional[DocFreqTable] = None,
) -> List[str]:
    """Extract important terms for a single step (domain-aware).

    df_table: when given, IDF is scored against that corpus (the global IRF setting) and
    all_step_documents is ignored; when None, IDF comes from all_step_documents, i.e. the
    within-sample setting.
    """
    if not step_text:
        return []

    tokens = tokenize_text(step_text, domain=domain)
    if not tokens:
        return []

    tf_dict = calculate_tf(tokens)
    common_filter = MATH_COMMON_WORDS if domain == "math" else COMMON_LOGICAL_WORDS

    step_terms = []
    for term, tf_value in tf_dict.items():
        if tf_value < min_tf:
            continue
        # math: do not filter MATH:/EQ: prefix tokens; filter operation words via common_filter
        if not (domain == "math" and (term.startswith("MATH:") or term.startswith("EQ:"))):
            if term in common_filter:
                continue

        idf = (df_table.idf(term) if df_table is not None
               else calculate_idf(all_step_documents, term))
        if idf < min_idf:
            continue

        tfidf_score = tf_value * idf
        if tfidf_score < min_tfidf:
            continue

        step_terms.append((term, tfidf_score))

    step_terms.sort(key=lambda x: x[1], reverse=True)
    return [term for term, _ in step_terms[:20]]


def extract_terms_for_all_steps(
    sample_traces: List[Dict[str, Any]],
    min_tf: float = 0.001,
    min_idf: float = 0.1,
    min_tfidf: float = 0.0,
    domain: str = "logical",
    df_table: Optional[DocFreqTable] = None,
    idf_norm: bool = False,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Extract terms for all steps across all traces in a sample.

    Returns:
        Dict[step_number, List[Dict]]:
        {
            step_number: [
                {
                    "trace_idx": int,
                    "step_text": str,
                    "terms": List[str],
                    "terms_set": Set[str]
                },
                ...
            ]
        }
    """
    # Step 1: collect all step texts (for IDF computation)
    all_step_texts = []
    all_steps_by_number = defaultdict(list)

    for trace_idx, trace in enumerate(sample_traces):
        parsed_steps = parse_steps_from_trace(trace)
        for step_info in parsed_steps:
            step_number = step_info["step_number"]
            step_text = step_info["step_text"]

            if step_text:
                all_step_texts.append(step_text)
                all_steps_by_number[step_number].append({
                    "trace_idx": trace_idx,
                    "step_text": step_text,
                })

    # Under the global setting df_table already carries the corpus; otherwise build this
    # sample's own, which is the within-sample setting. Either way the terms below are
    # scored against a table, so no raw document list needs to reach them.
    if df_table is None:
        df_table = DocFreqTable.from_documents(
            [tokenize_text(text, domain=domain) for text in all_step_texts],
            normalize=idf_norm,
        )

    # Step 2: extract terms for each step
    steps_with_terms = defaultdict(list)

    for step_number, step_list in all_steps_by_number.items():
        for step_info in step_list:
            terms = extract_step_terms(
                step_info["step_text"],
                [],
                min_tf=min_tf,
                min_idf=min_idf,
                min_tfidf=min_tfidf,
                domain=domain,
                df_table=df_table,
            )

            steps_with_terms[step_number].append({
                "trace_idx": step_info["trace_idx"],
                "step_text": step_info["step_text"],
                "terms": terms,
                "terms_set": set(terms),
            })

    return steps_with_terms


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def weighted_jaccard_similarity(
    set1: Set[str],
    set2: Set[str],
    tfidf1: Dict[str, float],
    tfidf2: Dict[str, float],
) -> float:
    """
    Compute weighted Jaccard similarity (inspired by GRPO).

    Uses TF-IDF scores as weights so that steps containing core terms
    receive higher similarity scores.

    Args:
        set1, set2: term sets
        tfidf1, tfidf2: corresponding TF-IDF score dicts

    Returns:
        Weighted Jaccard similarity [0, 1]
    """
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0

    # Intersection: sum of TF-IDF scores for shared terms
    intersection_terms = set1 & set2
    intersection_weight = sum(
        tfidf1.get(term, 0.0) + tfidf2.get(term, 0.0)
        for term in intersection_terms
    ) / 2.0  # take average

    # Union: sum of TF-IDF scores for all terms
    union_terms = set1 | set2
    union_weight = sum(
        max(tfidf1.get(term, 0.0), tfidf2.get(term, 0.0))
        for term in union_terms
    )

    return intersection_weight / union_weight if union_weight > 0 else 0.0


def compute_sample_consensus_core(
    steps_with_terms: List[Dict[str, Any]],
    consensus_threshold: float = 0.3,
) -> Set[str]:
    """
    Compute the consensus core for a sample.

    Inspired by GRPO: find terms that appear frequently within the sample
    to form a "semantic consensus core".

    Args:
        steps_with_terms: list of step info dicts for all steps
        consensus_threshold: term frequency threshold (e.g. 0.3 means the term
                             appears in at least 30% of steps)

    Returns:
        Set of consensus core terms
    """
    if not steps_with_terms:
        return set()

    # Count occurrences of each term
    term_counts = Counter()
    total_steps = len(steps_with_terms)

    for step_info in steps_with_terms:
        terms_set = step_info.get("terms_set", set())
        term_counts.update(terms_set)

    # Keep terms whose frequency >= threshold
    consensus_core = {
        term for term, count in term_counts.items()
        if count / total_steps >= consensus_threshold
    }

    return consensus_core


def detect_anomalous_steps_supervised(
    steps_with_terms: Dict[int, List[Dict[str, Any]]],
    similarity_threshold: float = 0.3,
) -> Set[Tuple[int, int]]:
    """
    Detect anomalous steps (Method 1: same-position comparison).

    Requires one-to-one step correspondence across traces
    (only compares steps with the same step_number).

    Args:
        steps_with_terms: step info for each step_number across all traces
        similarity_threshold: steps below this similarity are flagged as anomalous

    Returns:
        Set[Tuple[step_number, trace_idx]]: anomalous steps
    """
    """
    Detect anomalous steps.

    Args:
        steps_with_terms: step info for each step_number across all traces
        similarity_threshold: steps below this similarity are flagged as anomalous

    Returns:
        Set[Tuple[step_number, trace_idx]]: anomalous steps
    """
    anomalous_steps = set()

    for step_number, step_list in steps_with_terms.items():
        if len(step_list) < 2:
            # Only one trace has this step — cannot compare, skip
            continue

        # For each trace's step, check similarity against other traces' steps
        for i, step_i in enumerate(step_list):
            trace_idx_i = step_i["trace_idx"]
            terms_set_i = step_i["terms_set"]

            if not terms_set_i:
                # Nothing to compare: the tokenizer produced no terms for this step, which
                # says the step could not be measured, not that it is anomalous. A logical
                # step written purely in symbols, or a math step written purely in prose,
                # lands here through no fault of its own, so it is kept and left unscored.
                continue

            # Compute similarity against other traces
            max_similarity = 0.0

            for j, step_j in enumerate(step_list):
                if i == j:
                    continue

                terms_set_j = step_j["terms_set"]
                if not terms_set_j:
                    continue

                # Compute Jaccard similarity
                similarity = jaccard_similarity(terms_set_i, terms_set_j)
                max_similarity = max(max_similarity, similarity)

            # If max similarity is below threshold, flag as anomalous
            if max_similarity < similarity_threshold:
                anomalous_steps.add((step_number, trace_idx_i))

    return anomalous_steps


def extract_step_terms_with_tfidf(
    step_text: str,
    all_step_documents: List[List[str]],
    min_tf: float = 0.001,
    min_idf: float = 0.1,
    min_tfidf: float = 0.0,
    domain: str = "logical",
    df_table: Optional[DocFreqTable] = None,
) -> Tuple[List[str], Dict[str, float]]:
    """Extract important terms and their TF-IDF scores for a single step (domain-aware).

    df_table follows the same convention as extract_step_terms().
    """
    if not step_text:
        return [], {}

    tokens = tokenize_text(step_text, domain=domain)
    if not tokens:
        return [], {}

    tf_dict = calculate_tf(tokens)
    common_filter = MATH_COMMON_WORDS if domain == "math" else COMMON_LOGICAL_WORDS

    step_terms = []
    tfidf_scores = {}

    for term, tf_value in tf_dict.items():
        if tf_value < min_tf:
            continue
        if not (domain == "math" and (term.startswith("MATH:") or term.startswith("EQ:"))):
            if term in common_filter:
                continue

        idf = (df_table.idf(term) if df_table is not None
               else calculate_idf(all_step_documents, term))
        if idf < min_idf:
            continue

        tfidf_score = tf_value * idf
        if tfidf_score < min_tfidf:
            continue

        step_terms.append((term, tfidf_score))
        tfidf_scores[term] = tfidf_score

    step_terms.sort(key=lambda x: x[1], reverse=True)
    terms_list = [term for term, _ in step_terms[:20]]

    return terms_list, tfidf_scores


def collect_all_steps_with_terms(
    sample_traces: List[Dict[str, Any]],
    min_tf: float = 0.001,
    min_idf: float = 0.1,
    min_tfidf: float = 0.0,
    domain: str = "logical",
    df_table: Optional[DocFreqTable] = None,
    idf_norm: bool = False,
) -> List[Dict[str, Any]]:
    """
    Collect all steps from all traces in a sample and extract their terms
    (used for cross-step comparison).

    Returns:
        List[Dict]: info for each step
        {
            "trace_idx": int,
            "step_number": int,
            "step_text": str,
            "terms": List[str],
            "terms_set": Set[str],
            "tfidf_scores": Dict[str, float]  # added: TF-IDF scores
        }
    """
    # Step 1: collect all step texts (for IDF computation)
    all_step_texts = []
    all_steps_info = []

    for trace_idx, trace in enumerate(sample_traces):
        parsed_steps = parse_steps_from_trace(trace)
        for step_info in parsed_steps:
            step_text = step_info["step_text"]
            if step_text:
                all_step_texts.append(step_text)
                all_steps_info.append({
                    "trace_idx": trace_idx,
                    "step_number": step_info["step_number"],
                    "step_text": step_text,
                })

    # Same corpus choice as extract_terms_for_all_steps(): global table, or this sample's own.
    if df_table is None:
        df_table = DocFreqTable.from_documents(
            [tokenize_text(text, domain=domain) for text in all_step_texts],
            normalize=idf_norm,
        )

    # Step 2: extract terms and TF-IDF scores for each step
    steps_with_terms = []

    for step_info in all_steps_info:
        terms, tfidf_scores = extract_step_terms_with_tfidf(
            step_info["step_text"],
            [],
            min_tf=min_tf,
            min_idf=min_idf,
            min_tfidf=min_tfidf,
            domain=domain,
            df_table=df_table,
        )

        steps_with_terms.append({
            "trace_idx": step_info["trace_idx"],
            "step_number": step_info["step_number"],
            "step_text": step_info["step_text"],
            "terms": terms,
            "terms_set": set(terms),
            "tfidf_scores": tfidf_scores,  # added: TF-IDF scores
        })

    return steps_with_terms


def verify_step_math(step_text: str) -> Optional[bool]:
    """Use SymPy to verify whether equations in a math step are correct.

    Returns:
        True  — all parseable equations hold (step is valid)
        False — at least one equation does not hold (step has an error)
        None  — no equations could be parsed (fall back to frequency method)
    """
    if not _SYMPY_AVAILABLE or not step_text:
        return None

    # Reuse the tokenizer's equation pattern. The one that used to live here let whitespace
    # float anywhere, so it cut equations out of prose -- "Using Betty = 16", "80 pages =
    # 960 pages" -- and implicit multiplication turned the units into a product of letter
    # symbols that could never cancel. Every such step was force-deleted: 322 of 2798 GSM8K
    # steps failed against 78 that passed, a ratio no solver of this quality produces.
    verified_any = False
    for equation in iter_equations(step_text):
        if equation.count("=") != 1:
            continue
        lhs_str, rhs_str = equation.split("=")
        # Skip "Step X" style patterns
        if re.match(r"(?i)\s*step\s*\d+", lhs_str):
            continue

        # Judge closed arithmetic only. A letter on either side is either a quantity the
        # step is naming or a unit, and SymPy reads several single letters as constants
        # rather than symbols -- I is the imaginary unit, E is Euler's number -- so "7J =
        # 70" comes back with no free symbol and a non-zero difference, which reads as an
        # arithmetic error when it is nothing of the kind. Anything with letters goes to
        # the z-score instead.
        if re.search(r"[A-Za-z]", equation):
            continue

        lhs, rhs = safe_parse_expr(lhs_str), safe_parse_expr(rhs_str)
        if lhs is None or rhs is None:
            continue  # refused or unparseable, skip this equation
        try:
            diff = sympy.simplify(lhs - rhs)
        except Exception:
            continue

        # A leftover symbol means the step is naming a quantity rather than asserting
        # arithmetic. "Let the total be T = 3x" cannot simplify to zero and is not an
        # error, so only closed arithmetic is judged; the rest goes to the z-score.
        if diff.free_symbols:
            continue

        verified_any = True
        if diff != 0:
            return False  # found an incorrect equation — flag as anomalous

    return True if verified_any else None


def detect_anomalous_steps_unsupervised(
    steps_with_terms: List[Dict[str, Any]],
    similarity_threshold: float = 0.3,
    min_similar_steps: int = 2,
    use_grpo_optimization: bool = True,
    z_score_threshold: float = -1.5,
    consensus_threshold: float = 0.3,
    use_weighted_similarity: bool = True,
    domain: str = "logical",
) -> Set[Tuple[int, int]]:
    """
    Detect anomalous steps (Method 2: cross-step comparison, GRPO-style, optimized).

    Added:
    - When domain="math", first validate equations with SymPy:
        * Validation fails (incorrect equation) -> force flag as anomalous
        * Validation passes -> force keep (skip z-score filtering)
        * Cannot validate -> fall back to GRPO z-score
    - When std < 0.05, skip z-score filtering (all traces are highly consistent)

    Args:
        steps_with_terms: list of step info dicts for all steps
        similarity_threshold: similarity threshold (fallback when not using GRPO optimization)
        min_similar_steps: minimum number of similar steps required to be considered normal
        use_grpo_optimization: whether to use GRPO optimization (default True)
        z_score_threshold: z-score threshold; steps below this are flagged as anomalous (default -1.5)
        consensus_threshold: consensus core threshold (default 0.3, i.e. term appears in >= 30% of steps)
        use_weighted_similarity: whether to use weighted Jaccard similarity (default True)
        domain: "logical" or "math"

    Returns:
        Set[Tuple[trace_idx, step_number]]: anomalous steps
    """
    anomalous_steps = set()

    if len(steps_with_terms) < min_similar_steps:
        return anomalous_steps

    # ===== Math domain: SymPy pre-validation =====
    # Verification only vouches for a step; it never condemns one. Reading an equation out
    # of prose is lossy in a way no pattern fixes -- the left side of "25% of 36 =
    # (25/100)*36" is English -- so a failed check says the step could not be read, not
    # that its arithmetic is wrong, and four rounds of fixes each left a smaller set of
    # correct steps being deleted outright. A step that verifies skips the z-score;
    # everything else, failures included, is judged by consensus like any other step.
    math_force_keep: Set[Tuple[int, int]] = set()

    if domain == "math" and _SYMPY_AVAILABLE:
        for step_info in steps_with_terms:
            if verify_step_math(step_info["step_text"]) is True:
                math_force_keep.add((step_info["trace_idx"], step_info["step_number"]))

        steps_for_grpo = [
            s for s in steps_with_terms
            if (s["trace_idx"], s["step_number"]) not in math_force_keep
        ]
    else:
        steps_for_grpo = steps_with_terms

    if not steps_for_grpo:
        return anomalous_steps

    if use_grpo_optimization:
        # ========== GRPO optimized version ==========

        consensus_core = compute_sample_consensus_core(
            steps_for_grpo,
            consensus_threshold=consensus_threshold,
        )

        step_scores = []

        for i, step_i in enumerate(steps_for_grpo):
            trace_idx_i = step_i["trace_idx"]
            step_number_i = step_i["step_number"]
            terms_set_i = step_i["terms_set"]
            tfidf_i = step_i.get("tfidf_scores", {})

            if not terms_set_i:
                # Unscoreable, not anomalous — see the note in the supervised detector.
                continue

            if use_weighted_similarity and tfidf_i:
                consensus_tfidf = {
                    term: np.mean([
                        s.get("tfidf_scores", {}).get(term, 0.0)
                        for s in steps_for_grpo
                        if term in s.get("terms_set", set())
                    ])
                    for term in consensus_core
                }
                score = weighted_jaccard_similarity(
                    terms_set_i, consensus_core, tfidf_i, consensus_tfidf
                )
            else:
                score = jaccard_similarity(terms_set_i, consensus_core)

            step_scores.append({
                "trace_idx": trace_idx_i,
                "step_number": step_number_i,
                "score": score,
                "index": i,
            })

        if not step_scores:
            return anomalous_steps

        scores = [s["score"] for s in step_scores]
        mean_score = np.mean(scores)
        std_score = np.std(scores) if len(scores) > 1 else 0.0

        # *** Fix: small std means all traces are highly consistent — skip filtering ***
        if std_score < 0.05:
            return anomalous_steps

        for step_score_info in step_scores:
            score = step_score_info["score"]
            z_score = (score - mean_score) / std_score

            if z_score < z_score_threshold:
                anomalous_steps.add((
                    step_score_info["trace_idx"],
                    step_score_info["step_number"]
                ))

        return anomalous_steps

    else:
        # ========== Original version (without GRPO optimization) ==========

        for i, step_i in enumerate(steps_for_grpo):
            trace_idx_i = step_i["trace_idx"]
            step_number_i = step_i["step_number"]
            terms_set_i = step_i["terms_set"]
            tfidf_i = step_i.get("tfidf_scores", {})

            if not terms_set_i:
                # Unscoreable, not anomalous — see the note in the supervised detector.
                continue

            similarities = []

            for j, step_j in enumerate(steps_for_grpo):
                if i == j:
                    continue

                terms_set_j = step_j["terms_set"]
                tfidf_j = step_j.get("tfidf_scores", {})

                if not terms_set_j:
                    continue

                if use_weighted_similarity and tfidf_i and tfidf_j:
                    similarity = weighted_jaccard_similarity(
                        terms_set_i, terms_set_j, tfidf_i, tfidf_j
                    )
                else:
                    similarity = jaccard_similarity(terms_set_i, terms_set_j)

                similarities.append(similarity)

            if not similarities:
                anomalous_steps.add((trace_idx_i, step_number_i))
                continue

            max_similarity = max(similarities)
            similar_count = sum(1 for s in similarities if s >= similarity_threshold)

            if max_similarity < similarity_threshold or similar_count < min_similar_steps:
                anomalous_steps.add((trace_idx_i, step_number_i))

        return anomalous_steps


def remove_anomalous_steps(
    sample_traces: List[Dict[str, Any]],
    anomalous_steps: Set[Tuple[int, int]],
) -> List[Dict[str, Any]]:
    """Remove anomalous steps from traces (always preserving the final conclusion step)."""
    cleaned_traces = []

    for trace_idx, trace in enumerate(sample_traces):
        parsed_steps = parse_steps_from_trace(trace)

        if not parsed_steps:
            cleaned_traces.append(trace.copy())
            continue

        # Identify the last step (conclusion) — highest step_number
        last_step_number = max(step_info["step_number"] for step_info in parsed_steps)

        # Filter out anomalous steps but always keep the last step
        cleaned_steps = []
        for step_info in parsed_steps:
            step_number = step_info["step_number"]

            # Always keep the last step
            if step_number == last_step_number:
                cleaned_steps.append(step_info)
                continue

            # Support both formats: (step_number, trace_idx) or (trace_idx, step_number)
            if (step_number, trace_idx) not in anomalous_steps and \
               (trace_idx, step_number) not in anomalous_steps:
                cleaned_steps.append(step_info)

        # Reconstruct the trace
        cleaned_trace = trace.copy()

        # Update reasoning_steps
        cleaned_trace["reasoning_steps"] = [
            step["step_text"] for step in cleaned_steps
        ]

        # Update reasoning_text
        if cleaned_steps:
            cleaned_trace["reasoning_text"] = "\n".join([
                f"Step {step['step_number']}: {step['step_text']}"
                for step in cleaned_steps
            ])
        else:
            cleaned_trace["reasoning_text"] = ""

        # Record number of removed steps
        cleaned_trace["original_num_steps"] = len(parsed_steps)
        cleaned_trace["cleaned_num_steps"] = len(cleaned_steps)
        cleaned_trace["removed_steps"] = len(parsed_steps) - len(cleaned_steps)

        cleaned_traces.append(cleaned_trace)

    return cleaned_traces


#########################
# RKG-based Anomaly Detection
#########################

def detect_structural_anomalies(
    trace_rkg: Dict[str, Any],
    consensus_rkg: Dict[str, Any],
) -> Set[str]:
    """Phase 1: Detect structurally anomalous node IDs within an RKG.

    Anomaly types (purely structural, independent of content majority-vote assumptions):
    1. Isolated nodes  — non-Fact nodes with neither incoming nor outgoing edges
                         (disconnected from the reasoning chain)
    2. Dangling references — an incoming edge whose src node does not exist
                             in this trace's node set
    3. Forward references  — step B has an incoming edge from a step with a
                             higher step number than B (referencing a future conclusion)
    """
    nodes = {n["id"]: n for n in trace_rkg.get("nodes", [])}
    edges = trace_rkg.get("edges", [])

    in_degree:  defaultdict = defaultdict(int)
    out_degree: defaultdict = defaultdict(int)
    in_edges:   Dict[str, List[str]] = defaultdict(list)

    for e in edges:
        src, dst = e["src"], e["dst"]
        out_degree[src] += 1
        in_degree[dst] += 1
        in_edges[dst].append(src)

    anomalous: Set[str] = set()

    for nid, node in nodes.items():
        if node.get("type") == "fact":
            continue  # Fact nodes naturally have no incoming edges

        # 1. Isolated nodes
        if in_degree[nid] == 0 and out_degree[nid] == 0:
            anomalous.add(nid)
            continue

        step_num = node.get("step_number")

        # 2. Dangling references & 3. Forward references
        for src_id in in_edges[nid]:
            if src_id not in nodes:
                anomalous.add(nid)  # dangling reference
                break
            src_node = nodes[src_id]
            src_type = src_node.get("type", "step")
            src_step_num = src_node.get("step_number")
            if src_type not in ("fact",) and src_step_num and step_num and src_step_num > step_num:
                anomalous.add(nid)  # forward reference
                break

    return anomalous


def detect_edge_frequency_anomalies(
    trace_rkg: Dict[str, Any],
    consensus_rkg: Dict[str, Any],
    threshold: float = 0.3,
) -> Set[str]:
    """Phase 2: Edge-frequency voting: edges in trace_rkg with frequency < threshold in consensus RKG mark dst anomalous."""
    edge_frequencies = consensus_rkg.get("edge_frequencies", {})
    anomalous: Set[str] = set()

    for edge in trace_rkg.get("edges", []):
        src, dst = edge["src"], edge["dst"]
        key = f"{src}->{dst}"
        freq = edge_frequencies.get(key, 0.0)
        if freq < threshold:
            anomalous.add(dst)

    return anomalous


def detect_underthinking_traces(
    trace_rkgs: List[Dict[str, Any]],
    consensus_rkg: Dict[str, Any],
    underthinking_threshold: float = 0.3,
) -> Set[int]:
    """Phase 0: Detect underthinking traces (those with severely insufficient step counts overall).

    Detection logic:
      Non-Fact node count in consensus RKG = standard reasoning step count (n_consensus)
      Non-Fact node count in a given trace = n_trace
      deficit_ratio = (n_consensus - n_trace) / n_consensus
      If deficit_ratio > underthinking_threshold -> underthinking

    Handling of underthinking traces:
      Do not delete individual steps (they may themselves be correct);
      instead mark the whole trace as down-weighted.
      Add "underthinking": True flag to the cleaned_trace so that
      Step 5 synthesis can reduce this trace's term contribution weight.

    Returns:
        Set[int]: set of trace_idx values for underthinking traces
    """
    consensus_step_nodes = [
        n for n in consensus_rkg.get("nodes", [])
        if n.get("type") != "fact"
    ]
    n_consensus = len(consensus_step_nodes)
    if n_consensus == 0:
        return set()

    underthinking_indices: Set[int] = set()
    for rkg_trace in trace_rkgs:
        trace_idx = rkg_trace.get("trace_idx", 0)
        n_trace = sum(
            1 for n in rkg_trace.get("nodes", []) if n.get("type") != "fact"
        )
        deficit_ratio = (n_consensus - n_trace) / n_consensus
        if deficit_ratio > underthinking_threshold:
            underthinking_indices.add(trace_idx)

    return underthinking_indices


def detect_anomalous_steps_rkg(
    trace_rkgs: List[Dict[str, Any]],
    consensus_rkg: Dict[str, Any],
    consensus_threshold: float = 0.3,
    underthinking_threshold: float = 0.3,
    domain: str = "logical",
) -> Tuple[Set[Tuple[int, str]], Set[int]]:
    """RKG-based anomaly detection entry point.

    Handles two categories of issues simultaneously:
      Phase 0: Underthinking detection (trace overall step count severely insufficient
               -> down-weight, do not delete steps)
      Phase 1: Structural anomalies (isolated / dangling / forward-reference / contradictory
               edges -> delete corresponding steps)
      Phase 2: Edge frequency voting (low-frequency edges -> delete destination steps)
      Phase 3: math domain SymPy validation (optional, forcefully overrides)

    Returns:
        (anomalous_steps, underthinking_traces)
        - anomalous_steps     : Set of (trace_idx, node_id) — specific steps to remove
        - underthinking_traces: Set of trace_idx — traces with overall insufficient steps
                                (marked for down-weighting)
    """
    anomalous: Set[Tuple[int, str]] = set()

    # Phase 0: Underthinking detection
    underthinking_traces = detect_underthinking_traces(
        trace_rkgs, consensus_rkg,
        underthinking_threshold=underthinking_threshold,
    )

    for rkg_trace in trace_rkgs:
        trace_idx = rkg_trace.get("trace_idx", 0)

        # Phase 1: structural anomaly detection
        structural = detect_structural_anomalies(rkg_trace, consensus_rkg)

        # Phase 2: edge frequency voting
        freq_based = detect_edge_frequency_anomalies(
            rkg_trace, consensus_rkg, threshold=consensus_threshold
        )

        combined = structural | freq_based

        # Phase 3: math domain SymPy validation (forceful override)
        if domain == "math" and _SYMPY_AVAILABLE:
            for node in rkg_trace.get("nodes", []):
                if node.get("type") == "fact":
                    continue
                nid = node["id"]
                result = verify_step_math(node.get("text", ""))
                if result is False:
                    combined.add(nid)
                elif result is True and nid in combined:
                    combined.discard(nid)

        for nid in combined:
            anomalous.add((trace_idx, nid))

    return anomalous, underthinking_traces


def remove_anomalous_steps_rkg(
    sample_traces: List[Dict[str, Any]],
    trace_rkgs: List[Dict[str, Any]],
    anomalous: Set[Tuple[int, str]],
    underthinking_traces: Optional[Set[int]] = None,
) -> List[Dict[str, Any]]:
    """Remove RKG-flagged anomalous steps; mark underthinking traces.

    - Erroneous steps (anomalous): delete corresponding steps, always preserve the last step (conclusion)
    - Underthinking traces: do not delete steps; add "underthinking": True flag to cleaned_trace
      Step 5 synthesis reads this flag to reduce this trace's term contribution weight
    """
    underthinking_set = underthinking_traces or set()
    cleaned_traces = []

    for trace_idx, trace in enumerate(sample_traces):
        parsed_steps = parse_steps_from_trace(trace)
        if not parsed_steps:
            cleaned_traces.append(trace.copy())
            continue

        last_step_number = max(s["step_number"] for s in parsed_steps)
        anomalous_node_ids = {nid for (ti, nid) in anomalous if ti == trace_idx}

        cleaned_steps = []
        for step_info in parsed_steps:
            node_id = f"Step{step_info['step_number']}"
            is_last = step_info["step_number"] == last_step_number
            if is_last or node_id not in anomalous_node_ids:
                cleaned_steps.append(step_info)

        cleaned_trace = trace.copy()
        cleaned_trace["reasoning_steps"] = [s["step_text"] for s in cleaned_steps]
        if cleaned_steps:
            cleaned_trace["reasoning_text"] = "\n".join(
                f"Step {s['step_number']}: {s['step_text']}" for s in cleaned_steps
            )
        else:
            cleaned_trace["reasoning_text"] = ""
        cleaned_trace["original_num_steps"] = len(parsed_steps)
        cleaned_trace["cleaned_num_steps"] = len(cleaned_steps)
        cleaned_trace["removed_steps"] = len(parsed_steps) - len(cleaned_steps)

        # Mark underthinking (do not delete steps; Step 5 synthesis will down-weight)
        cleaned_trace["underthinking"] = trace_idx in underthinking_set

        cleaned_traces.append(cleaned_trace)

    return cleaned_traces


def process_sample(
    sample: Dict[str, Any],
    min_tf: float = 0.001,
    min_idf: float = 0.1,
    min_tfidf: float = 0.0,
    similarity_threshold: float = 0.3,
    method: str = "supervised",
    min_similar_steps: int = 2,
    use_grpo_optimization: bool = True,
    z_score_threshold: float = -1.5,
    consensus_threshold: float = 0.3,
    use_weighted_similarity: bool = True,
    domain: str = "logical",
    sample_rkg: Optional[Dict[str, Any]] = None,
    underthinking_threshold: float = 0.3,
    df_table: Optional[DocFreqTable] = None,
    idf_norm: bool = False,
) -> Dict[str, Any]:
    """
    Process a single sample: detect and remove anomalous steps.

    Args:
        sample: sample data dict
        method: "supervised" | "unsupervised" | "rkg"
        sample_rkg: required when method="rkg"; RKG data for this sample (from step3_2_build_rkg.py)
        domain: "logical" or "math"
        df_table: global IRF corpus from build_global_df_table(); None keeps IDF within the sample
        idf_norm: divide IDF by log(N) so thresholds mean the same under either corpus
    """
    traces = sample.get("traces", []) or sample.get("cleaned_traces", [])
    if not traces:
        return {
            "sample_id": sample.get("sample_id"),
            "original_traces": [],
            "cleaned_traces": [],
            "anomalous_steps": [],
            "stats": {
                "original_avg_steps": 0.0,
                "cleaned_avg_steps": 0.0,
                "original_steps_std": 0.0,
                "cleaned_steps_std": 0.0,
                "total_removed_steps": 0,
                "num_anomalous_steps": 0,
                "num_underthinking_traces": 0,
                "method": method,
            },
        }

    anomalous_info = []
    n_underthinking = 0  # only populated for method="rkg"

    if method == "supervised":
        steps_with_terms_dict = extract_terms_for_all_steps(
            traces, min_tf=min_tf, min_idf=min_idf, min_tfidf=min_tfidf, domain=domain,
            df_table=df_table, idf_norm=idf_norm,
        )
        anomalous_steps = detect_anomalous_steps_supervised(
            steps_with_terms_dict, similarity_threshold=similarity_threshold,
        )
        cleaned_traces = remove_anomalous_steps(traces, anomalous_steps)
        for step_num, trace_idx in anomalous_steps:
            step_info = next(
                (s for s in steps_with_terms_dict.get(step_num, []) if s["trace_idx"] == trace_idx),
                None,
            )
            if step_info:
                anomalous_info.append({
                    "step_number": step_num, "trace_idx": trace_idx,
                    "step_text": step_info["step_text"], "terms": list(step_info["terms_set"]),
                })

    elif method == "unsupervised":
        steps_with_terms_list = collect_all_steps_with_terms(
            traces, min_tf=min_tf, min_idf=min_idf, min_tfidf=min_tfidf, domain=domain,
            df_table=df_table, idf_norm=idf_norm,
        )
        anomalous_steps = detect_anomalous_steps_unsupervised(
            steps_with_terms_list,
            similarity_threshold=similarity_threshold,
            min_similar_steps=min_similar_steps,
            use_grpo_optimization=use_grpo_optimization,
            z_score_threshold=z_score_threshold,
            consensus_threshold=consensus_threshold,
            use_weighted_similarity=use_weighted_similarity,
            domain=domain,
        )
        cleaned_traces = remove_anomalous_steps(traces, anomalous_steps)
        for trace_idx, step_num in anomalous_steps:
            step_info = next(
                (s for s in steps_with_terms_list
                 if s["trace_idx"] == trace_idx and s["step_number"] == step_num),
                None,
            )
            if step_info:
                anomalous_info.append({
                    "trace_idx": trace_idx, "step_number": step_num,
                    "step_text": step_info["step_text"], "terms": list(step_info["terms_set"]),
                })

    elif method == "rkg":
        if sample_rkg is None:
            raise ValueError("method='rkg' requires sample_rkg argument (from step3_2_build_rkg.py output)")
        trace_rkgs    = sample_rkg.get("trace_dags", [])
        consensus_rkg = sample_rkg.get("consensus_dag", {})
        anomalous_rkg, underthinking_traces = detect_anomalous_steps_rkg(
            trace_rkgs, consensus_rkg,
            consensus_threshold=consensus_threshold,
            underthinking_threshold=underthinking_threshold,
            domain=domain,
        )
        cleaned_traces = remove_anomalous_steps_rkg(
            traces, trace_rkgs, anomalous_rkg, underthinking_traces
        )
        # Count underthinking traces
        n_underthinking = len(underthinking_traces)
        for trace_idx, node_id in anomalous_rkg:
            rkg_trace = next((d for d in trace_rkgs if d.get("trace_idx") == trace_idx), None)
            node = next((n for n in (rkg_trace or {}).get("nodes", []) if n["id"] == node_id), None)
            anomalous_info.append({
                "trace_idx": trace_idx,
                "node_id":   node_id,
                "step_text": node.get("text", "") if node else "",
                "diagnosis": "overthinking_or_error",
                "step_number": node.get("step_number") if node else None,
            })
        # Also record underthinking traces
        for tidx in underthinking_traces:
            anomalous_info.append({
                "trace_idx": tidx,
                "node_id":   None,
                "step_text": "",
                "diagnosis": "underthinking",
                "step_number": None,
            })
        anomalous_steps = anomalous_rkg

    else:
        raise ValueError(f"Unknown method: {method}. Choose 'supervised' / 'unsupervised' / 'rkg'")

    # Compute statistics
    original_steps_counts = [len(parse_steps_from_trace(t)) for t in traces]
    cleaned_steps_counts = [len(parse_steps_from_trace(t)) for t in cleaned_traces]

    stats = {
        "original_avg_steps": np.mean(original_steps_counts) if original_steps_counts else 0.0,
        "cleaned_avg_steps":  np.mean(cleaned_steps_counts)  if cleaned_steps_counts  else 0.0,
        "original_steps_std": np.std(original_steps_counts)  if original_steps_counts else 0.0,
        "cleaned_steps_std":  np.std(cleaned_steps_counts)   if cleaned_steps_counts  else 0.0,
        "total_removed_steps": sum(
            len(parse_steps_from_trace(t)) - len(parse_steps_from_trace(ct))
            for t, ct in zip(traces, cleaned_traces)
        ),
        "num_anomalous_steps":    len(anomalous_steps),
        "num_underthinking_traces": n_underthinking if method == "rkg" else 0,
        "method": method,
    }

    return {
        "sample_id": sample.get("sample_id"),
        "source_dataset": sample.get("source_dataset"),
        "source_index": sample.get("source_index"),
        "target_answer": sample.get("target_answer"),
        "original_traces": traces,
        "cleaned_traces": cleaned_traces,
        "anomalous_steps": anomalous_info,
        "stats": stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Detect and remove anomalous reasoning steps"
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
        default="cleaned_traces.json",
        help="Output JSON file path (relative paths resolve under the results root)",
    )
    parser.add_argument(
        "--min_tf",
        type=float,
        default=0.001,
        help="Minimum term frequency threshold (default 0.001)",
    )
    parser.add_argument(
        "--idf_scope",
        type=str,
        default="sample",
        choices=["sample", "global", "none"],
        help="IRF corpus: 'sample' scores IDF within each sample's own steps (default, "
             "current behaviour); 'global' scores it over every step of every sample; "
             "'none' disables the IRF factor entirely, leaving TF-IRF == TF",
    )
    parser.add_argument(
        "--idf_norm",
        type=str,
        default="raw",
        choices=["raw", "log_n"],
        help="IDF scale: 'raw' is log(N/df) (default); 'log_n' divides by log(N) so the "
             "score lands in [0,1] and min_tfidf means the same under either --idf_scope",
    )
    parser.add_argument(
        "--df_table",
        type=str,
        default=None,
        help="Path to a global DF table JSON (used with --idf_scope global). Loaded if it "
             "exists, otherwise built from the input and saved here for reuse across modules",
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
        "--similarity_threshold",
        type=float,
        default=0.3,
        help="Similarity threshold; steps below this are flagged as anomalous (default 0.3)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="supervised",
        choices=["supervised", "unsupervised", "rkg"],
        help="Detection method: supervised | unsupervised | rkg (RKG structural+edge-frequency filter, requires --rkg_file)",
    )
    parser.add_argument(
        "--rkg_file",
        type=str,
        default=None,
        help="RKG JSON file path (required for method=rkg; output from step3_2_build_rkg.py)",
    )
    parser.add_argument(
        "--min_similar_steps",
        type=int,
        default=2,
        help="Minimum number of similar steps required to be considered normal (unsupervised method only, default 2)",
    )
    parser.add_argument(
        "--use_grpo_optimization",
        action="store_true",
        default=True,
        help="Whether to use GRPO optimization (unsupervised method only, default True)",
    )
    parser.add_argument(
        "--no_grpo_optimization",
        action="store_false",
        dest="use_grpo_optimization",
        help="Disable GRPO optimization (use original method)",
    )
    parser.add_argument(
        "--z_score_threshold",
        type=float,
        default=-1.5,
        help="Z-score threshold; steps below this are flagged as anomalous (GRPO optimization only, default -1.5)",
    )
    parser.add_argument(
        "--consensus_threshold",
        type=float,
        default=0.3,
        help="Consensus core threshold; fraction of steps a term must appear in to be included in the consensus core (default 0.3)",
    )
    parser.add_argument(
        "--use_weighted_similarity",
        action="store_true",
        default=True,
        help="Whether to use weighted Jaccard similarity (accounts for TF-IDF weights, default True)",
    )
    parser.add_argument(
        "--no_weighted_similarity",
        action="store_false",
        dest="use_weighted_similarity",
        help="Disable weighted similarity (use plain Jaccard)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="logical",
        choices=["logical", "math"],
        help="Reasoning domain: 'logical' (logical reasoning, default) or 'math' (mathematical reasoning, enables SymPy equation verification)",
    )
    parser.add_argument(
        "--protect_last_step",
        action="store_true",
        default=False,
        help="Always preserve the last step (conclusion step) of each trace",
    )
    parser.add_argument(
        "--underthinking_threshold",
        type=float,
        default=0.3,
        help="Underthinking detection threshold: traces whose step count falls more than this fraction below consensus are flagged (default 0.3, only active for method=rkg)",
    )

    args = parser.parse_args()

    # Read input file
    input_path = _cfg.resolve_input(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Reading file: {input_path}")
    with input_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict) and "results" in raw_data:
        samples = raw_data["results"]
    else:
        samples = raw_data if isinstance(raw_data, list) else []

    # Load RKG file (for method=rkg)
    rkg_lookup: Dict[str, Dict] = {}
    if args.method == "rkg":
        if not args.rkg_file:
            raise ValueError("method=rkg requires --rkg_file argument")
        rkg_path = _cfg.resolve_input(args.rkg_file)
        if not rkg_path.exists():
            raise FileNotFoundError(f"RKG file not found: {rkg_path}")
        with rkg_path.open("r", encoding="utf-8") as f:
            rkg_raw = json.load(f)
        rkg_results = rkg_raw.get("results", rkg_raw) if isinstance(rkg_raw, dict) else rkg_raw
        rkg_lookup = {r["sample_id"]: r for r in rkg_results if "sample_id" in r}
        print(f"Loading RKG file: {rkg_path} ({len(rkg_lookup)} samples)")

    # Build (or reuse) the global IRF corpus once, before any per-sample work
    df_table = None
    if args.idf_scope == "global":
        table_path = resolve_df_table_path(args.df_table) if args.df_table else None
        if table_path is not None and table_path.exists():
            df_table = DocFreqTable.load(table_path)
            print(f"Loading global DF table: {table_path}")
        else:
            df_table = build_global_df_table(samples, domain=args.domain,
                                             normalize=(args.idf_norm == "log_n"))
            if table_path is not None:
                df_table.save(table_path)
                print(f"Saved global DF table: {table_path}")
        print(f"IRF scope: global | {df_table.n_docs} step documents, {len(df_table)} terms")
        check_df_table(df_table, args.domain, args.idf_norm == "log_n")
    elif args.idf_scope == "none":
        df_table = FlatDocFreqTable()
        print("IRF scope: none (IRF factor disabled, TF-IRF == TF)")
    else:
        print("IRF scope: sample (IDF computed within each sample's own steps)")
    print(f"IDF scale: {args.idf_norm}")

    print(f"Processing {len(samples)} samples | method: {args.method}")

    results = []
    for idx, sample in enumerate(samples):
        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx + 1}/{len(samples)}")

        sample_rkg = rkg_lookup.get(sample.get("sample_id", "")) if args.method == "rkg" else None

        result = process_sample(
            sample,
            min_tf=args.min_tf,
            min_idf=args.min_idf,
            min_tfidf=args.min_tfidf,
            similarity_threshold=args.similarity_threshold,
            method=args.method,
            min_similar_steps=args.min_similar_steps,
            use_grpo_optimization=args.use_grpo_optimization if args.method == "unsupervised" else False,
            z_score_threshold=args.z_score_threshold if args.method == "unsupervised" else -1.5,
            consensus_threshold=args.consensus_threshold,
            use_weighted_similarity=args.use_weighted_similarity if args.method == "unsupervised" else False,
            domain=args.domain,
            sample_rkg=sample_rkg,
            underthinking_threshold=args.underthinking_threshold,
            df_table=df_table,
            idf_norm=(args.idf_norm == "log_n"),
        )
        results.append(result)

    # Compute overall statistics
    total_samples = len(results)
    total_original_steps = sum(r["stats"]["original_avg_steps"] * len(r["original_traces"])
                               for r in results)
    total_cleaned_steps = sum(r["stats"]["cleaned_avg_steps"] * len(r["cleaned_traces"])
                              for r in results)
    total_removed_steps = sum(r["stats"]["total_removed_steps"] for r in results)

    overall_stats = {
        "total_samples": total_samples,
        "avg_original_steps": np.mean([r["stats"]["original_avg_steps"] for r in results]),
        "avg_cleaned_steps": np.mean([r["stats"]["cleaned_avg_steps"] for r in results]),
        "avg_original_std": np.mean([r["stats"]["original_steps_std"] for r in results]),
        "avg_cleaned_std": np.mean([r["stats"]["cleaned_steps_std"] for r in results]),
        "total_removed_steps": total_removed_steps,
        "total_anomalous_steps": sum(r["stats"]["num_anomalous_steps"] for r in results),
    }

    # Save results
    output_path = _cfg.resolve_output(args.output)

    output_data = {
        "metadata": {
            "input_file": str(input_path),
            "method": args.method,
            "min_tf": args.min_tf,
            "min_idf": args.min_idf,
            "similarity_threshold": args.similarity_threshold,
            "min_similar_steps": args.min_similar_steps if args.method == "unsupervised" else None,
            "use_grpo_optimization": args.use_grpo_optimization if args.method == "unsupervised" else False,
            "z_score_threshold": args.z_score_threshold if args.method == "unsupervised" else None,
            "consensus_threshold": args.consensus_threshold if args.method == "unsupervised" else None,
            "use_weighted_similarity": args.use_weighted_similarity if args.method == "unsupervised" else False,
            "overall_stats": overall_stats,
            "note": "supervised: same-step comparison, requires one-to-one step correspondence across traces; unsupervised: cross-step comparison, GRPO-style, no alignment required; GRPO optimization: relative comparison, dynamic threshold, weighted similarity",
        },
        "results": results,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nDone!")
    print(f"Overall statistics:")
    print(f"  - Total samples: {overall_stats['total_samples']}")
    print(f"  - Avg original steps: {overall_stats['avg_original_steps']:.2f} +/- {overall_stats['avg_original_std']:.2f}")
    print(f"  - Avg cleaned steps: {overall_stats['avg_cleaned_steps']:.2f} +/- {overall_stats['avg_cleaned_std']:.2f}")
    print(f"  - Total removed steps: {overall_stats['total_removed_steps']}")
    print(f"  - Total anomalous steps: {overall_stats['total_anomalous_steps']}")
    print(f"\nResults saved to: {output_path}")

    # Show detailed info for the first few samples
    print(f"\nProcessing results preview:")
    for i, result in enumerate(results[:3]):
        print(f"\n  Sample {i+1} ({result['sample_id']}):")
        print(f"    Original avg steps: {result['stats']['original_avg_steps']:.2f} +/- {result['stats']['original_steps_std']:.2f}")
        print(f"    Cleaned avg steps: {result['stats']['cleaned_avg_steps']:.2f} +/- {result['stats']['cleaned_steps_std']:.2f}")
        print(f"    Removed steps: {result['stats']['total_removed_steps']}")
        print(f"    Anomalous steps: {result['stats']['num_anomalous_steps']}")
        if result['anomalous_steps']:
            print(f"    Anomalous step examples:")
            for anomalous in result['anomalous_steps'][:2]:
                print(f"      Step {anomalous['step_number']} (Trace {anomalous['trace_idx']}): {anomalous['step_text'][:50]}...")


if __name__ == "__main__":
    main()
