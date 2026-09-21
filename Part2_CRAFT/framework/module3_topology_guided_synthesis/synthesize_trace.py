#!/usr/bin/env python3
"""
synthesize_trace.py  (Module III — Topology-guided Trace Synthesis)
---------------------------------------------------------------------------
Synthesize a single high-quality reasoning trace from the k cleaned traces.

Pipeline:
1. Extract terms from all traces per sample (by step position)
2. Aggregate term frequencies across traces (union + frequency weighting)
3. Use aggregated terms as reference anchors for each step
4. Autoregressive synthesis: generate one step at a time with reference-guided prompts

Usage:
    python synthesize_trace.py \
        --input cleaned_traces_rkg.json \
        --output synthesized_traces.json \
        --model gpt-4o \
        --synthesis_strategy rkg
"""

import argparse
import json
import asyncio
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp
import backoff
import numpy as np
from tqdm import tqdm

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

# Reuse functions from Module I's tfirf_terms and Module II's steps_filter
from framework.module1_generation_filtering.tfirf_terms import (
    tokenize_text,
    calculate_tf,
    calculate_idf,
    DocFreqTable,
    FlatDocFreqTable,
    resolve_df_table_path,
    check_df_table,
    COMMON_LOGICAL_WORDS,
    MATH_COMMON_WORDS,
)
from framework.module1_generation_filtering.steps_filter import (
    parse_steps_from_trace,
    build_global_df_table,
    STEP_PATTERN,
)

#########################
# Configuration — loaded from root config.py; change models there
#########################
import importlib.util as _ilu, pathlib as _pl

# normalise_math_answer comes from extract_label, the one copy of it.
def _normalise_math_pred(s: Optional[str]) -> Optional[str]:
    """The answer as the trace wrote it.

    This used to rewrite the answer before storing it, and the rewriting lost
    the answer: \\tfrac{1}{2}, \\frac{41}{2} and \\frac{5}{2},\\ 3,\\ \\sqrt{10} all
    came back None, and only a bare integer survived. Fractions, roots and
    multi-part answers are most of what a competition problem asks for, so the
    stored pred_label was empty for 28% of an Omni-MATH run and 22% of an
    OlympiadBench one — every one of them scored wrong although the trace had
    written the answer out in \\boxed{}.

    extract_label.py's own notes record the same function being dropped from the
    evaluation side for the same reason: it stripped every LaTeX command, so
    \\lceil n/2 \\rceil + 1 became n/2 + 1. Equivalence is answers_match's job, and
    it compares symbolically; nothing is gained by rewriting the answer first.
    """
    if s is None:
        return None
    s = s.strip()
    return s or None


def _normalise_math_pred_legacy(s: Optional[str]) -> Optional[str]:
    """The previous rewriting behaviour, kept for reference. Not called."""
    try:
        import sys as _sys
        _eval_dir = str(_pl.Path(__file__).resolve().parents[2] / "evaluation" / "label_prediction")
        if _eval_dir not in _sys.path:
            _sys.path.insert(0, _eval_dir)
        from extract_label import normalise_math_answer
        return normalise_math_answer(s)
    except ImportError:
        pass
    # Fallback: minimal inline normalization
    if s is None:
        return None
    s = s.strip().strip("$").strip()
    if not s:
        return None
    s = s.replace('\u2013', '-').replace('\u2014', '-').replace('\u2212', '-')
    s = re.sub(r'\^?\{?\\circ\}?|°|\\degree', '', s)
    s = re.sub(r'\\(?:text|mathrm|mbox)\{[^}]*\}', '', s)
    s = re.sub(r'\\[a-zA-Z]+\*?', '', s)
    s = re.sub(r'[{}]', '', s)
    s = s.strip().lstrip("$\\").strip(";")
    s = re.sub(r'\s+', '', s).strip().lower()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, OverflowError):
        pass
    return s if s else None
_cfg_path = _pl.Path(__file__).resolve().parents[2] / "config.py"
_spec = _ilu.spec_from_file_location("_root_config", _cfg_path)
_cfg  = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_cfg)

OPENAI_API_KEY        = _cfg.OPENAI_API_KEY
OPENAI_BASE_URL       = _cfg.OPENAI_BASE_URL
DEFAULT_MODEL         = _cfg.MODEL_SYNTHESIS   # Step 5: Reference-Guided Topological Synthesis
REQUEST_TIMEOUT       = getattr(_cfg, "REQUEST_TIMEOUT", 180)
RESPONSE_TOKENS       = 4096   # increased for o4-mini on competition math
REQUEST_TEMPERATURE   = 0.0
RESPONSE_TOKENS_MATH  = 8192   # per-step cap for math domain synthesis

def _build_auth_header(api_key: str, base_url: str) -> str:
    return f"Bearer {api_key}"

HEADERS = {
    "Authorization": _build_auth_header(OPENAI_API_KEY, OPENAI_BASE_URL),
    "Content-Type": "application/json",
}

CHAT_COMPLETIONS_URL = OPENAI_BASE_URL.rstrip("/") + "/chat/completions"


def extract_step_terms_with_tfidf(
    step_text: str,
    all_step_documents: List[List[str]],
    min_tf: float = 0.001,
    min_idf: float = 0.1,
    min_tfidf: float = 0.01,
    domain: str = "logical",
    df_table: Optional[DocFreqTable] = None,
) -> Tuple[List[str], Dict[str, float]]:
    """Extract important terms and their TF-IRF scores for a single step (domain-aware).

    df_table: when given, IDF is scored against that corpus (the global IRF setting) and
    all_step_documents is ignored; when None, IDF comes from all_step_documents, i.e. the
    within-sample setting.
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
    terms_list = [term for term, _ in step_terms]

    return terms_list, tfidf_scores


def collect_terms_by_step_position(
    sample_traces: List[Dict[str, Any]],
    min_tfidf: float = 0.01,
    use_percentage_alignment: bool = True,
    num_buckets: int = 10,
    domain: str = "logical",
    df_table: Optional[DocFreqTable] = None,
    idf_norm: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """
    Collect terms at each step position across all traces.

    Args:
        use_percentage_alignment: If True, normalize each trace's steps to percentage
            positions (0%~100% divided into num_buckets buckets), avoiding misalignment
            errors from absolute step numbers across traces of different lengths.
        num_buckets: Number of percentage buckets (default 10, i.e. 0, 10, 20, ..., 90)

    Returns:
        Dict[bucket_key, {...}]  bucket_key is the percentage bucket index (0~num_buckets-1)
        or (when use_percentage_alignment=False) the original step_number
    """
    # Step 1: Collect all step texts (for IDF computation)
    all_step_texts = []
    all_steps_by_position = defaultdict(list)

    for trace_idx, trace in enumerate(sample_traces):
        parsed_steps = parse_steps_from_trace(trace)
        total_steps_in_trace = len(parsed_steps)
        if total_steps_in_trace == 0:
            continue

        # Down-weight underthinking traces: contribution weight 0.3 (normal trace weight 1.0)
        trace_weight = 0.3 if trace.get("underthinking", False) else 1.0

        for local_idx, step_info in enumerate(parsed_steps):
            step_text = step_info.get("step_text", "")
            if not step_text:
                continue
            all_step_texts.append(step_text)

            if use_percentage_alignment and total_steps_in_trace > 1:
                # Percentage position → bucket index (0 ~ num_buckets-1)
                pct = local_idx / (total_steps_in_trace - 1)  # 0.0 ~ 1.0
                bucket = min(int(pct * num_buckets), num_buckets - 1)
            else:
                bucket = step_info["step_number"]

            all_steps_by_position[bucket].append({
                "trace_idx":   trace_idx,
                "step_text":   step_text,
                "step_number": step_info["step_number"],
                "weight":      trace_weight,   # underthinking trace weight 0.3
            })

    # Under the global setting df_table already carries the corpus; otherwise build this
    # sample's own, which is the within-sample setting.
    if df_table is None:
        df_table = DocFreqTable.from_documents(
            [tokenize_text(text, domain=domain) for text in all_step_texts],
            normalize=idf_norm,
        )

    # Step 2: Extract terms per bucket and compute frequency counts
    step_terms_summary = {}

    for step_position, steps_list in all_steps_by_position.items():
        all_terms_counter = Counter()
        term_tfidf_scores_dict = defaultdict(list)

        for step_info in steps_list:
            step_text = step_info["step_text"]
            weight    = step_info.get("weight", 1.0)
            terms, tfidf_scores = extract_step_terms_with_tfidf(
                step_text,
                [],
                min_tfidf=min_tfidf,
                domain=domain,
                df_table=df_table,
            )

            for term in terms:
                # Accumulate frequency with weight: underthinking trace contributes 0.3, normal trace contributes 1.0
                all_terms_counter[term] += weight
                if term in tfidf_scores:
                    term_tfidf_scores_dict[term].append(tfidf_scores[term] * weight)

        avg_tfidf_scores = {
            term: np.mean(scores)
            for term, scores in term_tfidf_scores_dict.items()
        }

        sorted_terms = sorted(
            all_terms_counter.items(),
            key=lambda x: (x[1], avg_tfidf_scores.get(x[0], 0.0)),
            reverse=True
        )

        step_terms_summary[step_position] = {
            "terms": [term for term, _ in sorted_terms],
            "term_frequencies": dict(all_terms_counter),
            "term_tfidf_scores": avg_tfidf_scores,
            "num_traces": len(steps_list),
        }

    return step_terms_summary


# A step's own "Step 7:" prefix, stripped before the trace is numbered.
_STEP_PREFIX = re.compile(r"^\s*\*{0,2}Step\s*\d+\*{0,2}\s*[:.\-]\s*", re.IGNORECASE)


def build_synthesis_prompt(
    problem_input: str,
    step_terms_summary: Dict[int, Dict[str, Any]],
    ground_truth: Optional[str] = None,
    current_step: Optional[int] = None,
    step_label: Optional[int] = None,
    previous_steps: Optional[List[str]] = None,
    domain: str = "logical",
    mv_label: Optional[str] = None,
    answer_is_prior: bool = False,
    atomic_steps: bool = False,
) -> str:
    """
    Build a prompt for generating a high-quality reasoning trace.

    Args:
        problem_input: Problem context
        step_terms_summary: Summary of terms at each step position
        ground_truth: Expected answer (optional)
        current_step: Which bucket of terms to draw on (if None, generate all at once)
        step_label: What to call this step in the prompt. The buckets are
            percentage positions over the K traces, so their keys skip a number
            whenever no trace put a step in that tenth -- and the step then went
            out labelled "Step 4" in a trace whose steps run 0,1,2,4. Numbering
            by how many have been written keeps the label contiguous, and the
            model's own citations with it. Defaults to current_step.
        previous_steps: List of previously generated steps (used for autoregressive generation)
        domain: "logical" or "math"
    """

    label = step_label if step_label is not None else current_step

    # The maths trace comes out five to seven times longer than the CoT it is
    # built from, and two instructions here are why: every equals sign is to be
    # written out on both sides with no algebra skipped, and every step is to
    # open by reviewing the ones before it. The second is a restatement asked
    # for in so many words -- 8% to 20% of the sentences in these traces repeat
    # an earlier one. --atomic_steps asks for the inference instead.
    if atomic_steps:
        _equations_rule = (
            "1. **Show the Work That Matters**: write the equation being solved "
            "and its result. Arithmetic that a reader can do in their head need "
            "not be spelled out."
        )
        _verify_block = (
            "**Before Generating, Verify**:\n"
            "- Does this step make exactly one inference?\n"
            "- Does it state something the steps before it have not?\n"
            "- If this is the final step, will you reach an explicit conclusion?\n"
        )
    else:
        _equations_rule = (
            "1. **Show Full Equations**: Write complete expressions on both sides "
            "of every equals sign (e.g., '2x + 4 = 10 \u2192 2x = 6 \u2192 x = 3'). "
            "Never skip algebraic steps."
        )
        _verify_block = (
            "**Before Generating, Verify**:\n"
            "- Have you reviewed all previous steps?\n"
            "- Are you using conclusions from previous steps as premises?\n"
            "- Is your reasoning logically consistent with previous steps?\n"
            "- If this is the final step, will you reach an explicit conclusion?\n"
        )
    sorted_steps = sorted(step_terms_summary.items())

    if domain == "math":
        role_desc = "You are an expert mathematician."
        task_desc = "Your task is to solve the math problem with detailed, step-by-step calculations based on the important terms extracted from multiple reasoning traces."
    else:
        role_desc = "You are an expert in logical reasoning."
        task_desc = "Your task is to generate a high-quality, step-by-step reasoning process based on the important terms extracted from multiple reasoning traces."

    prompt = f"""{role_desc} {task_desc}

**Problem Context**:
{problem_input}
"""

    # What sits in this slot is the majority vote over the k traces in every
    # blind setting — the pipeline never sees a gold answer — and heading it
    # "Expected Answer" told the model the vote was the answer. That is the
    # difference this path had against the RKG path, whose prior is phrased as
    # something to check, so the ablation's w/o RKG row was measuring the graph
    # against being handed the answer rather than against no graph. A real
    # ground truth, which only a non-blind run supplies, still reads as one.
    if ground_truth:
        if answer_is_prior:
            prompt += f"""
**Prior** (the majority of the independent traces reached this; they come from
one model and can agree on a mistake — treat it as a prior, not as the answer,
and follow your own derivation if it disagrees):
{ground_truth}
"""
        else:
            prompt += f"""
**Expected Answer**:
{ground_truth}
"""

    if previous_steps and len(previous_steps) > 0:
        prompt += f"""
**Previously Generated Steps**:
"""
        for i, step_text in enumerate(previous_steps, start=1):
            prompt += f"Step {i}: {step_text}\n"
        prompt += "\n"

    if current_step is not None:
        step_data = step_terms_summary.get(current_step)
        if step_data:
            terms = step_data["terms"]
            frequencies = step_data["term_frequencies"]
            num_traces = step_data["num_traces"]

            min_freq = max(1, int(num_traces * 0.3))
            high_freq_terms = [
                term for term in terms
                if frequencies.get(term, 0) >= min_freq
            ]

            prompt += f"""
**Current Step ({label}) - Important Terms** (extracted from {num_traces} traces):
  Key terms: {', '.join(high_freq_terms[:20])}
  Term frequencies: {dict((t, frequencies[t]) for t in high_freq_terms[:10])}
"""
    else:
        prompt += f"""
**Important Terms by Step Position** (extracted from multiple reasoning traces):

"""
        for step_pos, step_data in sorted_steps:
            terms = step_data["terms"]
            frequencies = step_data["term_frequencies"]
            num_traces = step_data["num_traces"]

            min_freq = max(1, int(num_traces * 0.3))
            high_freq_terms = [
                term for term in terms
                if frequencies.get(term, 0) >= min_freq
            ]

            if high_freq_terms:
                prompt += f"Step {step_pos} (appears in {num_traces} traces):\n"
                prompt += f"  Key terms: {', '.join(high_freq_terms[:20])}\n"
                prompt += f"  Term frequencies: {dict((t, frequencies[t]) for t in high_freq_terms[:10])}\n\n"

    # Add domain-specific quality guidelines to prevent common errors
    if domain == "math":
        prompt += f"""
**Critical Quality Guidelines** (MUST FOLLOW):

{_equations_rule}

2. **Cite Premises**: Each step must state which given value, equation, or previous result it uses.

3. **No Arithmetic Errors**: Double-check every calculation. If uncertain, re-derive from the previous step.

4. **No Hallucination**: Use ONLY values and equations present in the Problem Context or derived in previous steps.

5. **No Redundancy**: Each step must advance the solution. Do not restate already-established equations.

6. **Reach a Numeric Answer** (CRITICAL):
   - You MUST continue until you obtain a concrete numeric or symbolic answer.
   - The final step MUST include the answer in \\boxed{{<answer>}} notation.
   - Do NOT stop before boxing the answer.

**Instructions**:
"""
    else:
        prompt += """
**Critical Quality Guidelines** (MUST FOLLOW to avoid common errors):

1. **Prevent Logical Errors**:
   - Ensure each step's conclusion logically follows from its premises
   - Double-check mathematical calculations and logical inferences
   - Avoid contradictions or invalid logical leaps
   - If a step requires multiple premises, ensure all are explicitly stated

2. **Prevent Redundancy**:
   - Do NOT repeat the same information or reasoning in multiple steps
   - Each step should add NEW information or reasoning
   - Avoid restating what was already established in previous steps
   - Be concise: remove unnecessary filler words or repetitive phrases

3. **Prevent Missing Steps**:
   - Ensure smooth transitions between steps
   - If jumping from A to C, explicitly show the intermediate reasoning B
   - Do NOT skip logical connections that are needed for clarity
   - Each step should naturally lead to the next

4. **Prevent Hallucination**:
   - ONLY use facts and information provided in the Problem Context
   - Do NOT invent new facts or make unsupported assumptions
   - If referencing a fact, ensure it exists in the Problem Context
   - When making inferences, clearly state they are inferences, not facts

5. **Prevent Wrong Order**:
   - Use facts and intermediate conclusions only AFTER they have been established
   - Ensure prerequisites are established before they are used
   - Follow a logical sequence: simpler steps before complex ones

6. **Prevent Early Termination** (CRITICAL):
   - You MUST generate ALL steps until reaching a clear conclusion
   - Do NOT stop prematurely or leave the reasoning incomplete
   - The final step MUST explicitly state the conclusion (__PROVED__ or __DISPROVED__ — two choices only)
   - If you are not at the final step, continue building the reasoning chain
   - Verify: Have you reached an explicit conclusion about the hypothesis?

**Instructions**:
"""

    if current_step is not None:
        total_steps = len(sorted_steps)
        is_last_step = (current_step == sorted_steps[-1][0])

        prompt += f"""
**Current Task**: Generate Step {label} of {total_steps} total steps"""

        if is_last_step:
            prompt += f""" (THIS IS THE FINAL STEP - MUST REACH A CLEAR CONCLUSION)"""

        prompt += f"""

**CRITICAL REQUIREMENTS**:

1. **Logical Consistency Check**:
   - CAREFULLY review ALL previously generated steps above
   - Identify what conclusions/facts have been established in previous steps
   - Use ONLY the established conclusions from previous steps as premises
   - Ensure your reasoning logically follows from these established facts
   - If a previous step concluded "X", you can use "X" as a premise in this step
   - DO NOT contradict or ignore conclusions from previous steps

2. **Step Generation**:
   - Generate ONLY Step {label} based on the important terms provided
   - Build logically on the previously generated steps (use their conclusions as premises)
   - Use the key terms naturally in your reasoning
   - Follow all quality guidelines above
   - Output format: "Step {label}: [your reasoning here]"
   - Do NOT generate any other steps"""

        if is_last_step:
            if domain == "math":
                prompt += f"""

3. **Final Step Requirements** (CRITICAL - This is Step {label}, the LAST step):
   - You MUST arrive at a clear, numeric or symbolic answer
   - Express the final answer using \\boxed{{<answer>}} notation
   - Do NOT stop before boxing the answer
   - Example: "Therefore, x = 3, so the answer is \\boxed{{3}}"
   - This is REQUIRED - do not skip the boxed answer"""
            else:
                prompt += f"""

3. **Final Step Requirements** (CRITICAL - This is Step {label}, the LAST step):
   - You MUST reach a clear, explicit conclusion about the hypothesis
   - The conclusion should be one of: __PROVED__ or __DISPROVED__ (exactly two choices)
   - Do NOT stop before reaching this conclusion
   - Explicitly state your final answer in the format: "Therefore, [conclusion]"
   - Example: "Therefore, the hypothesis is __PROVED__" or "Therefore, the hypothesis is __DISPROVED__"
   - This is REQUIRED - do not skip this final conclusion"""
        else:
            prompt += f"""

3. **Continuation Requirements**:
   - This is Step {label} of {total_steps} - you are NOT done yet
   - Continue building the reasoning chain
   - Do NOT conclude or stop here
   - Prepare for the next step"""

        prompt += f"""

{_verify_block}
Please generate Step {label} now:"""
    else:
        if domain == "math":
            prompt += """
1. Solve the problem step-by-step, incorporating the important terms from each step position
2. Show complete equations at each step (both sides of the equals sign)
3. Cite which given value or previous result you use in each step
4. Follow ALL quality guidelines above
5. The final step MUST include the answer in \\boxed{<answer>} notation

**Output Format**:
Step 1: [arithmetic/algebraic manipulation with full equations]
Step 2: [next manipulation]
...
Step N: [final answer — must include \\boxed{<answer>}]

Please solve the problem now:"""
        else:
            prompt += """
1. Generate a step-by-step reasoning process that incorporates the important terms from each step position
2. Use the terms naturally in your reasoning, ensuring logical flow
3. Follow the step structure indicated by the term positions
4. Ensure each step builds logically on the previous steps
5. Follow ALL quality guidelines above
6. Use clear, precise language
7. End with a clear conclusion

**Output Format**:
Step 1: [reasoning using terms from Step 1]
Step 2: [reasoning using terms from Step 2]
...
Step N: [final conclusion]

Please generate the reasoning process now:"""

    return prompt


_REASONING_MODELS = ("o4-mini", "o4-mini-2025-04-16", "deepseek-r1")


def _apply_reasoning_budget(payload: dict, model: str) -> dict:
    """Reasoning models bill hidden reasoning against max_tokens, so a budget sized
    for the visible answer comes back empty. Raise the total and reserve a visible slice."""
    if model in _REASONING_MODELS:
        payload["max_tokens"] = max(payload.get("max_tokens") or 0, 16000)
        payload["max_output_tokens"] = min(payload["max_tokens"], 4096)
    return payload


# Every LLM request this module makes passes through one function, so counting
# there is the number of calls actually paid for — retries included — rather than
# the number a run was expected to need. Each stage writes it into its own
# metadata as api_calls, which is where the cost of a run is read from.
API_CALLS = {"count": 0}


@backoff.on_exception(
    backoff.expo,
    (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError),
    max_tries=7,
    factor=2
)
async def generate_reasoning_trace(
    session: aiohttp.ClientSession,
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: Optional[int] = None,
) -> str:
    """Call the model to generate a reasoning trace."""
    API_CALLS["count"] += 1
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": REQUEST_TEMPERATURE,
        "max_tokens": max_tokens or RESPONSE_TOKENS,
    }
    _apply_reasoning_budget(payload, model)
    
    async with session.post(
        CHAT_COMPLETIONS_URL,
        json=payload,
        headers=HEADERS,
        timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
    ) as resp:
        if resp.status != 200:
            detail = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {detail[:200]}")
        
        data = await resp.json()
        msg = data["choices"][0]["message"]
        # o4-mini / deepseek-r1 put reasoning in reasoning_content, not content
        if model in ("o4-mini", "o4-mini-2025-04-16", "deepseek-r1"):
            content = (msg.get("reasoning_content") or msg.get("content") or "").strip()
        else:
            content = (msg.get("content") or "").strip()
        return content


def extract_problem_input(sample: Dict[str, Any]) -> str:
    """Extract the problem context."""
    problem_block = sample.get("problem")
    if isinstance(problem_block, dict):
        parts = []
        primary = problem_block.get("input")
        if isinstance(primary, str) and primary.strip():
            parts.append(primary.strip())
        facts_block = problem_block.get("facts")
        if isinstance(facts_block, str) and facts_block.strip():
            parts.append(facts_block.strip())
        if isinstance(facts_block, list):
            joined = "\n".join(str(v).strip() for v in facts_block if str(v).strip())
            if joined.strip():
                parts.append(joined.strip())
        if parts:
            return "\n".join(parts)
    
    # Try other fields
    for key in ["problem_text", "input", "question", "premise", "context"]:
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            joined = "\n".join(str(v).strip() for v in value if str(v).strip())
            if joined.strip():
                return joined.strip()
    
    return ""


def extract_facts_from_traces(traces: List[Dict[str, Any]]) -> str:
    """Extract Facts information from traces' reasoning_text."""
    import re
    fact_pattern = re.compile(r'(fact\s*\d+)\s*[:\.]?\s*([^\.\n]+)', re.I)
    facts_dict = {}
    
    for trace in traces:
        reasoning_text = trace.get("reasoning_text", "") or trace.get("raw_response", "")
        if not reasoning_text:
            continue
        
        # Extract all Facts
        for match in fact_pattern.finditer(reasoning_text):
            fact_key = match.group(1).lower().replace(" ", "")
            fact_content = match.group(2).strip()
            if fact_key not in facts_dict:
                facts_dict[fact_key] = fact_content
    
    # Sort by Fact number
    sorted_facts = []
    fact_numbers = []
    for fact_key in facts_dict.keys():
        match = re.search(r'fact(\d+)', fact_key)
        if match:
            fact_numbers.append((int(match.group(1)), fact_key))
    
    fact_numbers.sort()
    for num, fact_key in fact_numbers:
        sorted_facts.append(f"{fact_key.capitalize()}: {facts_dict[fact_key]}")
    
    return "\n".join(sorted_facts) if sorted_facts else ""


    return "\n".join(sorted_facts) if sorted_facts else ""


#########################
# RKG-Guided Synthesis
#########################

def topological_sort_rkg(consensus_rkg: Dict[str, Any]) -> List[str]:
    """Kahn topological sort of consensus RKG nodes.

    If cycles exist (due to LLM misclassification), remove the lowest-confidence
    edge to break the cycle before sorting.

    Returns:
        Ordered list of node ids (Fact nodes first, Conclusion node last)
    """
    from collections import deque

    nodes = {n["id"]: n for n in consensus_rkg.get("nodes", [])}
    edges = consensus_rkg.get("edges", [])

    # Build adjacency list & in-degree table
    in_degree: Dict[str, int] = {nid: 0 for nid in nodes}
    adj: Dict[str, List[str]] = defaultdict(list)

    for e in edges:
        src, dst = e["src"], e["dst"]
        if src in nodes and dst in nodes:
            adj[src].append(dst)
            in_degree[dst] += 1

    # Prioritize Fact nodes (in-degree=0 and type=fact enqueued first)
    queue: deque = deque()
    for nid, node in nodes.items():
        if in_degree[nid] == 0:
            queue.append(nid)

    # Sort elements in queue by fact → step → conclusion priority
    def node_priority(nid: str) -> int:
        t = nodes[nid].get("type", "step")
        return {"fact": 0, "step": 1, "conclusion": 2}.get(t, 1)

    topo_order: List[str] = []
    while queue:
        # Each iteration: pick the element with the lowest priority (type first, step_number as tiebreaker)
        nxt = min(queue, key=node_priority)
        queue.remove(nxt)
        topo_order.append(nxt)
        for neighbor in adj[nxt]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If not all nodes were sorted (cycle detected), append remaining nodes
    remaining = [nid for nid in nodes if nid not in topo_order]
    topo_order.extend(sorted(remaining, key=node_priority))

    return topo_order


def build_synthesis_plan_from_rkg(
    consensus_rkg: Dict[str, Any],
    topo_order: List[str],
) -> List[Dict[str, Any]]:
    """Convert topological sort result into a step-by-step synthesis plan.

    Each plan entry contains:
    - node_id, node_type, step_position
    - direct_predecessors: list of direct predecessor node ids
    - predecessor_texts: {predecessor_id: text}
    - key_terms: keywords from the consensus node text
    - num_traces_supporting: number of traces supporting this node
    - edge_confidence: average confidence of incoming edges
    """
    nodes = {n["id"]: n for n in consensus_rkg.get("nodes", [])}
    edges = consensus_rkg.get("edges", [])
    edge_frequencies = consensus_rkg.get("edge_frequencies", {})
    # How many of the K traces derived this node. The graph is the only thing
    # that knows it: a linear rewrite of one trace cannot say whether a step was
    # reached by all five or by one. 77% of the nodes on an FLD run are
    # unanimous and 11% come from a single trace, and both used to be presented
    # to Module III the same way.
    node_frequencies = consensus_rkg.get("node_frequencies", {})

    # Build dst → src lookup
    predecessors: Dict[str, List[str]] = defaultdict(list)
    edge_confs: Dict[str, List[float]] = defaultdict(list)
    for e in edges:
        predecessors[e["dst"]].append(e["src"])
        edge_confs[e["dst"]].append(e.get("confidence", 0.7))

    plan = []
    # Numbered over the steps that get written, not over the topological order.
    # A fact node is given rather than derived, so it is skipped here -- but it
    # used to consume a position anyway, and the trace came out numbered
    # Step 0, Step 2, Step 4 with the later steps citing "from Step 1" for a
    # step that was never written. 579 of 4000 exported traces carry such a
    # citation, and the maths cells carry the most because their graphs hold
    # the most facts.
    written = 0
    for nid in topo_order:
        node = nodes.get(nid, {})
        node_type = node.get("type", "step")

        # Fact nodes do not need to be generated — skip them (they are given)
        if node_type == "fact":
            continue
        written += 1

        pred_ids = predecessors.get(nid, [])
        pred_texts = {
            pid: nodes[pid].get("text", "") for pid in pred_ids if pid in nodes
        }

        # Simple keyword extraction (high-frequency words after stopword removal)
        text = node.get("text", "")
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        freq = Counter(words)
        key_terms = [w for w, _ in freq.most_common(10)]

        avg_conf = float(np.mean(edge_confs[nid])) if edge_confs[nid] else 0.7

        plan.append({
            "node_id": nid,
            "node_type": node_type,
            "step_position": written,
            "direct_predecessors": pred_ids,
            "predecessor_texts": pred_texts,
            "key_terms": key_terms,
            "node_text_hint": text[:200] if text else "",
            "node_frequency": node_frequencies.get(nid),
            "edge_confidence": round(avg_conf, 3),
        })

    return plan


def build_rkg_synthesis_prompt(
    problem_input: str,
    plan_entry: Dict[str, Any],
    generated_nodes: Dict[str, str],
    domain: str = "logical",
    total_steps: int = 1,
    mv_label: Optional[str] = None,
    mv_answer: Optional[str] = None,
    mv_strength: Optional[float] = None,
    prior_mode: str = "verify",
    atomic_steps: bool = False,
    gt_label: Optional[str] = None,
    step_terms_summary: Optional[Dict[int, Dict]] = None,
    previous_steps: Optional[List[str]] = None,
    prev_context: str = "all",
) -> str:
    """Build generation prompt for a single RKG node.

    For math domain (P1-B): use consensus node_text as reference anchor.
    For logical domain: use RKG topology (predecessor dependencies) + TF-IRF
      frequency hints — avoids contamination from biased node_texts in samples
      where most k-traces predict the wrong label.
    """
    node_id    = plan_entry["node_id"]
    node_type  = plan_entry["node_type"]
    step_pos   = plan_entry["step_position"]
    pred_texts = plan_entry["predecessor_texts"]
    is_last    = (node_type == "conclusion")
    ref_hint   = plan_entry.get("node_text_hint", "")  # consensus reference text

    if domain == "math":
        role = "You are an expert mathematician solving a math problem step by step."
    else:
        role = "You are an expert logician building a rigorous reasoning chain."

    prompt = f"""{role}

**Problem**:
{problem_input}
**Your current task**: Generate {node_id} (Step {step_pos} of {total_steps})
"""

    # Math domain: show ALL previously generated steps for full context
    # (like step_by_step does), plus highlight direct prerequisites.
    # Logical domain: show only direct prerequisites to avoid label contamination.
    # Both domains see the steps already written, and the graph says which of
    # them this one is built on. Only the maths branch used to show them: the
    # logical branch showed the direct predecessors alone and told the model not
    # to reference anything else, which is the graph used to withhold context
    # rather than to order it. Measured against synthesis without any graph, on
    # the same 49 samples, that cost 4.1 points on FLD and 3.6 extra steps, while
    # maths — which saw everything — came out level. The topology still fixes the
    # order and the dependencies; it no longer hides what has been derived.
    # --prev_context direct narrows this to the graph's own dependencies, which
    # is what the logical branch used to do on its own and what the comment
    # above records the cost of. It is here so the choice can be measured
    # rather than assumed, not because the default is in doubt.
    if previous_steps and prev_context == "all":
        prompt += "\n**Previously Generated Steps**:\n"
        for prev_step in previous_steps:
            prompt += f"{prev_step}\n"
        if pred_texts:
            pred_ids_str = ", ".join(pred_texts.keys())
            prompt += f"\n(Direct prerequisites for this step: {pred_ids_str})\n"
    elif pred_texts:
        prompt += "\n**Direct prerequisites** (the steps this one builds on):\n"
        for pid, ptext in pred_texts.items():
            actual_text = generated_nodes.get(pid, ptext)
            prompt += f"  - {pid}: {actual_text[:400]}\n"
    else:
        prompt += "\n**Direct prerequisites**: (none — use only the problem statement above)\n"

    # P1-B: Reference hint — consensus node_text as quality anchor.
    # For intermediate steps (non-conclusion): biased ref hint helps coherence.
    # The conclusion step is handled independently via _generate_once() and never
    # reaches this function, so label bias in ref_hint only affects intermediate steps.
    if ref_hint and ref_hint.strip():
        freq = plan_entry.get("node_frequency")
        if freq is not None and freq <= 0.4:
            # Only a minority of the traces reached this point. Anchoring on it
            # as if it were settled is how a single trace's mistake survives
            # into the synthesized chain.
            prompt += f"""
**Reference step** (only {round(freq * 100)}% of the traces derived this — treat it as a
suggestion, not a settled result):
  \"{ref_hint[:300]}\"
  (Derive this step yourself from the prerequisites; keep it only if it follows.)
"""
        else:
            agreed = "" if freq is None else f" — {round(freq * 100)}% of the traces agree on it"
            prompt += f"""
**Reference step** (distilled from high-frequency traces{agreed} — use as a quality anchor):
  \"{ref_hint[:300]}\"
  (Preserve the logical structure and content; you may rephrase for clarity.)
"""
    # Additionally provide TF-IRF term hints for logical domain (supplementary guidance)
    if domain == "logical" and step_terms_summary and total_steps > 1:
        bucket = round((step_pos - 1) / max(total_steps - 1, 1) * 9)
        step_data = step_terms_summary.get(bucket)
        if step_data:
            high_freq = [
                t for t in step_data.get("terms", [])[:20]
                if step_data.get("term_frequencies", {}).get(t, 0)
                   >= max(1, int(step_data.get("num_traces", 1) * 0.3))
            ]
            if high_freq:
                prompt += (
                    f"\n**Key terms for this step** (frequency-weighted from "
                    f"{step_data.get('num_traces', '?')} traces): "
                    f"{', '.join(high_freq[:15])}\n"
                )

    prompt += "\n**Requirements**:\n"
    if domain == "math":
        prompt += (
            "- Show FULL equations: write complete expressions on both sides of every = sign "
            "(e.g., '2x + 4 = 10 → 2x = 6 → x = 3'). Never skip algebraic steps\n"
            "- Compute the exact numerical result — no symbolic placeholders\n"
            "- Cite which given value, equation, or previous result you are using\n"
            "- Double-check every calculation; if uncertain, re-derive from the previous step\n"
            + ("- Keep this step to ONE inference: one equation solved, one "
               "substitution made, one quantity computed. If the work needs "
               "several, this step does the first and says what remains\n"
               if atomic_steps else
               "- If this step involves multiple operations, break them into labeled sub-steps: (a), (b), (c)...\n"
               "- Write a DETAILED derivation — show ALL intermediate work, not just the final result of this step\n") +
            "- Only use values from the problem statement or the steps listed above\n"
        )
    else:
        prompt += (
            "- Build this step on the prerequisites named above\n"
            + ("- ONE inference only: apply exactly one rule to exactly one set "
               "of premises and state what follows. Do not chain two rules in "
               "this step, and do not restate what earlier steps established\n"
               if atomic_steps else
               "- Each logical inference must be explicit and atomic\n")
        )

    if is_last:
        if domain == "math":
            prompt += (
                "- This is the FINAL step — compute the exact final answer\n"
                "- Must end with \\boxed{<answer>} where <answer> is a concrete number or simplified expression\n"
            )
            if mv_answer and prior_mode == "follow":
                # The same knob as the logical branch. nano's single
                # re-derivation is weaker than its own vote on Omni-MATH too —
                # the vote scores 54.5% and re-deriving scores 45.7%, losing in
                # every agreement bucket including the unanimous one — so for
                # that configuration the consensus answer is the one to reach.
                prompt += (
                    f"- The majority of the independent reasoning traces reach "
                    f"{mv_answer}. Produce a derivation that arrives at it.\n"
                )
            elif mv_answer:
                prompt += (
                    f"- Note: majority of reasoning traces suggest the answer may be {mv_answer} "
                    f"— verify this against your computation before committing\n"
                )
        else:
            prompt += (
                "- This is the FINAL step — you MUST end with the EXACT string "
                "__PROVED__ or __DISPROVED__ as the last part of your response\n"
                "- Example final sentence: \"Therefore, the hypothesis is __PROVED__.\"\n"
                "- Do NOT omit this marker — it is mandatory\n"
            )
            if gt_label and domain == "logical":
                prompt += (
                    f"- The correct answer is {gt_label}. "
                    f"Generate reasoning that rigorously leads to this conclusion.\n"
                )
            elif mv_label and not gt_label and domain == "logical" and prior_mode == "follow":
                # The consensus is taken as settled. Worth having as the other
                # setting of the knob: where a model's single re-derivation is
                # weaker than its own vote — nano on FLD, whose traces score
                # 77-81% each and 82.7% pooled — inviting it to overrule the
                # vote loses in every agreement bucket.
                prompt += (
                    f"- The majority of independent reasoning traces reach {mv_label}. "
                    f"Generate reasoning that rigorously leads to this conclusion.\n"
                )
            elif mv_label and not gt_label and domain == "logical":
                # The vote is a prior, not the answer. Telling the model to
                # "generate reasoning that leads to" the majority label makes the
                # conclusion a copy of the vote, so synthesis can never revisit a
                # unanimous one — and on the depth-5 ProofWriter slice the five
                # traces agree and are still wrong on a third of the samples they
                # agree on. The maths branch above has always phrased its prior as
                # something to verify against the derivation; this is the same
                # phrasing, carrying how much of the weighted vote actually backs
                # the label so the model can tell a split vote from a unanimous one.
                _share = ("" if mv_strength is None
                          else f" ({round(100 * mv_strength)}% of the weighted vote)")
                prompt += (
                    f"- Prior: the independent reasoning traces mostly reach "
                    f"{mv_label}{_share}. This is a prior, not the answer — the "
                    f"traces come from one model and can agree on a mistake.\n"
                    f"- Decide from the chain derived above: if it supports "
                    f"{mv_label}, say so; if it supports the other label, state "
                    f"the other label instead.\n"
                )
    else:
        prompt += "- Do NOT conclude the entire problem here — more steps follow\n"

    prompt += f"\nGenerate ONLY {node_id}: [your reasoning here]"
    return prompt


# Module level: synthesize_trace_rkg and synthesize_trace_for_sample both read
# a boxed answer, and this used to be nested inside the first of them. The
# second's reference to it raised NameError on every maths sample it reached,
# which is the whole of the ablation's w/o RKG row — the step_by_step path
# never returned a trace on Omni-MATH or OlympiadBench.
def _norm_ans(text) -> str:
    """Loose answer key, only for deciding whether two answers are the same one.

    The scorer compares answers symbolically; this is the cheap check Module III
    makes while it is still writing, where the cost of being wrong is one extra
    call.
    """
    if not text:
        return ""
    return re.sub(r"[\s{}$\\,]+", "", str(text)).strip().lower().rstrip(".")


def _extract_boxed_content(text: str) -> list:
    """Extract \\boxed{...} contents handling nested braces."""
    results = []
    i = 0
    while i < len(text):
        idx = text.find('\\boxed{', i)
        if idx == -1:
            break
        start = idx + 7  # len('\\boxed{')
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


async def synthesize_trace_rkg(
    session: aiohttp.ClientSession,
    sample: Dict[str, Any],
    sample_rkg: Dict[str, Any],
    model: str = DEFAULT_MODEL,
    domain: str = "logical",
    anchor_conclusion: bool = False,
    no_mv: bool = False,
    prior_mode: str = "verify",
    atomic_steps: bool = False,
    prev_context: str = "all",
    df_table: Optional[DocFreqTable] = None,
    idf_norm: bool = False,
    min_tfidf: float = 0.01,
) -> Dict[str, Any]:
    """RKG-guided high-quality trace generation (blind synthesis, no access to ground_truth).

    Synthesis strategy:
      1. Extract majority-vote answer/label from k-traces as synthesis direction prior (no GT access)
      2. Generate nodes in RKG topological order; each step provides predecessor steps + consensus reference text
      3. Append majority-vote answer hint to the final step (math: _mv_answer; logical: _mv_label)
      4. If final step is missing \\boxed{} / __PROVED__, blind force-conclude to append it
    """
    _LABEL_RE_LOCAL = re.compile(r"(__PROVED__|__DISPROVED__)", re.IGNORECASE)
    _MATH_ANS_RE_LOCAL = re.compile(
        r"(?:the\s+answer\s+is|final\s+answer\s*[:\=]|answer\s*[:\=])\s*([^\n\.]+)",
        re.IGNORECASE,
    )

    sample_id      = sample.get("sample_id", "unknown")
    consensus_rkg  = sample_rkg.get("consensus_rkg") or sample_rkg.get("consensus_dag") or {}

    if not consensus_rkg.get("nodes"):
        return {"sample_id": sample_id, "error": "empty_consensus_rkg", "synthesized_trace": None}

    problem_input = (
        sample.get("problem_text")
        or sample.get("problem_input")
        or sample.get("input", "")
        or sample.get("problem", {}).get("input", "")
    )
    ground_truth = (
        sample.get("target_answer")
        or sample.get("ground_truth")
        or sample.get("problem", {}).get("ground_truth")
    )

    # Fallback: extract problem context from trace reasoning text when top-level fields are missing.
    # Same fallback used by synthesize_trace_for_sample() (step_by_step path).
    if not problem_input or not problem_input.strip():
        _traces_for_extract = (
            sample.get("cleaned_traces") or sample.get("traces") or
            sample.get("original_traces", [])
        )
        _extracted = extract_facts_from_traces(_traces_for_extract)
        if _extracted:
            problem_input = f"Facts:\n{_extracted}"

    # Compute majority-vote answer/label from k_traces as a synthesis prior
    # math:    majority-vote numeric answer (extracted from \boxed{} in k-traces)
    # logical: majority-vote __PROVED__ / __DISPROVED__ label
    # NOTE: uses only the model's own predictions — no ground_truth access
    _mv_label:  Optional[str] = None   # logical domain
    _mv_answer: Optional[str] = None   # math domain
    _mv_strength: Optional[float] = None  # share of the weighted vote behind _mv_label
    _all_traces = sample.get("cleaned_traces") or sample.get("traces", [])
    if no_mv:
        _all_traces = []  # skip MV computation entirely

    # The weight Module II gave each trace when it built the consensus. This
    # vote used to count every trace as one, which meant Module II's weighting
    # never reached the answer: whatever the graph decided, the label came from
    # a plain count of the same traces. It is why the synthesized label agreed
    # with an unweighted vote on 98-100% of samples, why weighting the vote by
    # proof depth gained 2.8 points and lost all but 1.2 of them by the end of
    # the pipeline, and why Module II could not show a contribution in the
    # ablation — its output was not on the path to the prediction.
    _tw = (sample_rkg.get("consensus_rkg") or {}).get("trace_weights") or {}
    def _weight_of(_t) -> float:
        if not _tw:
            return 1.0
        idx = _t.get("trace_idx")
        return float(_tw.get(str(idx), _tw.get(idx, 1.0)))
    if _all_traces:
        if domain == "math":
            _ans_counts: Dict[str, int] = {}
            for _t in _all_traces:
                _txt = (_t.get("reasoning_text") or "") + " " + (_t.get("raw_response") or "")
                _boxed = _extract_boxed_content(_txt)
                if _boxed:
                    _a = _boxed[-1].strip()
                    _ans_counts[_a] = _ans_counts.get(_a, 0) + _weight_of(_t)
                else:
                    # fallback: "the answer is X"
                    _m2 = re.findall(
                        r"(?:the\s+answer\s+is|final\s+answer\s*[:\=])\s*([^\n\.]+)",
                        _txt, re.IGNORECASE,
                    )
                    if _m2:
                        _a = _m2[-1].strip()
                        _ans_counts[_a] = _ans_counts.get(_a, 0) + _weight_of(_t)
            if _ans_counts:
                _mv_answer = max(_ans_counts, key=_ans_counts.get)
        else:
            _label_counts: Dict[str, int] = {}
            for _t in _all_traces:
                _lbl = _t.get("label")
                if _lbl and _lbl in ("__PROVED__", "__DISPROVED__"):
                    _label_counts[_lbl] = _label_counts.get(_lbl, 0) + _weight_of(_t)
                    continue
                _txt = (_t.get("reasoning_text") or "") + " " + (_t.get("raw_response") or "")
                _ms = re.findall(r"__(PROVED|DISPROVED)__", _txt, re.IGNORECASE)
                if _ms:
                    _lbl = f"__{_ms[-1].upper()}__"
                    _label_counts[_lbl] = _label_counts.get(_lbl, 0) + _weight_of(_t)
            if _label_counts:
                _mv_label = max(_label_counts, key=_label_counts.get)
                _total = sum(_label_counts.values())
                if _total > 0:
                    _mv_strength = _label_counts[_mv_label] / _total

    topo_order = topological_sort_rkg(consensus_rkg)
    plan       = build_synthesis_plan_from_rkg(consensus_rkg, topo_order)

    if not plan:
        return {"sample_id": sample_id, "error": "empty_synthesis_plan", "synthesized_trace": None}

    # For math domain: RKG consensus typically has fewer nodes than the actual
    # number of reasoning steps needed. Use k-traces' median step count as
    # total_steps so the prompt tells the model how many steps to expect.
    if domain == "math" and _all_traces:
        _trace_step_counts = []
        for _t in _all_traces:
            _steps = _t.get("reasoning_steps", [])
            if _steps:
                _trace_step_counts.append(len(_steps))
            else:
                _text = _t.get("reasoning_text", "")
                _trace_step_counts.append(max(1, _text.count("\nStep ")))
        _median_steps = sorted(_trace_step_counts)[len(_trace_step_counts) // 2] if _trace_step_counts else len(plan)
        total_steps = max(len(plan), _median_steps)
    else:
        total_steps = len(plan)

    # Compute TF-IRF step-term hints.
    # For logical domain: unbiased frequency counts as content hints.
    # For math domain: also compute — helps RKG synthesis fill intermediate steps.
    _step_terms_summary: Dict[int, Dict] = {}
    if _all_traces:
        # alpha, as the run was given it. This call had 0.01 written into it,
        # so --min_tfidf never reached the graph-guided path at all: the term
        # hints each step is shown were always cut at the default, whatever the
        # run asked for. It is the one Module III hyperparameter the paper
        # names, and the ablation that varies it was varying nothing.
        _step_terms_summary = collect_terms_by_step_position(
            _all_traces, min_tfidf=min_tfidf,
            use_percentage_alignment=True, domain=domain,
            df_table=df_table, idf_norm=idf_norm,
        )

    if domain == "math":
        final_markers = [r"\boxed", "the answer is", "final answer"]
    else:
        final_markers = ["__proved__", "__disproved__", "therefore", "conclusion"]

    # ── Domain config: define extractors and text templates once, unified for all downstream code ──
    if domain == "math":
        def _extract_answer(text: str) -> Optional[str]:
            """Extract answer from text: prefer \\boxed{}, fallback to 'the answer is X'."""
            m = _extract_boxed_content(text or "")
            if m:
                return m[-1].strip()
            m2 = _MATH_ANS_RE_LOCAL.findall(text or "")
            return m2[-1].strip() if m2 else None
        def _has_conclusion(text: str) -> bool:
            return '\\boxed{' in (text or "")
        _conclusion_question = "what is the final answer?"
        _conclusion_format   = "Output ONLY the answer in the format: \\boxed{<answer>}"
        _conclude_append     = lambda ans: f"\nFinal Answer: \\boxed{{{ans}}}"
        _missing_desc        = "missing boxed final answer"
        _required_action     = "include the final answer in \\boxed{<answer>} notation"
    else:  # logical
        def _extract_answer(text: str) -> Optional[str]:
            """Extract conclusion label from text: __PROVED__ / __DISPROVED__."""
            m = _LABEL_RE_LOCAL.findall(text or "")
            return m[-1].upper() if m else None
        def _has_conclusion(text: str) -> bool:
            return "__PROVED__" in (text or "") or "__DISPROVED__" in (text or "")
        _conclusion_question = "what is the conclusion?"
        _conclusion_format   = "Output ONLY: __PROVED__ or __DISPROVED__ (two choices only)"
        _conclude_append     = lambda ans: f"\nFinal conclusion: {ans}"
        _missing_desc        = "lacks a proper conclusion"
        _required_action     = (
            "include a clear final conclusion — EXACTLY __PROVED__ or __DISPROVED__"
        )

    async def _post_process_add_marker(text: str) -> str:
        """When the synthesized trace is missing a conclusion marker, rewrite the last step via LLM.

        math:    append \\boxed{answer}
        logical: append __PROVED__ / __DISPROVED__
        """
        if _has_conclusion(text):
            return text
        prompt = (
            f"Problem:\n{problem_input}\n\n"
            f"Reasoning chain (incomplete — {_missing_desc}):\n{text}\n\n"
            "Your task: Rewrite ONLY the last step of the reasoning chain above "
            f"to {_required_action}.\n"
            "Derive it from the reasoning — do NOT guess.\n"
            "Do NOT add new reasoning — only rewrite/append to the last step.\n"
            "Output the COMPLETE revised reasoning chain with the corrected last step."
        )
        resp = await generate_reasoning_trace(session, prompt, model)
        if resp and _has_conclusion(resp):
            return resp.strip()

        # The rewrite is asked to reproduce the whole chain, which gives the model room
        # to drop the marker again. Fall back to asking for the verdict alone: a one-token
        # answer the model has almost no way to get wrong, then append it ourselves.
        verdict = await generate_reasoning_trace(
            session,
            (f"Problem:\n{problem_input}\n\nReasoning chain:\n{text}\n\n"
             f"Based only on the reasoning above, {_conclusion_question}\n"
             f"{_conclusion_format}"),
            model, max_tokens=64,
        )
        ans = _extract_answer(verdict or "")
        if ans:
            return (text.rstrip() + _conclude_append(ans)).strip()
        return text  # Fallback: return original if post-processing failed

    async def _generate_once() -> Tuple[str, List[str]]:
        generated_nodes: Dict[str, str] = {}
        generated_steps: List[str]      = []

        # ── Math domain: hybrid RKG + step_by_step autoregressive ──────────
        # Uses RKG topo order for step count & reference hints, but generates
        # each step autoregressively (seeing full prior context) like
        # synthesize_trace_for_sample — this produces detailed arithmetic
        # chains that per-node independent generation cannot.
        if domain == "math":
            _math_max_tok = RESPONSE_TOKENS_MATH

            # Build RKG reference hint lookup: map bucket position → hint text
            _dag_hints: Dict[int, str] = {}
            for entry in plan:
                pos = entry.get("step_position", 1)
                hint = entry.get("node_text_hint", "")
                if hint:
                    # Map RKG step position to percentage bucket (0-9)
                    bucket = round((pos - 1) / max(total_steps - 1, 1) * 9)
                    _dag_hints[bucket] = hint

            # Iterate over step_terms_summary buckets (same as step_by_step)
            sorted_step_positions = sorted(_step_terms_summary.keys()) if _step_terms_summary else list(range(total_steps))

            for idx, step_pos in enumerate(sorted_step_positions):
                is_last = (idx == len(sorted_step_positions) - 1)

                prompt = build_synthesis_prompt(
                    problem_input,
                    _step_terms_summary,
                    ground_truth=_mv_answer,  # majority-vote answer (no GT leakage)
                    answer_is_prior=True,
                    current_step=step_pos,
                    step_label=idx + 1,
                    previous_steps=generated_steps,
                    domain=domain,
                    atomic_steps=atomic_steps,
                )
                # Note: RKG reference hints are NOT injected for math domain.
                # Math intermediate results are problem-specific; consensus node
                # texts from different traces can mislead the solver.
                # RKG value for math is in filtering (Steps 3.2/3.3), not generation.
                if is_last:
                    if _mv_answer:
                        prompt += (
                            f"\n- Note: majority of traces suggest the answer may be "
                            f"{_mv_answer} — verify against your computation\n"
                        )
                    prompt += (
                        "\n- This is the FINAL step — compute the exact final answer\n"
                        "- Must end with \\boxed{<answer>}\n"
                    )

                response = await generate_reasoning_trace(
                    session, prompt, model, max_tokens=_math_max_tok)
                step_content = parse_step_from_response(response, idx + 1)
                if step_content:
                    generated_steps.append(step_content)
                elif response and response.strip():
                    generated_steps.append(response.strip())
                else:
                    if is_last:
                        prev_text = "\n".join(generated_steps)
                        retry_prompt = (
                            f"Problem:\n{problem_input}\n\n"
                            f"Reasoning so far:\n{prev_text}\n\n"
                            "Continue and complete the solution. "
                            "Compute the exact final answer and express it as "
                            "\\boxed{<answer>}. Show your work."
                        )
                        retry_resp = await generate_reasoning_trace(
                            session, retry_prompt, model, max_tokens=_math_max_tok)
                        if retry_resp and retry_resp.strip():
                            generated_steps.append(retry_resp.strip())

            # The label the model was given is contiguous, so its own prefix
            # agrees with the position; strip it anyway and number here, so a
            # model that writes its own number cannot reintroduce a gap.
            text = "\n".join([
                f"Step {i+1}: {_STEP_PREFIX.sub('', s).strip()}"
                for i, s in enumerate(generated_steps)
            ])

            # Force-conclude: ensure \boxed{} present
            if not _has_conclusion(text):
                force_prompt = (
                    f"Problem:\n{problem_input}\n\n"
                    f"Reasoning:\n{text}\n\n"
                    "Based on the complete reasoning above, what is the final numerical answer?\n"
                    "Respond with ONLY: \\boxed{<answer>} — no other text."
                )
                force_resp = await generate_reasoning_trace(session, force_prompt, model, max_tokens=256)
                boxes = re.findall(r'\\boxed\{([^}]+)\}', force_resp or "")
                if boxes:
                    text += f"\nFinal Answer: \\boxed{{{boxes[-1]}}}"
                    generated_steps.append(f"Final Answer: \\boxed{{{boxes[-1]}}}")

            return text, generated_steps

        # ── Logical domain: per-node RKG synthesis (original path) ─────────
        for entry in plan:
            step_pos   = entry.get("step_position", len(generated_steps) + 1)
            is_last_e  = (step_pos == total_steps)

            # Logical domain conclusion nodes: use RKG consensus label directly
            if anchor_conclusion and domain == "logical":
                ref_hint = entry.get("node_text_hint", "").strip()
                _is_conclusion_node = (
                    entry.get("node_type") == "conclusion"
                    or ref_hint.lower().startswith("final conclusion")
                )
                if _is_conclusion_node and any(
                    lbl in ref_hint
                    for lbl in ["__PROVED__", "__DISPROVED__"]
                ):
                    nid = entry["node_id"]
                    generated_nodes[nid] = ref_hint
                    step_num = entry.get("step_position", len(generated_steps) + 1)
                    generated_steps.append(f"Step {step_num}: {ref_hint}")
                    continue

            prompt = build_rkg_synthesis_prompt(
                problem_input, entry, generated_nodes,
                domain=domain, total_steps=total_steps,
                mv_label=_mv_label, mv_answer=_mv_answer,
                mv_strength=_mv_strength, prior_mode=prior_mode,
                atomic_steps=atomic_steps,
                gt_label=None,
                step_terms_summary=_step_terms_summary,
                previous_steps=generated_steps,
                prev_context=prev_context,
            )
            response = await generate_reasoning_trace(session, prompt, model)
            nid      = entry["node_id"]
            content  = response.strip()
            prefix_m = re.match(
                rf'^\*{{0,2}}{re.escape(nid)}\*{{0,2}}\s*[:\.]?\s*\*{{0,2}}\s*',
                content, re.IGNORECASE)
            if not prefix_m:
                prefix_m = re.match(r'^\*{0,2}Step\s*\d+\*{0,2}\s*[:\.]?\s*\*{0,2}\s*',
                                    content, re.IGNORECASE)
            if prefix_m:
                content = content[prefix_m.end():].strip()
            if not content:
                content = response.strip()
            generated_nodes[nid] = content
            step_num = entry.get("step_position", len(generated_steps) + 1)
            generated_steps.append(f"Step {step_num}: {content}")

        text = "\n".join(generated_steps)
        if not _has_conclusion(text):
            force_prompt = (
                f"Problem:\n{problem_input}\n\n"
                f"Reasoning so far:\n{text}\n\n"
                f"Based on the complete reasoning above, {_conclusion_question}\n"
                f"{_conclusion_format}\n"
                "Output ONLY the answer/conclusion, with no other text."
            )
            conclusion_resp = await generate_reasoning_trace(session, force_prompt, model)
            conclusion_ans  = _extract_answer(conclusion_resp)
            if conclusion_ans:
                suffix = _conclude_append(conclusion_ans)
                text  += suffix
                generated_steps.append(suffix.strip())
        return text, generated_steps

    try:
        synthesis_method = "rkg"

        synthesized_text, generated_steps = await _generate_once()

        pred_label = _extract_answer(synthesized_text)

        # prior_mode="follow" on a mathematical problem: the consensus decides
        # the answer, so a chain that lands somewhere else gets one chance to
        # re-derive its ending. Saying so in the final node's requirements is
        # not enough — nano followed the consensus on 62% of Omni-MATH there,
        # against 65% when it was merely asked to check it, because the answer
        # comes from whatever \boxed{} the last node computed and the model
        # recomputes. This re-asks for the closing step with the target named,
        # so the trace still argues its way to the answer rather than having one
        # pasted onto it; if the model will not get there, its own answer stands.
        if (prior_mode == "follow" and domain == "math" and _mv_answer
                and synthesized_text
                and _norm_ans(pred_label) != _norm_ans(_mv_answer)):
            retarget = (
                f"Problem:\n{problem_input}\n\n"
                f"Derivation so far:\n{synthesized_text}\n\n"
                f"Independent reasoning traces agree the answer is {_mv_answer}, "
                f"and the derivation above ends at {pred_label or 'no answer'}. "
                f"Find where it goes wrong and write the corrected closing step, "
                f"ending with \\boxed{{{_mv_answer}}}.\n"
                "Output ONLY the corrected closing step."
            )
            fixed = await generate_reasoning_trace(session, retarget, model)
            if fixed and _norm_ans(_extract_answer(fixed)) == _norm_ans(_mv_answer):
                synthesized_text = synthesized_text.rstrip() + "\n" + fixed.strip()
                generated_steps.append(fixed.strip())
                pred_label = _extract_answer(synthesized_text)
            else:
                # The re-ask did not get there, and under "follow" the consensus
                # is what decides: Modules I and II pick the answer, Module III
                # writes the reasoning. Letting the chain keep its own ending
                # instead costs this configuration real accuracy — on nano's two
                # maths sets the consensus is right on 61.0 and 53.6 of a hundred
                # and the chains that wander off it land on 59.8 and 51.4 — and
                # the wandering is not rare enough to ignore at 11% and 13% of
                # samples. Of the ones this settles, 15 against 7 and 21 against
                # 6 go to the consensus's answer over the chain's.
                closing = _conclude_append(_mv_answer)
                synthesized_text = synthesized_text.rstrip() + closing
                generated_steps.append(closing.strip())
                pred_label = _mv_answer

        if pred_label and not _has_conclusion(synthesized_text):
            synthesized_text += _conclude_append(pred_label)

        # Post-processing: if still missing conclusion marker, LLM rewrites last step
        if synthesized_text and not _has_conclusion(synthesized_text):
            synthesized_text = await _post_process_add_marker(synthesized_text)
            pred_label = _extract_answer(synthesized_text) or pred_label

        best_text = synthesized_text

        # ── Math: expand compressed RKG skeleton into full arithmetic chain ───
        # Only needed when math domain uses per-node RKG synthesis (not hybrid).
        # The hybrid path already produces detailed autoregressive traces.
        if False and domain == "math" and best_text:
            expand_prompt = (
                f"Problem:\n{problem_input}\n\n"
                f"High-level reasoning outline:\n{best_text}\n\n"
                "Task: Using the outline above as a guide, write a COMPLETE and DETAILED "
                "step-by-step solution to the problem from scratch.\n"
                "- The outline shows the key ideas — you must fill in ALL intermediate "
                "algebraic manipulations, substitutions, and arithmetic\n"
                "- Show EVERY equation transformation and compute EVERY numerical result explicitly\n"
                "- Break complex steps into multiple sub-steps — do NOT skip any calculation\n"
                "- Each step must produce a concrete number or simplified expression\n"
                "- Label each step: Step 1:, Step 2:, Step 3:, ... (use as many steps as needed)\n"
                "- End with \\boxed{<final answer>}\n"
                "Output ONLY the complete solution."
            )
            expanded = await generate_reasoning_trace(
                session, expand_prompt, model, max_tokens=RESPONSE_TOKENS_MATH)
            if expanded and _has_conclusion(expanded):
                best_text  = expanded.strip()
                pred_label = _extract_answer(best_text) or pred_label

        if domain == "math" and pred_label:
            pred_label = _normalise_math_pred(pred_label)
        return {
            "sample_id":         sample_id,
            "source_dataset":    sample.get("source_dataset", ""),
            "problem_input":     problem_input,
            "ground_truth":      ground_truth,
            "domain":            domain,
            "synthesis_method":  synthesis_method,
            "topo_order":        topo_order,
            "synthesis_plan":    plan,
            "synthesized_trace": best_text,
            "num_rkg_nodes":     len(plan),
            "pred_label":        pred_label,
        }

    except Exception as e:
        return {"sample_id": sample_id, "error": str(e), "synthesized_trace": None}


def parse_step_from_response(response: str, step_number: int) -> Optional[str]:
    """Parse the content of the specified step from the response."""
    # Try to match "Step X: ..." format
    pattern = re.compile(rf"Step\s+{step_number}\s*[:\.]\s*(.+?)(?=\n\s*Step\s+\d+\s*[:\.]|\Z)", re.DOTALL | re.IGNORECASE)
    match = pattern.search(response)
    if match:
        return match.group(1).strip()
    
    # If only one step was returned (no Step marker), try to extract the entire content
    if step_number == 1 and not re.search(r"Step\s+\d+\s*[:\.]", response, re.IGNORECASE):
        return response.strip()
    
    return None


async def synthesize_trace_for_sample(
    session: aiohttp.ClientSession,
    sample: Dict[str, Any],
    min_tfidf: float = 0.01,
    model: str = DEFAULT_MODEL,
    step_by_step: bool = True,
    domain: str = "logical",
    df_table: Optional[DocFreqTable] = None,
    idf_norm: bool = False,
) -> Dict[str, Any]:
    """
    Generate a high-quality reasoning trace for a single sample.

    Args:
        step_by_step: If True, generate each step one at a time; if False, generate all steps at once
        domain: "logical" or "math"
    """
    sample_id = sample.get("sample_id", "unknown")
    traces = sample.get("cleaned_traces", []) or sample.get("traces", [])

    if not traces:
        return {
            "sample_id": sample_id,
            "error": "no_traces",
            "synthesized_trace": None,
        }

    problem_input = extract_problem_input(sample)

    if not problem_input or not problem_input.strip():
        facts_from_traces = extract_facts_from_traces(traces)
        if facts_from_traces:
            problem_input = f"Facts:\n{facts_from_traces}"

    _gt_raw = sample.get("problem", {}).get("ground_truth") or sample.get("target_answer")

    # Use majority-vote answer from k-traces instead of GT to avoid leakage.
    # For math: extract \boxed{} answers and take the most common one.
    # For logical: extract __PROVED__/__DISPROVED__ labels.
    _mv_answer_sbs: Optional[str] = None
    if domain == "math":
        _ans_counts_sbs: Dict[str, int] = {}
        for _t in traces:
            _txt = (_t.get("reasoning_text") or "") + " " + (_t.get("raw_response") or "")
            # extract \boxed{}
            _i = 0
            while _i < len(_txt):
                _idx = _txt.find('\\boxed{', _i)
                if _idx == -1: break
                _start = _idx + 7
                _depth, _j = 1, _start
                while _j < len(_txt) and _depth > 0:
                    if _txt[_j] == '{': _depth += 1
                    elif _txt[_j] == '}': _depth -= 1
                    _j += 1
                if _depth == 0:
                    _a = _txt[_start:_j-1].strip()
                    _ans_counts_sbs[_a] = _ans_counts_sbs.get(_a, 0) + 1
                _i = _j
            if not _ans_counts_sbs:
                _m2 = re.findall(r"(?:the\s+answer\s+is|final\s+answer\s*[:\=])\s*([^\n\.]+)", _txt, re.IGNORECASE)
                if _m2:
                    _a = _m2[-1].strip()
                    _ans_counts_sbs[_a] = _ans_counts_sbs.get(_a, 0) + 1
        if _ans_counts_sbs:
            _mv_answer_sbs = max(_ans_counts_sbs, key=_ans_counts_sbs.get)
    else:
        _label_counts_sbs: Dict[str, int] = {}
        for _t in traces:
            _lbl = _t.get("label")
            if _lbl and _lbl in ("__PROVED__", "__DISPROVED__"):
                _label_counts_sbs[_lbl] = _label_counts_sbs.get(_lbl, 0) + 1
                continue
            _txt = (_t.get("reasoning_text") or "") + " " + (_t.get("raw_response") or "")
            _ms = re.findall(r"__(PROVED|DISPROVED)__", _txt, re.IGNORECASE)
            if _ms:
                _lbl = f"__{_ms[-1].upper()}__"
                _label_counts_sbs[_lbl] = _label_counts_sbs.get(_lbl, 0) + 1
        if _label_counts_sbs:
            _mv_answer_sbs = max(_label_counts_sbs, key=_label_counts_sbs.get)

    # The prompt is given the majority vote, never the gold answer — this path is
    # as blind as the RKG one. The record, though, has to carry the gold answer,
    # because that is the field the scorer reads as ground truth. Writing the
    # vote there made "w/o RKG" score agreement with the vote instead of
    # accuracy: on a 500-sample FLD run the row read 0.972 where it is 0.875,
    # and the whole ablation column was built on that.
    ground_truth = _mv_answer_sbs  # prompt hint only — majority vote, not GT
    answer_is_prior = True         # …so it is presented as a prior, not an answer

    # Collect terms by step position (using percentage alignment)
    step_terms_summary = collect_terms_by_step_position(
        traces,
        min_tfidf=min_tfidf,
        use_percentage_alignment=True,
        domain=domain,
        df_table=df_table,
        idf_norm=idf_norm,
    )

    if not step_terms_summary:
        return {
            "sample_id": sample_id,
            "error": "no_terms_extracted",
            "synthesized_trace": None,
        }

    sorted_step_positions = sorted(step_terms_summary.keys())

    # math domain: final step must contain \boxed or a numeric answer
    if domain == "math":
        final_markers = [r"\boxed", "the answer is", "final answer", "="]
    else:
        final_markers = ["__proved__", "__disproved__", "therefore", "conclusion"]

    try:
        if step_by_step:
            generated_steps = []

            _math_max_tok = RESPONSE_TOKENS_MATH if domain == "math" else None

            for idx, step_pos in enumerate(sorted_step_positions):
                is_last_step = (idx == len(sorted_step_positions) - 1)

                prompt = build_synthesis_prompt(
                    problem_input,
                    step_terms_summary,
                    ground_truth,
                    current_step=step_pos,
                    previous_steps=generated_steps,
                    domain=domain,
                    answer_is_prior=answer_is_prior,
                )

                response = await generate_reasoning_trace(
                    session, prompt, model, max_tokens=_math_max_tok)

                step_content = parse_step_from_response(response, step_pos)
                if step_content:
                    generated_steps.append(step_content)
                else:
                    cleaned_response = (response or "").strip()
                    if cleaned_response:
                        generated_steps.append(cleaned_response)
                    else:
                        # Final step empty → retry with a direct solve prompt
                        if is_last_step and domain == "math":
                            prev_text = "\n".join(generated_steps)
                            retry_prompt = (
                                f"Problem:\n{problem_input}\n\n"
                                f"Reasoning so far:\n{prev_text}\n\n"
                                "Continue and complete the solution. "
                                "Compute the exact final answer and express it as "
                                "\\boxed{<answer>}. Show your work."
                            )
                            retry_resp = await generate_reasoning_trace(
                                session, retry_prompt, model, max_tokens=_math_max_tok)
                            if retry_resp and retry_resp.strip():
                                generated_steps.append(retry_resp.strip())
                            else:
                                generated_steps.append(
                                    f"[Step {step_pos} content not generated]")
                        else:
                            generated_steps.append(
                                f"[Step {step_pos} content not generated]")

            synthesized_text = "\n".join([
                f"Step {i+1}: {step_content}"
                for i, step_content in enumerate(generated_steps)
            ])

            # ── Force-conclude for math: ensure \boxed{} is present ──────────
            if domain == "math" and r'\boxed' not in synthesized_text:
                force_prompt = (
                    f"Problem:\n{problem_input}\n\n"
                    f"Reasoning:\n{synthesized_text}\n\n"
                    "Based on the complete reasoning above, what is the final numerical answer?\n"
                    "Respond with ONLY: \\boxed{<answer>} — no other text."
                )
                force_resp = await generate_reasoning_trace(
                    session, force_prompt, model, max_tokens=256)
                boxes = re.findall(r'\\boxed\{([^}]+)\}', force_resp or "")
                if boxes:
                    synthesized_text += f"\nFinal Answer: \\boxed{{{boxes[-1]}}}"
                else:
                    # Last resort: fresh solve attempt
                    fresh_prompt = (
                        f"Solve the following math problem. "
                        f"Show your work step by step and end with \\boxed{{answer}}.\n\n"
                        f"{problem_input}"
                    )
                    fresh_resp = await generate_reasoning_trace(
                        session, fresh_prompt, model, max_tokens=_math_max_tok)
                    if fresh_resp and r'\boxed' in fresh_resp:
                        synthesized_text = fresh_resp.strip()
        else:
            prompt = build_synthesis_prompt(
                problem_input, step_terms_summary, ground_truth, domain=domain,
                answer_is_prior=answer_is_prior,
            )
            synthesized_text = await generate_reasoning_trace(session, prompt, model)

        # Extract pred_label from synthesized text
        if domain == "math":
            _boxes = _extract_boxed_content(synthesized_text or "")
            _raw_pred = _boxes[-1].strip() if _boxes else None
            if not _raw_pred:
                _m2 = re.findall(r"(?:the\s+answer\s+is|final\s+answer\s*[:\=])\s*([^\n\.]+)", synthesized_text or "", re.IGNORECASE)
                _raw_pred = _m2[-1].strip() if _m2 else None
            sbs_pred_label = _normalise_math_pred(_raw_pred)
        else:
            _ms = re.findall(r"__(PROVED|DISPROVED)__", synthesized_text or "", re.IGNORECASE)
            sbs_pred_label = f"__{_ms[-1].upper()}__" if _ms else None

        return {
            "sample_id": sample_id,
            "problem_input": problem_input,
            "ground_truth": _gt_raw,
            "mv_hint": ground_truth,
            "domain": domain,
            "pred_label": sbs_pred_label,
            "step_terms_summary": {
                step_pos: {
                    "terms": step_data["terms"][:30],
                    "term_frequencies": dict(list(step_data["term_frequencies"].items())[:20]),
                    "num_traces": step_data["num_traces"],
                }
                for step_pos, step_data in step_terms_summary.items()
            },
            "synthesized_trace": synthesized_text,
            "num_original_traces": len(traces),
            "num_step_positions": len(step_terms_summary),
            "generation_method": "step_by_step" if step_by_step else "all_at_once",
        }
    except Exception as e:
        return {
            "sample_id": sample_id,
            "error": str(e),
            "synthesized_trace": None,
        }


async def synthesize_traces_for_dataset(
    input_file: Path,
    output_file: Path,
    min_tfidf: float = 0.01,
    model: str = DEFAULT_MODEL,
    concurrency: int = 20,
    max_samples: Optional[int] = None,
    original_file: Optional[Path] = None,
    step_by_step: bool = True,
    domain: str = "logical",
    synthesis_strategy: str = "step_by_step",
    rkg_file: Optional[Path] = None,
    anchor_conclusion: bool = False,
    no_mv: bool = False,
    prior_mode: str = "verify",
    atomic_steps: bool = False,
    prev_context: str = "all",
    idf_scope: str = "sample",
    idf_norm: str = "raw",
    df_table_path: Optional[Path] = None,
):
    """Generate high-quality reasoning traces for the dataset (supports step_by_step and rkg strategies).

    idf_scope selects the IRF corpus: "sample" scores IDF within each sample's own steps
    (default, current behaviour); "global" scores it over every step of every sample.
    """
    print(f"Reading file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle data format
    if isinstance(data, dict):
        if 'results' in data:
            samples = data['results']
        else:
            samples = [data]
    else:
        samples = data if isinstance(data, list) else []

    # Load problem information from original file (if provided)
    original_problem_info = {}
    if original_file and original_file.exists():
        print(f"Loading original file for problem information: {original_file}")
        try:
            with open(original_file, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
            
            # Handle data format
            if isinstance(original_data, dict):
                if 'results' in original_data:
                    original_samples = original_data['results']
                else:
                    original_samples = [original_data]
            else:
                original_samples = original_data if isinstance(original_data, list) else []
            
            for orig_sample in original_samples:
                sample_id = orig_sample.get('sample_id', 'unknown')
                problem_block = orig_sample.get('problem', {})
                parts = []

                if isinstance(problem_block, dict) and problem_block:
                    # nested problem dict
                    problem_input = problem_block.get('input', '')
                    facts = problem_block.get('facts', [])
                    if isinstance(facts, str):
                        facts = [facts] if facts.strip() else []
                    if problem_input and problem_input.strip():
                        parts.append(problem_input.strip())
                    if facts:
                        facts_text = '\n'.join(str(f).strip() for f in facts if str(f).strip())
                        if facts_text:
                            parts.append(facts_text)
                    gt = problem_block.get('ground_truth') or orig_sample.get('target_answer')
                elif orig_sample.get('problem_text'):
                    # flat k_traces format: problem_text + target_answer at top level
                    parts.append(orig_sample.get('problem_text', '').strip())
                    gt = orig_sample.get('target_answer') or orig_sample.get('ground_truth')
                else:
                    gt = orig_sample.get('target_answer') or orig_sample.get('ground_truth')

                if parts:
                    original_problem_info[sample_id] = {
                        'problem_text': '\n'.join(parts),
                        'ground_truth': gt,
                    }
            
            print(f"Loaded problem information for {len(original_problem_info)} samples")
        except Exception as e:
            print(f"Warning: failed to load original file: {e}")
    
    # Build (or reuse) the global IRF corpus once. Built from the full file, before the
    # --max_samples cut, so a truncated debug run scores terms exactly like a full run and
    # so Module II and Module III agree on the same input.
    df_table = None
    if idf_scope == "global":
        if df_table_path is not None and Path(df_table_path).exists():
            df_table = DocFreqTable.load(df_table_path)
            print(f"Loading global DF table: {df_table_path}")
        else:
            df_table = build_global_df_table(samples, domain=domain,
                                             normalize=(idf_norm == "log_n"))
            if df_table_path is not None:
                df_table.save(Path(df_table_path))
                print(f"Saved global DF table: {df_table_path}")
        print(f"IRF scope: global | {df_table.n_docs} step documents, {len(df_table)} terms")
        check_df_table(df_table, domain, idf_norm == "log_n")
    elif idf_scope == "none":
        df_table = FlatDocFreqTable()
        print("IRF scope: none (IRF factor disabled, TF-IRF == TF)")
    else:
        print("IRF scope: sample (IDF computed within each sample's own steps)")
    print(f"IDF scale: {idf_norm}")

    if max_samples:
        samples = samples[:max_samples]

    # Load RKG file (required for strategy=rkg)
    rkg_lookup: Dict[str, Dict] = {}
    if synthesis_strategy == "rkg":
        if rkg_file is None or not rkg_file.exists():
            raise FileNotFoundError(
                f"synthesis_strategy=rkg requires a valid --rkg_file path (current: {rkg_file}). "
                "Build one with module2_consensus_rkg_construction/build_rkg.py, or pass "
                "--synthesis_strategy step_by_step to synthesize without the graph — "
                "which is the ablation's 'w/o RKG' setting, not CRAFT."
            )
        with open(rkg_file, 'r', encoding='utf-8') as f:
            rkg_raw = json.load(f)
        rkg_results = rkg_raw.get("results", rkg_raw) if isinstance(rkg_raw, dict) else rkg_raw
        rkg_lookup = {r["sample_id"]: r for r in rkg_results if "sample_id" in r}
        print(f"Loading RKG file: {rkg_file} ({len(rkg_lookup)} samples)")

    print(f"Processing {len(samples)} samples | strategy: {synthesis_strategy} | model: {model}")

    connector = aiohttp.TCPConnector(limit=concurrency)
    results = []

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for sample in samples:
            sample_id = sample.get('sample_id', 'unknown')

            # Fill in problem information
            if sample_id in original_problem_info:
                if 'problem' not in sample or not sample.get('problem'):
                    sample = sample.copy()
                    sample['problem'] = {
                        'input': original_problem_info[sample_id]['problem_text'],
                        'ground_truth': original_problem_info[sample_id].get('ground_truth'),
                    }

            # Route to RKG-guided or traditional synthesis
            if synthesis_strategy == "rkg":
                sample_rkg = rkg_lookup.get(sample_id, {})
                tasks.append(synthesize_trace_rkg(session, sample, sample_rkg, model=model, domain=domain, anchor_conclusion=anchor_conclusion, no_mv=no_mv, prior_mode=prior_mode, atomic_steps=atomic_steps, prev_context=prev_context, df_table=df_table, idf_norm=(idf_norm == "log_n"), min_tfidf=min_tfidf))
            else:
                tasks.append(synthesize_trace_for_sample(
                    session, sample, min_tfidf, model,
                    step_by_step=(synthesis_strategy != "all_at_once"),
                    domain=domain,
                    df_table=df_table,
                    idf_norm=(idf_norm == "log_n"),
                ))
        
        pbar = tqdm(total=len(tasks), desc="Generation progress", unit="sample")
        
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            pbar.update(1)
        
        pbar.close()
    
    # Save results
    output_data = {
        "metadata": {
            "input_file": str(input_file),
            "model": model,
            "min_tfidf": min_tfidf,
            "total_samples": len(samples),
            "successful_samples": sum(1 for r in results if r.get("synthesized_trace")),
            "api_calls": API_CALLS["count"],
        },
        "results": results,
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nDone!")
    print(f"Statistics:")
    print(f"  - Total samples: {len(samples)}")
    print(f"  - Successfully generated: {sum(1 for r in results if r.get('synthesized_trace'))}")
    print(f"  - Failed: {sum(1 for r in results if 'error' in r)}")
    print(f"\nResults saved to: {output_file}")


def _load_records(path: Path) -> List[Dict[str, Any]]:
    """Read a pipeline JSON that may be a bare list or a {"results": [...]} envelope."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get('results', [data])
    return data if isinstance(data, list) else []


async def retry_failed_synthesis(
    synth_file: Path,
    input_file: Path,
    rkg_file: Path,
    model: str = DEFAULT_MODEL,
    concurrency: int = 4,
    domain: str = "logical",
    anchor_conclusion: bool = False,
    no_mv: bool = False,
    prior_mode: str = "verify",
    atomic_steps: bool = False,
    prev_context: str = "all",
    idf_scope: str = "sample",
    idf_norm: str = "raw",
    df_table_path: Optional[Path] = None,
    in_place: bool = False,
) -> None:
    """Re-synthesize only the samples a prior run left without a pred_label.

    Reads the finished synthesis file, finds the samples that failed, runs RKG
    synthesis again for those alone, and merges the recovered ones back in. The
    IRF settings must match the original run, otherwise the retried samples are
    scored on a different term weighting than the ones beside them in the file.
    """
    synth_records = _load_records(synth_file)
    failed_sids = {r["sample_id"] for r in synth_records
                   if "sample_id" in r and not r.get("pred_label")}
    print(f"Failed samples to retry: {len(failed_sids)}")
    if not failed_sids:
        print("Nothing to retry.")
        return

    rkg_lookup     = {r["sample_id"]: r for r in _load_records(rkg_file) if "sample_id" in r}
    cleaned_lookup = {s["sample_id"]: s for s in _load_records(input_file) if "sample_id" in s}

    # A retry only sees the failed subset, so a sample-local corpus would differ from the
    # original run's; the global table has to come from the file that run saved.
    df_table = None
    if idf_scope == "global":
        if df_table_path is None or not Path(df_table_path).exists():
            raise ValueError(
                "--idf_scope global needs --df_table pointing at the table the original run "
                "saved; a retry sees only the failed subset and cannot rebuild that corpus"
            )
        df_table = DocFreqTable.load(Path(df_table_path))
        print(f"IRF scope: global | {df_table.n_docs} step documents, {len(df_table)} terms")
        check_df_table(df_table, domain, idf_norm == "log_n")
    elif idf_scope == "none":
        df_table = FlatDocFreqTable()
        print("IRF scope: none (IRF factor disabled, TF-IRF == TF)")
    else:
        print("IRF scope: sample (IDF computed within each sample's own steps)")
    print(f"IDF scale: {idf_norm}")

    semaphore = asyncio.Semaphore(concurrency)
    recovered: Dict[str, Dict[str, Any]] = {}
    connector = aiohttp.TCPConnector(limit=concurrency * 3)

    async with aiohttp.ClientSession(connector=connector) as session:
        async def worker(sid: str):
            async with semaphore:
                sample, rkg = cleaned_lookup.get(sid), rkg_lookup.get(sid)
                if not sample or not rkg:
                    return sid, {"sample_id": sid, "error": "missing_input",
                                 "synthesized_trace": None}
                try:
                    return sid, await synthesize_trace_rkg(
                        session, sample, rkg, model=model, domain=domain,
                        anchor_conclusion=anchor_conclusion, no_mv=no_mv,
                        prior_mode=prior_mode, atomic_steps=atomic_steps,
                        prev_context=prev_context,
                        df_table=df_table, idf_norm=(idf_norm == "log_n"),
                    )
                except Exception as e:
                    return sid, {"sample_id": sid, "error": f"retry_failed: {e}",
                                 "synthesized_trace": None}

        tasks = [asyncio.create_task(worker(sid)) for sid in failed_sids]
        for done, coro in enumerate(asyncio.as_completed(tasks), start=1):
            sid, result = await coro
            recovered[sid] = result
            print(f"  [{done}/{len(failed_sids)}] {sid} pred={result.get('pred_label')!r} "
                  f"err={(result.get('error') or '')[:60]!r}")

    merged = [recovered[r["sample_id"]]
              if r.get("sample_id") in recovered and recovered[r["sample_id"]].get("pred_label")
              else r
              for r in synth_records]

    out_path = Path(synth_file)
    if in_place:
        backup = out_path.with_suffix(".bak.json")
        out_path.rename(backup)
        print(f"Backup: {backup}")
    else:
        out_path = out_path.with_name(out_path.stem + "_retried.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({"results": merged}, f, indent=2, ensure_ascii=False)

    n_recovered = sum(1 for r in recovered.values() if r.get("pred_label"))
    print(f"\nWrote: {out_path}")
    print(f"Total: {len(merged)}, with pred_label: "
          f"{sum(1 for r in merged if r.get('pred_label'))}, recovered: {n_recovered}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a high-quality reasoning trace by combining terms from multiple traces"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the cleaned traces JSON input file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="synthesized_traces.json",
        help="Path to the output JSON file (relative paths resolve under the results root)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--min_tfidf",
        type=float,
        default=0.01,
        help="Minimum TF-IRF threshold (default: 0.01)"
    )
    parser.add_argument(
        "--idf_scope",
        type=str,
        default="sample",
        choices=["sample", "global", "none"],
        help="IRF corpus: 'sample' scores IDF within each sample's own steps (default, "
             "current behaviour); 'global' scores it over every step of every sample; "
             "'none' disables the IRF factor entirely, leaving TF-IRF == TF"
    )
    parser.add_argument(
        "--idf_norm",
        type=str,
        default="raw",
        choices=["raw", "log_n"],
        help="IDF scale: 'raw' is log(N/df) (default); 'log_n' divides by log(N) so the "
             "score lands in [0,1] and min_tfidf means the same under either --idf_scope"
    )
    parser.add_argument(
        "--df_table",
        type=str,
        default=None,
        help="Path to a global DF table JSON (used with --idf_scope global). Loaded if it "
             "exists, otherwise built from the input and saved here for reuse across modules"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Concurrency level (default: 20)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit the number of samples to process (for testing)"
    )
    parser.add_argument(
        "--original_file",
        type=str,
        default=None,
        help="Path to the original traces file (used to extract problem information if absent from cleaned_traces)"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key (overrides environment variable)"
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="API Base URL (overrides default)"
    )
    parser.add_argument(
        "--step_by_step",
        action="store_true",
        default=False,
        help="Shorthand for --synthesis_strategy step_by_step: generate one step at a "
             "time without the graph, each step seeing the previous ones"
    )
    parser.add_argument(
        "--all_at_once",
        action="store_true",
        default=False,
        help="Shorthand for --synthesis_strategy all_at_once: generate the whole trace "
             "in one call"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="logical",
        choices=["logical", "math"],
        help="Reasoning domain: 'logical' (logical reasoning, default) or 'math' (math reasoning, uses \\boxed{} conclusion format)",
    )
    parser.add_argument(
        "--synthesis_strategy",
        type=str,
        default="rkg",
        choices=["step_by_step", "all_at_once", "rkg"],
        help="Synthesis strategy. 'rkg' (default) is Module III as the paper describes "
             "it — a topological walk over the consensus RKG, so it requires --rkg_file. "
             "'step_by_step' and 'all_at_once' ignore the graph and are the ablation's "
             "'w/o RKG' setting rather than CRAFT",
    )
    parser.add_argument(
        "--rkg_file",
        type=str,
        default=None,
        help="RKG JSON file path (required for synthesis_strategy=rkg; from build_rkg.py)",
    )
    parser.add_argument(
        "--anchor_conclusion",
        action="store_true",
        default=False,
        help=(
            "For logical domain RKG synthesis: use the RKG consensus node_text "
            "directly for conclusion nodes instead of LLM re-generation. "
            "Recommended for complex FOL tasks where synthesis accuracy "
            "falls below MV baseline. NOT recommended for FLD where synthesis "
            "already beats MV through reference-guided intermediate steps."
        ),
    )

    parser.add_argument(
        "--prev_context", choices=["all", "direct"], default="all",
        help="What a step is shown of the trace so far: 'all' every step "
             "written before it, 'direct' only the ones the graph makes it "
             "depend on. 'all' is the default because withholding the rest "
             "cost 4.1 points on FLD and 3.6 extra steps when it was measured; "
             "the switch is here to measure it again, not because that is in "
             "doubt.")
    parser.add_argument(
        "--atomic_steps", action="store_true", default=False,
        help="Ask each synthesized step for one inference rather than for all "
             "the intermediate work. Raises the step count with it — the two "
             "cannot both be optimised, since a problem needing thirty "
             "inferences cannot be nine atomic steps",
    )
    parser.add_argument(
        "--prior_mode", choices=["verify", "follow"], default="verify",
        help="How Module III is told to treat the consensus vote when it writes "
             "the conclusion. 'verify' (default) gives it as a prior the derived "
             "chain may overrule; 'follow' asks for reasoning that leads to it. "
             "Selected per configuration on a validation split: a model whose "
             "single re-derivation is weaker than its own vote does better with "
             "'follow'.",
    )
    parser.add_argument(
        "--no_mv",
        action="store_true",
        default=False,
        help="Ablation: disable majority-vote closed-loop verification in RKG synthesis.",
    )

    # Repair mode: re-run only the samples a finished run left without a pred_label.
    parser.add_argument(
        "--retry_failed",
        action="store_true",
        default=False,
        help="Repair a finished run: re-synthesize the samples in --output that have no "
             "pred_label, reading problem text from --input and graphs from --rkg_file, "
             "and merge the recovered ones back. Pass the same IRF flags the run used.",
    )
    parser.add_argument(
        "--in_place",
        action="store_true",
        default=False,
        help="--retry_failed: overwrite --output (the original is kept as *.bak.json) "
             "instead of writing a *_retried.json beside it",
    )

    args = parser.parse_args()

    # Synthesis strategy
    # The two shorthands select a strategy, and saying both is a contradiction
    # rather than a precedence puzzle.
    if args.all_at_once and args.step_by_step:
        parser.error("--step_by_step and --all_at_once select different strategies; pass one")
    synthesis_strategy = args.synthesis_strategy
    if args.all_at_once:
        synthesis_strategy = "all_at_once"
    elif args.step_by_step:
        synthesis_strategy = "step_by_step"

    # Update configuration
    global OPENAI_API_KEY, OPENAI_BASE_URL, CHAT_COMPLETIONS_URL, HEADERS
    if args.api_key:
        OPENAI_API_KEY = args.api_key

    if args.base_url:
        OPENAI_BASE_URL = args.base_url
        CHAT_COMPLETIONS_URL = OPENAI_BASE_URL.rstrip("/") + "/chat/completions"

    HEADERS["Authorization"] = _build_auth_header(OPENAI_API_KEY, OPENAI_BASE_URL)
    
    input_path = _cfg.resolve_input(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    output_path = _cfg.resolve_output(args.output)

    if args.retry_failed:
        if args.rkg_file is None:
            parser.error("--retry_failed requires --rkg_file (the RKG the original run synthesized from)")
        synth_path = _cfg.resolve_input(args.output)
        if not synth_path.exists():
            raise FileNotFoundError(f"--retry_failed needs an existing synthesis file: {synth_path}")
        asyncio.run(
            retry_failed_synthesis(
                synth_file=synth_path,
                input_file=input_path,
                rkg_file=_cfg.resolve_input(args.rkg_file),
                model=args.model,
                concurrency=args.concurrency,
                domain=args.domain,
                anchor_conclusion=args.anchor_conclusion,
                no_mv=args.no_mv,
                prior_mode=args.prior_mode,
                atomic_steps=args.atomic_steps,
                prev_context=args.prev_context,
                idf_scope=args.idf_scope,
                idf_norm=args.idf_norm,
                df_table_path=resolve_df_table_path(args.df_table) if args.df_table else None,
                in_place=args.in_place,
            )
        )
        return

    original_path = _cfg.resolve_input(args.original_file) if args.original_file else None

    # Auto-detect original_file (k_traces) in the same directory as input
    # so problem text is always available for synthesis prompts
    if original_path is None:
        _input_dir = input_path.parent
        _candidates = sorted(_input_dir.glob("k_traces_*_samples.json"))
        if _candidates:
            original_path = _candidates[0]
            print(f"Auto-detected original_file: {original_path}")

    asyncio.run(
        synthesize_traces_for_dataset(
            input_path,
            output_path,
            min_tfidf=args.min_tfidf,
            model=args.model,
            concurrency=args.concurrency,
            max_samples=args.max_samples,
            original_file=original_path,
            step_by_step=(synthesis_strategy == "step_by_step"),
            domain=args.domain,
            synthesis_strategy=synthesis_strategy,
            rkg_file=_cfg.resolve_input(args.rkg_file) if args.rkg_file else None,
            anchor_conclusion=args.anchor_conclusion,
            no_mv=args.no_mv,
            prior_mode=args.prior_mode,
            atomic_steps=args.atomic_steps,
            prev_context=args.prev_context,
            idf_scope=args.idf_scope,
            idf_norm=args.idf_norm,
            df_table_path=resolve_df_table_path(args.df_table) if args.df_table else None,
        )
    )


if __name__ == "__main__":
    main()
