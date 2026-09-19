#!/usr/bin/env python3
"""
generate_traces.py  (Module I — Multi-Trace Generation)
---------------------------------------------------------------------------
Rolls out the K candidate traces the rest of CRAFT reaches consensus over.
Generate k diverse reasoning traces per sample from FLD/FOLIO datasets.

Key features:
- Loads unified-format logical reasoning datasets
- Generates k diverse reasoning traces per sample using linearly-spaced temperatures
  across [base-0.3, base+0.3] for diversity
- Unified OpenAI Chat Completions API calls
- Outputs results containing multiple reasoning traces per sample
- Comprehensive error handling and statistics
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

try:
    import aiohttp
except ModuleNotFoundError:
    aiohttp = None
try:
    import backoff
except ModuleNotFoundError:
    backoff = None
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None
try:
    from openai import AsyncOpenAI
except ModuleNotFoundError:
    AsyncOpenAI = None

def with_backoff(func: Callable) -> Callable:
    if backoff is None or aiohttp is None:
        return func
    exceptions = (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError)
    return backoff.on_exception(
        backoff.expo,
        exceptions,
        max_tries=8,
        factor=2,
    )(func)

if tqdm is None:
    class _DummyTqdm:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
        def update(self, *args: Any, **kwargs: Any) -> None:
            pass
        def close(self) -> None:
            pass
    def tqdm(*args: Any, **kwargs: Any) -> _DummyTqdm:
        return _DummyTqdm()

#########################
# OpenAI configuration — loaded from root config.py; switch models by editing config.py only
#########################
import importlib.util as _ilu, pathlib as _pl
_cfg_path = _pl.Path(__file__).resolve().parents[2] / "config.py"
_spec = _ilu.spec_from_file_location("_root_config", _cfg_path)
_cfg  = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_cfg)

OPENAI_API_KEY:  str            = _cfg.OPENAI_API_KEY
MODEL_NAME:      str            = _cfg.MODEL_TRACE_GEN   # Step 1: generate k diverse traces
OPENAI_BASE_URL: Optional[str]  = _cfg.OPENAI_BASE_URL
API_CLIENT:      Optional[AsyncOpenAI] = None
TEMPERATURE:     float          = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
MAX_OUTPUT_TOKENS: int          = int(os.getenv("MAX_OUTPUT_TOKENS", "2048"))
# Reasoning models (o4-mini, deepseek-r1) bill hidden reasoning against max_tokens.
REASONING_TOTAL_TOKENS: int     = int(os.getenv("REASONING_TOTAL_TOKENS", "16000"))
REASONING_VISIBLE_TOKENS: int   = int(os.getenv("REASONING_VISIBLE_TOKENS", "4096"))

# Relative — resolved under the results root by _cfg.resolve_output()
DEFAULT_OUTPUT_PATH = Path("generated_k_traces_reasoning.json")
DEFAULT_DATASET_PATH = _cfg.DATASET_ROOT / "label_prediction" / "logical" / "FLD.json"

HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY or ''}",
    "Content-Type": "application/json",
}

#########################
# Utility functions
#########################

def slugify_brand(name: str) -> str:
    """Convert a brand name into a safe key string."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

def _domain_of(sample: Dict[str, Any]) -> str:
    """Domain for a record that does not carry one: only ROSCOE's GSM8K is math."""
    return "math" if str(sample.get("_dataset", "")).lower() == "gsm8k" else "logical"


def _roscoe_problem_text(sample: Dict[str, Any]) -> Optional[str]:
    """Problem text for a ROSCOE record, which keeps the answer out of the prompt.

    ROSCOE's sets are built for verification rather than for asking, so `hypothesis`
    is not part of the question: for CosmosQA and DROP it holds the correct answer
    (the question is already appended to `premise`), and for GSM8K it holds the
    reference solution. eSNLI is the exception — there the hypothesis is the sentence
    whose relation to the premise is the task, so it has to be shown.
    """
    if "_dataset" not in sample or "gpt-3" not in sample:
        return None
    premise = str(sample.get("premise", "")).strip()
    if str(sample["_dataset"]).lower() == "esnli":
        hypothesis = str(sample.get("hypothesis", "")).strip()
        return f"Premise: {premise}\nHypothesis: {hypothesis}".strip()
    return premise


def extract_problem_text(sample: Dict[str, Any]) -> str:
    """Uniformly extract problem text, preferring the `input` field."""
    primary = sample.get("input")
    if isinstance(primary, str) and primary.strip():
        return primary.strip()

    roscoe = _roscoe_problem_text(sample)
    if roscoe:
        return roscoe

    parts: List[str] = []
    for key in ("Facts", "facts", "Premise", "ori_premises", "ori_conclusion", "hypothesis"):
        value = sample.get(key)
        if not value:
            continue
        if isinstance(value, str):
            text = value.strip()
        elif isinstance(value, list):
            text = "\n".join(str(v).strip() for v in value if str(v).strip())
        else:
            text = str(value).strip()
        if text and text not in parts:
            parts.append(text)

    return "\n".join(parts).strip() if parts else ""

def pick_target_answer(sample: Dict[str, Any]) -> Optional[str]:
    """Return proof_label first, then Label; for math domain return answer/Answer field."""
    # ROSCOE's `answer` is a human judgement of the trace it ships ("yes"/"no"), not
    # the problem's answer, so those records deliberately carry no target.
    if "_dataset" in sample and "gpt-3" in sample:
        return None
    for key in ("proof_label", "Label", "label", "answer", "Answer", "solution"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None

#########################
# Prompt construction
#########################

def build_reasoning_prompt(problem: str) -> List[Dict[str, str]]:
    """Build a logical reasoning prompt (logical domain)."""
    system_msg = {
        "role": "system",
        "content": (
            "You are a meticulous logician. Produce exhaustive, atomic reasoning. "
            "Each step must cite the facts or earlier steps it depends on. "
            "Avoid circular references and avoid skipping steps."
        ),
    }

    user_msg = {
        "role": "user",
        "content": (
            "Generate extremely detailed reasoning for the following problem.\n\n"
            f"Problem Statement:\n{problem}\n\n"
            "Instructions:\n"
            "- Provide 6-10 numbered steps in the format \"Step 1:\", \"Step 2:\", etc.\n"
            "- Each step must cite the exact facts or previous steps it uses.\n"
            "- Use natural language sentences without JSON or markdown code fences.\n"
            "- After the steps, include a short summary paragraph.\n"
            "- Do not stop early; if the answer is cut off, immediately continue until the summary and final conclusion are delivered.\n"
            "- Determine whether the hypothesis is __PROVED__ or __DISPROVED__ based on your reasoning.\n"
            "- IMPORTANT: Your response is incomplete without a final conclusion line. You MUST end with EXACTLY this format:\n"
            "  Final Conclusion: __PROVED__\n"
            "  (or __DISPROVED__ depending on your reasoning)\n"
            "- A response missing the Final Conclusion line will be rejected and retried."
        ),
    }
    return [system_msg, user_msg]


def build_reasoning_prompt_math(problem: str) -> List[Dict[str, str]]:
    """Build a mathematical reasoning prompt (math domain)."""
    system_msg = {
        "role": "system",
        "content": (
            "You are a meticulous mathematician. Produce exhaustive, step-by-step mathematical reasoning. "
            "Each step must show the complete transformation with full equations written out. "
            "Cite which previous result or given value you are using in each step. "
            "Never skip algebraic steps or arithmetic transitions."
        ),
    }

    user_msg = {
        "role": "user",
        "content": (
            "Solve the following math problem with detailed step-by-step reasoning.\n\n"
            f"Problem:\n{problem}\n\n"
            "Instructions:\n"
            "- Provide 4-10 numbered steps in the format \"Step 1:\", \"Step 2:\", etc.\n"
            "- In each step, write out the full equation or expression on BOTH sides (e.g., '2x + 3 = 7 → 2x = 4').\n"
            "- Each step must state which fact, given value, or previous result it uses.\n"
            "- Do NOT skip arithmetic or algebraic manipulations.\n"
            "- Use natural language sentences; do NOT use JSON or markdown code fences.\n"
            "- Do not stop early; continue until you reach a numeric or symbolic final answer.\n"
            "- The final step MUST state the answer clearly using \\boxed{<answer>} notation.\n"
            "- End with a single line exactly formatted as: Final Answer: \\boxed{<answer>}"
        ),
    }
    return [system_msg, user_msg]

#########################
# OpenAI API wrapper
#########################

# Every LLM request this module makes passes through one function, so counting
# there is the number of calls actually paid for — retries included — rather than
# the number a run was expected to need. Each stage writes it into its own
# metadata as api_calls, which is where the cost of a run is read from.
API_CALLS = {"count": 0}


@with_backoff
async def ask_model_text(session: aiohttp.ClientSession, messages: List[Dict[str, str]], temperature: float = None) -> str:
    """Call the Chat Completions endpoint and return the response text."""
    API_CALLS["count"] += 1
    if aiohttp is None:
        raise RuntimeError("Missing aiohttp dependency")

    REQUEST_TIMEOUT = 300
    temp = temperature if temperature is not None else TEMPERATURE

    # Reasoning models spend max_tokens on hidden reasoning before writing anything,
    # so a budget sized for the visible answer alone comes back empty. Give the total
    # a large ceiling and reserve a slice of it for the visible response.
    _is_reasoning = MODEL_NAME in ("o4-mini", "o4-mini-2025-04-16", "deepseek-r1")
    effective_max_tokens = MAX_OUTPUT_TOKENS
    _visible_tokens = None
    if _is_reasoning:
        effective_max_tokens = max(effective_max_tokens, REASONING_TOTAL_TOKENS)
        _visible_tokens = min(effective_max_tokens, REASONING_VISIBLE_TOKENS)

    if API_CLIENT is not None:
        data = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": temp,
            "max_tokens": effective_max_tokens,
            "timeout": REQUEST_TIMEOUT,
        }
        if OPENAI_BASE_URL:
            data["extra_body"] = {"max_output_tokens": _visible_tokens or effective_max_tokens}
        try:
            resp = await asyncio.wait_for(
                API_CLIENT.chat.completions.create(**data),
                timeout=REQUEST_TIMEOUT
            )
            if isinstance(resp, str):
                raise RuntimeError(f"{MODEL_NAME} API returned a string instead of a response object: {resp[:200]}")
            if not hasattr(resp, 'choices') or not resp.choices:
                raise RuntimeError(f"{MODEL_NAME} API response format error: {type(resp)}")
            msg = resp.choices[0].message
            # o4-mini / deepseek-r1 put reasoning in reasoning_content, not content
            if MODEL_NAME in ("o4-mini", "o4-mini-2025-04-16", "deepseek-r1"):
                return (getattr(msg, "reasoning_content", None) or msg.content or "").strip()
            return (msg.content or "").strip()
        except asyncio.TimeoutError:
            raise RuntimeError(f"{MODEL_NAME} request timed out ({REQUEST_TIMEOUT}s)")
        except Exception as e:
            if "'str' object has no attribute 'choices'" in str(e):
                raise RuntimeError(f"{MODEL_NAME} API returned unexpected format: {e}")
            raise
    else:
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": temp,
            "max_tokens": effective_max_tokens,
        }
        if _visible_tokens:
            payload["max_output_tokens"] = _visible_tokens
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers=HEADERS,
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
        ) as resp:
            if resp.status != 200:
                detail = await resp.text()
                raise RuntimeError(f"{MODEL_NAME} HTTP {resp.status}: {detail[:150]}")
            data = await resp.json()
            msg = data["choices"][0]["message"]
            # o4-mini / deepseek-r1 put reasoning in reasoning_content, not content
            if MODEL_NAME in ("o4-mini", "o4-mini-2025-04-16", "deepseek-r1"):
                return (msg.get("reasoning_content") or msg.get("content") or "").strip()
            return (msg.get("content") or "").strip()

#########################
# Core generation logic
#########################

VALID_LABELS = {"__PROVED__", "__DISPROVED__"}
LABEL_ORDER = ["__PROVED__", "__DISPROVED__"]

STEP_PATTERN = re.compile(r"^Step\s*\d+\s*:", re.IGNORECASE | re.MULTILINE)
FINAL_CONCLUSION_PATTERN = re.compile(
    r"Final\s+Conclusion\s*:\s*(__PROVED__|__DISPROVED__)", re.IGNORECASE
)
LABEL_TOKEN_PATTERN = re.compile(r"__PROVED__|__DISPROVED__", re.IGNORECASE)

# Math domain: extract boxed answer or plain numeric/expression answer
BOXED_PATTERN = re.compile(r"\\boxed\{([^}]+)\}", re.IGNORECASE)
MATH_ANSWER_PATTERN = re.compile(
    r"(?:the\s+answer\s+is|final\s+answer\s*[:\=]|answer\s*[:\=])\s*([^\n\.]+)", re.IGNORECASE
)

def sanitize_label(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    cleaned = re.sub(r"[\s_]", "", label).upper()
    if "DISPROVED" in cleaned:
        return "__DISPROVED__"
    if "PROVED" in cleaned:
        return "__PROVED__"
    return None

def extract_reasoning_steps(text: str) -> List[str]:
    if not text:
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    steps = [line for line in lines if STEP_PATTERN.match(line)]
    if steps:
        # Retain Final Conclusion lines so the RKG can detect the conclusion node
        conclusion_lines = [l for l in lines if FINAL_CONCLUSION_PATTERN.search(l) or LABEL_TOKEN_PATTERN.search(l)]
        # Avoid duplicating lines already in steps
        for cl in conclusion_lines:
            if cl not in steps:
                steps.append(cl)
        return steps
    return lines  # Retain all lines as-is, including conclusion lines

def extract_final_statement(text: str) -> str:
    if not text:
        return ""
    matches = list(FINAL_CONCLUSION_PATTERN.finditer(text))
    match = matches[-1] if matches else None
    if not match:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for candidate in reversed(lines):
            if STEP_PATTERN.match(candidate):
                continue
            if LABEL_TOKEN_PATTERN.search(candidate):
                return candidate
            if candidate.lower().startswith("final conclusion"):
                continue
            return candidate
        return ""
    preceding = text[: match.start()]
    lines = [line.strip() for line in preceding.splitlines() if line.strip()]
    for candidate in reversed(lines):
        if not STEP_PATTERN.match(candidate):
            return candidate
    return ""

def extract_label_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    matches = list(FINAL_CONCLUSION_PATTERN.finditer(text))
    if matches:
        return sanitize_label(matches[-1].group(1))

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        fallback = LABEL_TOKEN_PATTERN.search(line)
        if fallback:
            return sanitize_label(fallback.group(0))
    return None


def extract_math_answer_from_text(text: str) -> Optional[str]:
    """Extract the final answer from math reasoning text (\\boxed{} or 'the answer is ...')."""
    if not text:
        return None
    # Prefer \boxed{...}
    boxed_matches = BOXED_PATTERN.findall(text)
    if boxed_matches:
        return boxed_matches[-1].strip()
    # Fallback: "the answer is X" / "Final Answer: X"
    answer_matches = MATH_ANSWER_PATTERN.findall(text)
    if answer_matches:
        return answer_matches[-1].strip()
    return None

async def generate_single_trace(
    session: aiohttp.ClientSession,
    problem_text: str,
    trace_idx: int,
    k: int,
    base_temperature: float = 0.7,
    max_retries: int = 3,
    domain: str = "logical",
    fixed_temp: bool = False,
    temp_min: Optional[float] = None,
    temp_max: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Generate a single reasoning trace.

    Args:
        trace_idx: Index of the current trace (0-based)
        k: Total number of traces to generate
        base_temperature: Base temperature (used directly when fixed_temp=True)
        max_retries: Maximum number of retry attempts
        domain: "logical" or "math"
        fixed_temp: True → all traces use the same temperature (base_temperature), disables linear spacing
        temp_min: Lower bound for linear spacing (takes precedence over base±0.3 when specified)
        temp_max: Upper bound for linear spacing (takes precedence over base±0.3 when specified)

    Returns:
        Generated trace dict, or None on failure
    """
    if fixed_temp or k == 1:
        temperature = base_temperature
    else:
        # Arithmetic sampling: k traces uniformly cover [t_min, t_max]
        if temp_min is not None and temp_max is not None:
            t_min, t_max = temp_min, temp_max
        else:
            t_min = max(0.3, base_temperature - 0.3)
            t_max = min(1.2, base_temperature + 0.3)
        temperature = round(t_min + (t_max - t_min) * trace_idx / (k - 1), 2)

    if domain == "math":
        messages = build_reasoning_prompt_math(problem_text)
    else:
        messages = build_reasoning_prompt(problem_text)

    for attempt in range(max_retries):
        try:
            raw_text = await ask_model_text(session, messages, temperature=temperature)

            steps = extract_reasoning_steps(raw_text)
            reasoning_text = "\n".join(steps) if steps else raw_text.strip()
            final_statement = extract_final_statement(raw_text)

            if domain == "math":
                detected_label = extract_math_answer_from_text(raw_text)
                label = detected_label
            else:
                detected_label = extract_label_from_text(raw_text)
                label = detected_label

            # If no label found and retries remain, retry once
            if label is None and attempt < max_retries - 1:
                print(f"  [generate_single_trace] trace_idx={trace_idx} attempt={attempt+1}/{max_retries} no label found, retrying")
                continue

            return {
                "trace_idx": trace_idx,
                "temperature": temperature,
                "reasoning_steps": steps,
                "reasoning_text": reasoning_text,
                "label": label,
                "final_statement": final_statement,
                "raw_response": raw_text,
                "domain": domain,
            }
        except Exception as e:
            print(f"  [generate_single_trace] trace_idx={trace_idx} attempt={attempt+1}/{max_retries} error: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            else:
                return None

async def generate_k_traces_for_sample(
    session: aiohttp.ClientSession,
    sample_entry: Dict[str, Any],
    k: int,
    base_temperature: float = 0.7,
    domain: Optional[str] = None,
    fixed_temp: bool = False,
    temp_min: Optional[float] = None,
    temp_max: Optional[float] = None,
) -> Dict[str, Any]:
    """Generate k reasoning traces for a single sample."""
    problem_text = sample_entry["problem_text"]
    # Per-sample domain: read from sample_entry, fall back to parameter default
    sample_domain = sample_entry.get("domain", domain or "logical")

    tasks = [
        generate_single_trace(session, problem_text, trace_idx, k, base_temperature,
                              domain=sample_domain, fixed_temp=fixed_temp,
                              temp_min=temp_min, temp_max=temp_max)
        for trace_idx in range(k)
    ]
    traces_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out None values and exceptions
    traces = []
    errors = []
    for idx, result in enumerate(traces_results):
        if isinstance(result, Exception):
            errors.append({
                "trace_idx": idx,
                "error": str(result)
            })
        elif result is not None:
            traces.append(result)
        else:
            errors.append({
                "trace_idx": idx,
                "error": "Generation failed (returned None)"
            })

    return {
        "sample_id": sample_entry["sample_id"],
        "source_dataset": sample_entry["source_dataset"],
        "source_index": sample_entry["source_index"],
        "target_answer": sample_entry["target_answer"],
        "problem_text": problem_text,
        "domain": sample_entry.get("domain", "logical"),
        "traces": traces,
        "num_traces": len(traces),
        "requested_traces": k,
        "errors": errors if errors else None,
    }

async def generate_k_traces_dataset(
    samples: List[Dict[str, Any]],
    output_path: Path,
    k: int,
    concurrency: int,
    base_temperature: float = 0.7,
    domain: str = "logical",
    fixed_temp: bool = False,
    temp_min: Optional[float] = None,
    temp_max: Optional[float] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Process all samples in batch, generating k traces per sample.

    Returns:
        (results, statistics): list of results and statistics dictionary
    """
    if aiohttp is None:
        raise RuntimeError("Missing aiohttp dependency")

    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    results: List[Optional[Dict[str, Any]]] = [None] * len(samples)

    semaphore = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession(connector=connector) as session:
        pbar = tqdm(total=len(samples), desc=f"Generating {k} traces/sample", unit="sample")

        async def worker(idx: int, entry: Dict[str, Any]) -> None:
            async with semaphore:
                try:
                    # Per-sample domain: read from sample_entry, do not pass global domain parameter
                    results[idx] = await generate_k_traces_for_sample(
                        session, entry, k, base_temperature, domain=domain,
                        fixed_temp=fixed_temp, temp_min=temp_min, temp_max=temp_max
                    )
                except Exception as e:
                    results[idx] = {
                        "sample_id": entry.get("sample_id", f"unknown_{idx}"),
                        "error": str(e),
                        "traces": [],
                        "num_traces": 0,
                    }
                finally:
                    pbar.update(1)

        tasks = [asyncio.create_task(worker(idx, entry)) for idx, entry in enumerate(samples)]
        await asyncio.gather(*tasks)
        pbar.close()

    resolved = [res for res in results if res is not None]

    # Compute statistics
    total_samples = len(resolved)
    total_traces = sum(r.get("num_traces", 0) for r in resolved)
    samples_with_errors = sum(1 for r in resolved if r.get("errors") or r.get("error"))
    successful_traces = sum(r.get("num_traces", 0) for r in resolved if not r.get("error"))
    requested_traces = total_samples * k

    statistics = {
        "total_samples": total_samples,
        "requested_traces_per_sample": k,
        "total_requested_traces": requested_traces,
        "total_generated_traces": total_traces,
        "successful_traces": successful_traces,
        "failed_traces": requested_traces - successful_traces,
        "samples_with_errors": samples_with_errors,
        "success_rate": round(successful_traces / requested_traces * 100, 2) if requested_traces > 0 else 0,
        "average_traces_per_sample": round(total_traces / total_samples, 2) if total_samples > 0 else 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "metadata": {
            "model": MODEL_NAME,
            "temperature": base_temperature,
            "k": k,
            "api_calls": API_CALLS["count"],
            "statistics": statistics,
        },
        "results": resolved,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    return resolved, statistics

#########################
# Data loading
#########################

def load_unified_datasets(paths: Iterable[Path]) -> tuple[List[List[Dict[str, Any]]], Dict[Path, List[Dict[str, Any]]]]:
    """Load multiple JSON datasets with per-sample domain detection."""
    grouped_samples: List[List[Dict[str, Any]]] = []
    dataset_store: Dict[Path, List[Dict[str, Any]]] = {}

    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            if path.suffix == ".jsonl":
                # ROSCOE ships its four sets newline-delimited; every other dataset
                # here is one JSON list, and both end up as a list of records.
                data = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Dataset {path} is not a list")

        dataset_store[path] = data

        file_samples: List[Dict[str, Any]] = []
        for idx, item in enumerate(data):
            # Skip proof-type math problems: answer_type key present but None/empty
            # (OlympiadBench proof problems — distinct from GSM8K which has no answer_type key)
            sample_domain = item.get("domain") or _domain_of(item)
            if sample_domain == "math" and "answer_type" in item and not item.get("answer_type"):
                continue

            problem_text = extract_problem_text(item)
            target_answer = pick_target_answer(item)
            sample_id = f"{path.stem}_{idx}"
            file_samples.append({
                "sample_id": sample_id,
                "source_dataset": path.name,
                "dataset_path": path,
                "source_index": idx,
                "problem_text": problem_text,
                "target_answer": target_answer,
                "domain": sample_domain,
                "raw": item,
                "record": item,
            })
        grouped_samples.append(file_samples)

    return grouped_samples, dataset_store

def interleave_samples(grouped_samples: List[List[Dict[str, Any]]], total_limit: Optional[int]) -> List[Dict[str, Any]]:
    """Round-robin sample selection across dataset files."""
    if not grouped_samples:
        return []

    total_available = sum(len(g) for g in grouped_samples)
    limit = total_limit if total_limit is not None else total_available
    limit = min(limit, total_available)

    indices = [0] * len(grouped_samples)
    result: List[Dict[str, Any]] = []

    while len(result) < limit:
        progress = False
        for idx, group in enumerate(grouped_samples):
            pos = indices[idx]
            if pos < len(group):
                result.append(group[pos])
                indices[idx] += 1
                progress = True
                if len(result) >= limit:
                    break
        if not progress:
            break

    return result

#########################
# CLI entry point
#########################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate k diverse reasoning traces per sample"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[str(DEFAULT_DATASET_PATH)],
        help="Input dataset JSON file paths",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="OpenAI API model name, default gemini-2.5-flash-lite",
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="API key (passed directly)",
    )
    parser.add_argument(
        "--base_url",
        default=None,
        help="Custom OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output result JSON file path (relative paths resolve under the results root)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=50,
        help="Number of concurrent requests",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Number of items to process",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of reasoning traces to generate per item (paper: K=5)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Base temperature; distributed linearly over [base-0.3, base+0.3] by default",
    )
    parser.add_argument(
        "--fixed_temp",
        action="store_true",
        help="Fixed temperature mode: all k traces use the same temperature, no linear spacing",
    )
    parser.add_argument(
        "--temp_range",
        type=float, nargs=2, metavar=("MIN", "MAX"),
        help="Explicitly specify temperature range for linear spacing, e.g. --temp_range 0.3 0.6 (overrides base±0.3 default)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="logical",
        choices=["logical", "math"],
        help="Reasoning domain: 'logical' (logical reasoning, default) or 'math' (mathematical reasoning)",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    global MODEL_NAME, OPENAI_API_KEY, HEADERS, TEMPERATURE, MAX_OUTPUT_TOKENS
    if args.model:
        MODEL_NAME = args.model.strip()
    if args.api_key:
        OPENAI_API_KEY = args.api_key.strip()
    if OPENAI_API_KEY:
        HEADERS["Authorization"] = f"Bearer {OPENAI_API_KEY}"
    else:
        raise RuntimeError("No API key detected. Pass one via --api_key or the OPENAI_API_KEY environment variable")

    global OPENAI_BASE_URL, API_CLIENT
    if args.base_url:
        OPENAI_BASE_URL = args.base_url.strip()
    if OPENAI_BASE_URL and OPENAI_API_KEY:
        if AsyncOpenAI is None:
            raise RuntimeError("Missing openai dependency")
        API_CLIENT = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    TEMPERATURE = args.temperature

    dataset_paths = [_cfg.resolve_input(p).resolve() for p in args.datasets]
    for path in dataset_paths:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

    grouped_samples, dataset_store = load_unified_datasets(dataset_paths)
    samples = interleave_samples(grouped_samples, args.max_samples)

    # Compute per-sample domain distribution
    domain_counts: Dict[str, int] = {}
    for s in samples:
        d = s.get("domain", args.domain)
        domain_counts[d] = domain_counts.get(d, 0) + 1

    output_path = _cfg.resolve_output(args.output).resolve()

    print(f"Model: {MODEL_NAME}")
    print(f"Samples loaded: {len(samples)}")
    print(f"Traces per sample: {args.k}")
    temp_min = args.temp_range[0] if args.temp_range else None
    temp_max = args.temp_range[1] if args.temp_range else None
    if args.fixed_temp:
        temp_desc = f"fixed {args.temperature}"
    elif temp_min is not None:
        step = (temp_max - temp_min) / max(args.k - 1, 1)
        temp_desc = f"linear [{temp_min:.2f} -> {temp_max:.2f}], step={step:.3f}"
    else:
        t_min = max(0.3, args.temperature - 0.3)
        t_max = min(1.2, args.temperature + 0.3)
        temp_desc = f"linear [{t_min:.2f} -> {t_max:.2f}] (base±0.3)"
    print(f"Temperature mode: {temp_desc}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Reasoning domain (per-sample, auto-detected): {domain_counts}")
    print(f"Output file: {output_path}")

    results, statistics = asyncio.run(
        generate_k_traces_dataset(
            samples=samples,
            output_path=output_path,
            k=args.k,
            concurrency=args.concurrency,
            base_temperature=args.temperature,
            domain=args.domain,
            fixed_temp=args.fixed_temp,
            temp_min=temp_min,
            temp_max=temp_max,
        )
    )

    print(f"\nDone!")
    print(f"Statistics:")
    print(f"  - Total samples: {statistics['total_samples']}")
    print(f"  - Total requested traces: {statistics['total_requested_traces']}")
    print(f"  - Successfully generated traces: {statistics['successful_traces']}")
    print(f"  - Failed traces: {statistics['failed_traces']}")
    print(f"  - Success rate: {statistics['success_rate']}%")
    print(f"  - Average traces per sample: {statistics['average_traces_per_sample']}")
    print(f"  - Samples with errors: {statistics['samples_with_errors']}")
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
