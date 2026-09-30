#!/usr/bin/env python3
"""
generate_traces.py  (Module I — Multi-Trace Generation)
---------------------------------------------------------------------------
Rolls out the K candidate traces the rest of CRAFT reaches consensus over.
Generate k diverse reasoning traces per sample from the benchmark datasets.

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
# Table 1 hyperparameters, set once in config.py
K = _cfg.K
T = _cfg.T
T_SPREAD = _cfg.T_SPREAD


OPENAI_API_KEY:  str            = _cfg.OPENAI_API_KEY
MODEL_NAME:      str            = _cfg.MODEL_TRACE_GEN   # Step 1: generate k diverse traces
OPENAI_BASE_URL: Optional[str]  = _cfg.OPENAI_BASE_URL
API_CLIENT:      Optional[AsyncOpenAI] = None
TEMPERATURE:     float          = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
# The first budget of a request. A reply cut off by it is asked again with a
# larger one (framework/llm_reply.py), so a reasoning model that bills hidden
# reasoning against max_tokens gets the budget it needs without being named.
MAX_OUTPUT_TOKENS: int          = int(os.getenv("MAX_OUTPUT_TOKENS", "2048"))

# Relative — resolved under the results root by _cfg.resolve_output()
DEFAULT_OUTPUT_PATH = Path("generated_k_traces_reasoning.json")
DEFAULT_DATASET_PATH = _cfg.DATASET_ROOT / "FLD.json"

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

# Models whose API refused a temperature in this run. Their requests go out
# without one, so the model samples at its own default. No model is listed in
# advance: a model lands here only when its own API rejects a request that
# carries a temperature and then accepts the same request without it. A gateway
# that accepts any temperature (and may ignore it) never lands here.
TEMPERATURE_REJECTED: set = set()


def rollout_temperatures(k: int, t: float = None, spread: float = None) -> List[float]:
    """The K rollout temperatures: evenly spaced over [t - spread, t + spread].

    0.4, 0.55, 0.7, 0.85, 1.0 for the defaults K=5, T=0.7, T_SPREAD=0.3. The
    ends are clamped to [0.3, 1.2]; spread 0, or k 1, gives t for every trace.
    """
    t = T if t is None else t
    spread = T_SPREAD if spread is None else spread
    if k <= 1 or not spread:
        return [t] * max(k, 1)
    lo, hi = max(0.3, t - spread), min(1.2, t + spread)
    return [round(lo + (hi - lo) * i / (k - 1), 2) for i in range(k)]


def _may_be_temperature_refusal(status: Optional[int], error_text: str) -> bool:
    """Could this failure be the endpoint refusing the temperature parameter?

    Worth one retry without the temperature when it is a client error (400 or
    422, the codes an API uses for a parameter it will not take), or, where no
    status is available, when the message names the parameter. Whether it
    really was the temperature is settled by that retry, not by the wording,
    so a provider that phrases the refusal differently is handled the same way.
    """
    if status in (400, 422):
        return True
    return status is None and "temperature" in (error_text or "").lower()


def _status_of(error: Exception) -> Optional[int]:
    """The HTTP status an API client attached to its exception, if any."""
    for attr in ("status_code", "status", "http_status"):
        v = getattr(error, attr, None)
        if isinstance(v, int):
            return v
    return None


@with_backoff
async def ask_model_text(session: aiohttp.ClientSession, messages: List[Dict[str, str]], temperature: float = None,
                         sent: Optional[Dict[str, Any]] = None, max_tokens: Optional[int] = None) -> str:
    """Call the Chat Completions endpoint and return the response text.

    sent, when given, receives "temperature": the temperature the request that
    answered actually carried, or None when it went out without one. That is
    this request's own record, not the model's state at some later moment.
    """
    API_CALLS["count"] += 1
    if aiohttp is None:
        raise RuntimeError("Missing aiohttp dependency")

    REQUEST_TIMEOUT = 300
    temp = temperature if temperature is not None else TEMPERATURE

    effective_max_tokens = max_tokens or MAX_OUTPUT_TOKENS

    if API_CLIENT is not None:
        data = {
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": effective_max_tokens,
            "timeout": REQUEST_TIMEOUT,
        }
        if MODEL_NAME not in TEMPERATURE_REJECTED:
            data["temperature"] = temp
        if OPENAI_BASE_URL:
            data["extra_body"] = {"max_output_tokens": effective_max_tokens}
        try:
            used = data.get("temperature")
            try:
                resp = await asyncio.wait_for(
                    API_CLIENT.chat.completions.create(**data),
                    timeout=REQUEST_TIMEOUT
                )
            except Exception as e:
                # Asked again at once without the temperature; the backoff
                # retries would only repeat a refusal. If that succeeds the
                # temperature was what the API refused; if not, the first error
                # stands and the model keeps its temperature.
                if "temperature" not in data or not _may_be_temperature_refusal(_status_of(e), str(e)):
                    raise
                retry = {k: v for k, v in data.items() if k != "temperature"}
                try:
                    resp = await asyncio.wait_for(
                        API_CLIENT.chat.completions.create(**retry),
                        timeout=REQUEST_TIMEOUT
                    )
                except Exception:
                    raise e
                TEMPERATURE_REJECTED.add(MODEL_NAME)
                used = None
            if sent is not None:
                sent["temperature"] = used
            if isinstance(resp, str):
                raise RuntimeError(f"{MODEL_NAME} API returned a string instead of a response object: {resp[:200]}")
            if not hasattr(resp, 'choices') or not resp.choices:
                raise RuntimeError(f"{MODEL_NAME} API response format error: {type(resp)}")
            choice = resp.choices[0]
            if was_cut_off(choice):
                bigger = larger_budget(effective_max_tokens)
                if bigger is not None:
                    return await ask_model_text(session, messages, temperature, sent, max_tokens=bigger)
            return reply_text(choice.message)
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
            "max_tokens": effective_max_tokens,
        }
        if MODEL_NAME not in TEMPERATURE_REJECTED:
            payload["temperature"] = temp
        first_error = None
        for body in (payload, {k: v for k, v in payload.items() if k != "temperature"}):
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                json=body,
                headers=HEADERS,
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if first_error is not None:
                        TEMPERATURE_REJECTED.add(MODEL_NAME)
                    if sent is not None:
                        sent["temperature"] = body.get("temperature")
                    break
                detail = await resp.text()
                if first_error is None:
                    first_error = RuntimeError(f"{MODEL_NAME} HTTP {resp.status}: {detail[:150]}")
                    # Same rule as above: one retry without the temperature,
                    # and only a success there marks the model.
                    if "temperature" in payload and _may_be_temperature_refusal(resp.status, detail):
                        continue
                raise first_error
        choice = data["choices"][0]
        if was_cut_off(choice):
            bigger = larger_budget(effective_max_tokens)
            if bigger is not None:
                return await ask_model_text(session, messages, temperature, sent, max_tokens=bigger)
        return reply_text(choice["message"])

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

# Nested braces are read by the evaluation side's reader, so a trace's stored
# answer and the answer it is later scored on come from the same rules.
try:
    import sys as _sys
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]
                            / "evaluation" / "label_prediction"))
    from extract_label import _extract_boxed_content as _shared_boxed
except Exception:
    _shared_boxed = None
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

import sys as _sys_mt
import pathlib as _pl_mt
_sys_mt.path.insert(0, str(_pl_mt.Path(__file__).resolve().parents[2]))
from framework.domain_optimization.math_text import split_steps  # noqa: E402
from framework.llm_reply import larger_budget, reply_text, was_cut_off  # noqa: E402


def extract_reasoning_steps(text: str) -> List[str]:
    """Split a generation into its steps, each with the lines that belong to it.

    Keeping only the "Step N:" lines, as this once did, dropped whatever a
    model writes on the lines after the header: gpt-5.4-nano puts each step's
    mathematics in a display block on its own lines, so its OmniMATH and
    OlympiadBench traces kept 16% to 19% of their characters and every
    equation was lost before the TF-IRF terms, the z-score filter and the RKG
    ever saw the trace. The one splitter that keeps a block with its step is
    math_text.split_steps; a Final Conclusion line stays as a step of its own
    so the RKG can find the conclusion node.
    """
    return split_steps(text)

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
    """Extract the final answer from math reasoning text (\\boxed{} or 'the answer is ...').

    The shared reader in evaluation/label_prediction/extract_label.py is used for
    the \\boxed{} case. The local pattern here was \\boxed\\{([^}]+)\\}, which stops
    at the first closing brace and so cut every nested answer short: \\boxed{\\sqrt{2}}
    was recorded as "\\sqrt{2", \\boxed{\\frac{1}{2}} as "\\frac{1". On Omni-MATH that
    truncated 22% of the traces' stored labels, and those labels are what the
    k-trace vote is counted over and what Module III is handed as its
    majority-vote prior — so a cut answer was both scored wrong and fed forward.
    """
    if not text:
        return None
    # Prefer \boxed{...}
    if _shared_boxed is not None:
        boxed = _shared_boxed(text)
        if boxed:
            return boxed[-1].strip()
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
    base_temperature: float = T,
    max_retries: int = 3,
    domain: str = "logical",
    spread: float = T_SPREAD,
) -> Optional[Dict[str, Any]]:
    """Generate a single reasoning trace.

    Args:
        trace_idx: Index of the current trace (0-based)
        k: Total number of traces to generate
        base_temperature: T, the centre of the K rollout temperatures
        max_retries: Maximum number of retry attempts
        domain: "logical" or "math"
        spread: the K temperatures are spaced over [T - spread, T + spread];
            0 samples every trace at T (see rollout_temperatures)

    Returns:
        Generated trace dict, or None on failure. Its "temperature" is the one
        the request carried, or None when the model's API refused a temperature
        and the trace was sampled at the model's default.
    """
    temperature = rollout_temperatures(k, base_temperature, spread)[trace_idx]

    if domain == "math":
        messages = build_reasoning_prompt_math(problem_text)
    else:
        messages = build_reasoning_prompt(problem_text)

    for attempt in range(max_retries):
        try:
            sent: Dict[str, Any] = {}
            raw_text = await ask_model_text(session, messages, temperature=temperature, sent=sent)

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
                "temperature": sent.get("temperature", temperature),
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
    base_temperature: float = T,
    domain: Optional[str] = None,
    spread: float = T_SPREAD,
) -> Dict[str, Any]:
    """Generate k reasoning traces for a single sample."""
    problem_text = sample_entry["problem_text"]
    # Per-sample domain: read from sample_entry, fall back to parameter default
    sample_domain = sample_entry.get("domain", domain or "logical")

    tasks = [
        generate_single_trace(session, problem_text, trace_idx, k, base_temperature,
                              domain=sample_domain, spread=spread)
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
    base_temperature: float = T,
    domain: str = "logical",
    spread: float = T_SPREAD,
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
                        spread=spread
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
            "temperature_spread": spread,
            # Read back from the traces: the temperatures their requests
            # actually carried, and how many went out without one because
            # the model's API refused it (sampled at the model's default).
            "rollout_temperatures": sorted({t["temperature"] for r in resolved for t in r.get("traces", [])
                                            if t.get("temperature") is not None}),
            "traces_at_default_temperature": sum(1 for r in resolved for t in r.get("traces", [])
                                                 if t.get("temperature") is None),
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
        "--K",
        dest="k",
        type=int,
        default=K,
        help="Number of traces K rolled out per sample (default: config.K)",
    )
    parser.add_argument(
        "--T",
        dest="temperature",
        type=float,
        default=T,
        help="Centre T of the K rollout temperatures (default: config.T)",
    )
    parser.add_argument(
        "--T_spread",
        dest="spread",
        type=float,
        default=T_SPREAD,
        help="The K temperatures are spaced evenly over [T - spread, T + spread] "
             "(default: config.T_SPREAD, 0.3, which gives 0.4 ... 1.0 for K=5); "
             "0 samples every trace at T. A model whose API refuses a "
             "temperature is sampled at its default.",
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
    print(f"Rollout temperatures: {rollout_temperatures(args.k, args.temperature, args.spread)}")
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
            spread=args.spread,
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
