"""
receval_generate_traces.py
---------------------------
ReCEval — Reasoning Trace Quality: with_answer vs blind

Research question: Does telling the LLM the correct answer/label produce
higher-quality reasoning traces than letting it reason freely?

Supports two modes:

Mode 1 — Entailment Bank (original):
    python receval_generate_traces.py \\
        --input ReCEval/entailment_bank/.../test.jsonl \\
        --output receval_traces.json \\
        --model gpt-4.1-mini --concurrency 10

Mode 2 — ROSCOE datasets:
    Sample items from each of drop / esnli / cosmos / gsm8k,
    generate with_answer + blind traces, export for ROSCOE evaluation.

    python receval_generate_traces.py \\
        --roscoe_mode \\
        --roscoe_data_dir ParlAI/projects/roscoe/roscoe_data/generated \\
        --output roscoe_traces.json \\
        --export_dir roscoe_results/ \\
        --model gpt-4.1-mini --concurrency 10

Output format (both modes):
    [{
        "id": str,
        "dataset": str,          # NEW: source dataset name
        "hypothesis": str,
        "proof_label": str,      # ground truth answer
        "question": str,         # context / premises (joined)
        "with_answer": {"steps": [...], "raw": str},
        "blind":       {"steps": [...], "predicted_label": str, "raw": str}
    }]

Export for ROSCOE roscoe.py (via --export_dir):
    roscoe_results/with_answer.jsonl  — each line: {premise, hypothesis, gpt-3}
    roscoe_results/blind.jsonl        — each line: {premise, hypothesis, gpt-3}
    (field 'gpt-3' = our generated reasoning trace, matching ROSCOE's expected key)

Then score with:
    cd ParlAI
    python projects/roscoe/roscoe.py \\
        -t sentence_transformer \\
        -m all-mpnet-base-v2 \\
        --dataset-path ../roscoe_results/ \\
        --datasets with_answer blind
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
from pathlib import Path

import aiohttp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from config import (OPENAI_API_KEY, OPENAI_BASE_URL, DEFAULT_MODEL, REQUEST_TIMEOUT,
                        resolve_input, resolve_output)
except ImportError:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    REQUEST_TIMEOUT = int(os.getenv("OPENAI_REQUEST_TIMEOUT", "180"))
    resolve_input = resolve_output = Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts — Entailment Bank (original, logical reasoning)
# ---------------------------------------------------------------------------
SYSTEM_WITH_ANSWER = (
    "You are an expert at logical reasoning. "
    "Given a set of premises and a known conclusion, write a clear numbered "
    "step-by-step reasoning chain that derives the conclusion from the premises. "
    "Each step must be a single sentence and logically follow from prior steps or premises."
)

SYSTEM_BLIND = (
    "You are an expert at logical reasoning. "
    "Given a set of premises and a hypothesis, reason step-by-step to decide "
    "whether the hypothesis is proved or disproved based solely on the premises. "
    "Each step must be a single sentence. "
    "End the last line with exactly one of: __PROVED__ or __DISPROVED__."
)


def build_prompt_with_answer(hypothesis: str, premises: list[str], proof_label: str) -> str:
    prem_block = "\n".join(f"- {p}" for p in premises)
    return (
        f"Premises:\n{prem_block}\n\n"
        f"Conclusion: {hypothesis}\n"
        f"(This conclusion is verified as: {proof_label})\n\n"
        "Write a numbered step-by-step reasoning chain that logically derives "
        "this conclusion from the premises."
    )


def build_prompt_blind(hypothesis: str, premises: list[str]) -> str:
    prem_block = "\n".join(f"- {p}" for p in premises)
    return (
        f"Premises:\n{prem_block}\n\n"
        f"Hypothesis: {hypothesis}\n\n"
        "Write a numbered step-by-step reasoning chain. "
        "On the very last line, write exactly one of: __PROVED__ or __DISPROVED__."
    )


# ---------------------------------------------------------------------------
# Prompts — ROSCOE datasets (domain-specific, NEW)
# ---------------------------------------------------------------------------

# DROP / CosmosQA: reading comprehension + commonsense
SYSTEM_RC_WITH_ANSWER = (
    "You are an expert at reading comprehension and reasoning. "
    "Given a passage and a question with a known correct answer, write a clear numbered "
    "step-by-step reasoning chain that derives the answer from the passage. "
    "Each step must be a single sentence."
)

SYSTEM_RC_BLIND = (
    "You are an expert at reading comprehension and reasoning. "
    "Given a passage and a question, reason step-by-step to find the answer. "
    "Each step must be a single sentence. "
    "End with a final sentence stating your answer."
)

# e-SNLI: natural language inference
SYSTEM_NLI_WITH_ANSWER = (
    "You are an expert at natural language inference. "
    "Given two sentences (premise and hypothesis) and the known relationship between them, "
    "write a clear numbered step-by-step explanation of why the relationship holds. "
    "Each step must be a single sentence."
)

SYSTEM_NLI_BLIND = (
    "You are an expert at natural language inference. "
    "Given two sentences (premise and hypothesis), reason step-by-step to determine "
    "whether the hypothesis is entailed, contradicted, or neutral with respect to the premise. "
    "Each step must be a single sentence. "
    "End with a final sentence stating your conclusion (entailment / contradiction / neutral)."
)

# GSM8K: math word problems
SYSTEM_MATH_WITH_ANSWER = (
    "You are an expert at solving math word problems. "
    "Given a math problem and its correct final answer, write a clear numbered "
    "step-by-step solution that shows how to reach the answer. "
    "Each step must be a single sentence showing one calculation or logical deduction."
)

SYSTEM_MATH_BLIND = (
    "You are an expert at solving math word problems. "
    "Given a math problem, solve it step by step. "
    "Each step must be a single sentence showing one calculation or logical deduction. "
    "End with a final sentence stating the numeric answer."
)


def _rc_with_answer_prompt(premise: str, hypothesis: str, answer: str) -> str:
    return (
        f"Passage:\n{premise}\n\n"
        f"Question: {hypothesis}\n"
        f"Correct answer: {answer}\n\n"
        "Write a numbered step-by-step reasoning chain that derives the answer from the passage."
    )


def _rc_blind_prompt(premise: str, hypothesis: str) -> str:
    return (
        f"Passage:\n{premise}\n\n"
        f"Question: {hypothesis}\n\n"
        "Write a numbered step-by-step reasoning chain to answer the question."
    )


def _nli_with_answer_prompt(premise: str, hypothesis: str, answer: str) -> str:
    # answer in e-SNLI is 'yes'=entailment, 'no'=contradiction, 'maybe'=neutral
    label_map = {"yes": "entailment", "no": "contradiction", "maybe": "neutral"}
    label = label_map.get(answer.lower(), answer)
    return (
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        f"Relationship: {label}\n\n"
        "Write a numbered step-by-step explanation of why this relationship holds."
    )


def _nli_blind_prompt(premise: str, hypothesis: str) -> str:
    return (
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n\n"
        "Write a numbered step-by-step reasoning chain to determine the relationship "
        "(entailment / contradiction / neutral) between premise and hypothesis."
    )


def _math_with_answer_prompt(premise: str, answer: str) -> str:
    return (
        f"Problem: {premise}\n"
        f"Answer: {answer}\n\n"
        "Write a numbered step-by-step solution showing how to reach this answer."
    )


def _math_blind_prompt(premise: str) -> str:
    return (
        f"Problem: {premise}\n\n"
        "Write a numbered step-by-step solution."
    )


# ---------------------------------------------------------------------------
# ROSCOE dataset helpers
# ---------------------------------------------------------------------------

def load_roscoe_dataset(path: Path, dataset_name: str, n: Optional[int], seed: int = 42) -> list[dict]:
    """Load n random items from a ROSCOE dataset JSONL file. If n is None, load all items.

    Extracts unified fields:
        premise, hypothesis, answer (ground truth), dataset
    """
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    random.seed(seed)
    if n is None:
        chosen = list(items)
    else:
        chosen = random.sample(items, min(n, len(items)))

    unified = []
    for raw in chosen:
        item = {
            "id":        raw.get("key", f"{dataset_name}_{len(unified)}"),
            "dataset":   dataset_name,
            "premise":   raw.get("premise", ""),
            "hypothesis": raw.get("hypothesis", ""),
            "answer":    raw.get("answer", ""),
            # original gpt-3 trace (kept for reference, not used in generation)
            "original_gpt3": raw.get("gpt-3", ""),
        }
        # e-SNLI has multiple reference explanations
        if dataset_name == "esnli":
            item["reference_explanations"] = [
                raw.get("explanation_1", ""),
                raw.get("explanation_2", ""),
                raw.get("explanation_3", ""),
            ]
        unified.append(item)

    logger.info("Loaded %d items from %s (%s)", len(unified), dataset_name, path.name)
    return unified


def build_roscoe_prompts(item: dict) -> tuple[str, str, str, str]:
    """Return (system_wa, prompt_wa, system_blind, prompt_blind) for a ROSCOE item."""
    ds       = item["dataset"]
    premise  = item["premise"]
    hypo     = item["hypothesis"]
    answer   = item["answer"]

    if ds == "esnli":
        sys_wa   = SYSTEM_NLI_WITH_ANSWER
        sys_bl   = SYSTEM_NLI_BLIND
        p_wa     = _nli_with_answer_prompt(premise, hypo, answer)
        p_bl     = _nli_blind_prompt(premise, hypo)
    elif ds == "gsm8k":
        sys_wa   = SYSTEM_MATH_WITH_ANSWER
        sys_bl   = SYSTEM_MATH_BLIND
        p_wa     = _math_with_answer_prompt(premise, answer)
        p_bl     = _math_blind_prompt(premise)
    else:  # drop, cosmos
        sys_wa   = SYSTEM_RC_WITH_ANSWER
        sys_bl   = SYSTEM_RC_BLIND
        p_wa     = _rc_with_answer_prompt(premise, hypo, answer)
        p_bl     = _rc_blind_prompt(premise, hypo)

    return sys_wa, p_wa, sys_bl, p_bl


# ---------------------------------------------------------------------------
# Entailment Bank helpers (original)
# ---------------------------------------------------------------------------
def extract_premises(item: dict) -> list[str]:
    ctx = item.get("context", {})
    triples = ctx.get("triples", {})
    return [v.strip() for v in triples.values() if v.strip()]


def infer_proof_label(item: dict) -> str:
    return item.get("proof_label", "__PROVED__")


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
async def call_llm(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    system: str,
    user: str,
    model: str,
    temperature: float,
    api_key: str,
    base_url: str,
) -> str | None:
    url     = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Reasoning models (o1/o3/o4) use max_completion_tokens, not max_tokens
    _REASONING = ("o1", "o3", "o4", "o-1", "o-3", "o-4")
    is_reasoning = any(model.lower().startswith(p) for p in _REASONING)

    if is_reasoning:
        payload = {
            "model":                 model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "temperature":           0.5,
            "max_completion_tokens": 1024,
        }
    else:
        payload = {
            "model":       model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "temperature": temperature,
            "max_tokens":  1024,
        }

    for attempt in range(4):
        async with semaphore:
            try:
                async with session.post(
                    url, headers=headers, json=payload,
                    timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"].strip()
                    text = await resp.text()
                    logger.warning("HTTP %s: %s", resp.status, text[:200])
            except Exception as e:
                logger.warning("Attempt %d error: %s", attempt + 1, e)
        await asyncio.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------
def parse_steps(raw: str) -> list[str]:
    """Split numbered LLM output into step strings, stripping numbering."""
    steps = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line or line in ("__PROVED__", "__DISPROVED__"):
            continue
        line = re.sub(r"^(Step\s*)?\d+[.):\-]\s*", "", line, flags=re.IGNORECASE)
        if line:
            steps.append(line)
    return steps


def extract_predicted_label(raw: str) -> str:
    for marker in ("__PROVED__", "__DISPROVED__"):
        if marker in raw:
            return marker
    return ""


def export_for_roscoe(results: list[dict], export_dir: str) -> None:
    """Export generated traces in ROSCOE's expected JSONL format.

    ROSCOE's roscoe.py reads files where each line has:
        {
            "premise":   str,   # input context
            "hypothesis": str,  # question / hypothesis
            "gpt-3":     str,   # reasoning chain (our generated trace)
            "answer":    str,   # ground truth answer
            "key":       str,   # item id
            ... dataset-specific extras (e-SNLI: explanation_1/2/3)
        }

    We write two files:
        {export_dir}/with_answer.json   — traces generated with ground truth label
        {export_dir}/blind.json         — traces generated without label

    File names use .json extension (not .jsonl) because roscoe.py checks for
    dataset name prefix in the filename, not the extension.
    Each line is still newline-delimited JSON.
    """
    out = Path(export_dir)
    out.mkdir(parents=True, exist_ok=True)

    for setting in ("with_answer", "blind"):
        # Group by dataset so we can write one file per dataset per setting,
        # matching ROSCOE's convention: filename must start with dataset name.
        by_dataset: dict[str, list] = {}
        for r in results:
            ds = r.get("dataset", "unknown")
            by_dataset.setdefault(ds, []).append(r)

        for ds, items in by_dataset.items():
            path = out / f"{ds}_{setting}.json"
            with open(path, "w", encoding="utf-8") as f:
                for r in items:
                    trace_steps = r[setting]["steps"]
                    trace_text  = " ".join(trace_steps)

                    entry: dict = {
                        "key":       r["id"],
                        "premise":   r["question"],    # context for ROSCOE
                        "hypothesis": r["hypothesis"],
                        "answer":    r["proof_label"],
                        "gpt-3":     trace_text,       # ROSCOE reads this field
                        "dataset":   ds,
                        "setting":   setting,
                    }
                    # e-SNLI: include reference explanations for reference-based metrics
                    if ds == "esnli" and "reference_explanations" in r:
                        exps = r["reference_explanations"]
                        for i, exp in enumerate(exps, 1):
                            entry[f"explanation_{i}"] = exp

                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            logger.info("Exported %d %s/%s entries → %s",
                        len(items), ds, setting, path)


def export_for_receval(results: list[dict], export_dir: str) -> None:
    """Export traces for ReCEval's evaluate_receval.py (original EB format)."""
    Path(export_dir).mkdir(parents=True, exist_ok=True)
    for setting in ("with_answer", "blind"):
        path = Path(export_dir) / f"{setting}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in results:
                chain = r[setting]["steps"]
                entry = {
                    "question":   r["question"],
                    "steps":      " ".join(chain),
                    "sentences":  {"hypothesis": r["hypothesis"]},
                    "perturbed":  0,
                    "id":         r["id"],
                    "proof_label": r["proof_label"],
                    "setting":    setting,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("Exported %d entries for setting '%s' → %s",
                    len(results), setting, path)


# ---------------------------------------------------------------------------
# ROSCOE mode main
# ---------------------------------------------------------------------------
async def run_roscoe(args: argparse.Namespace) -> None:
    """Mode 2: generate traces on ROSCOE datasets and run evaluation end-to-end."""

    # ── Load items: from pre-saved sample file or fresh sampling ──────────
    if args.sampled_input and Path(args.sampled_input).exists():
        logger.info("Loading pre-sampled items from %s", args.sampled_input)
        with open(args.sampled_input, encoding="utf-8") as f:
            raw_items = json.load(f)
        # Convert raw ROSCOE format to unified format
        all_items: list[dict] = []
        for raw in raw_items:
            ds = raw.get("_dataset", raw.get("dataset", "unknown"))
            all_items.append({
                "id":       raw.get("key", f"{ds}_{len(all_items)}"),
                "dataset":  ds,
                "premise":  raw.get("premise", ""),
                "hypothesis": raw.get("hypothesis", ""),
                "answer":   raw.get("answer", ""),
                "original_gpt3": raw.get("gpt-3", ""),
                "reference_explanations": [
                    raw.get("explanation_1", ""),
                    raw.get("explanation_2", ""),
                    raw.get("explanation_3", ""),
                ] if ds == "esnli" else [],
            })
        logger.info("Loaded %d pre-sampled items", len(all_items))
    else:
        # Fresh sampling (saves to --sampled_output if specified)
        data_dir = Path(args.roscoe_data_dir)
        datasets = {
            "drop":   data_dir / "drop.json",
            "esnli":  data_dir / "esnli.json",
            "cosmos": data_dir / "cosmos.json",
            "gsm8k":  data_dir / "gsm8k.json",
        }
        for name, path in datasets.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"ROSCOE dataset not found: {path}\n"
                    f"Run: bash ParlAI/projects/roscoe/roscoe_data/download_annotated.sh"
                )
        all_items = []
        for name, path in datasets.items():
            items = load_roscoe_dataset(path, name, args.samples_per_dataset, seed=args.seed)
            all_items.extend(items)

        # Save sampled items for reproducibility
        sampled_out = Path(args.sampled_output) if args.sampled_output \
                      else Path(args.output).parent / "roscoe_sampled.json"
        sampled_out.parent.mkdir(parents=True, exist_ok=True)
        raw_save = []
        for item in all_items:
            entry = {
                "key": item["id"], "_dataset": item["dataset"],
                "premise": item["premise"], "hypothesis": item["hypothesis"],
                "answer": item["answer"], "gpt-3": item["original_gpt3"],
            }
            if item["dataset"] == "esnli":
                for i, exp in enumerate(item.get("reference_explanations", []), 1):
                    entry[f"explanation_{i}"] = exp
            raw_save.append(entry)
        with open(sampled_out, "w", encoding="utf-8") as f:
            json.dump(raw_save, f, indent=2, ensure_ascii=False)
        logger.info("Saved %d sampled items → %s (reuse with --sampled_input %s)",
                    len(all_items), sampled_out, sampled_out)

    logger.info("Total items: %d", len(all_items))

    # Build all LLM tasks concurrently
    semaphore = asyncio.Semaphore(args.concurrency)

    async with aiohttp.ClientSession() as session:
        wa_tasks, bl_tasks = [], []
        for item in all_items:
            sys_wa, p_wa, sys_bl, p_bl = build_roscoe_prompts(item)
            wa_tasks.append(call_llm(session, semaphore, sys_wa, p_wa,
                                     args.model, args.temperature,
                                     args.api_key, args.base_url))
            bl_tasks.append(call_llm(session, semaphore, sys_bl, p_bl,
                                     args.model, args.temperature,
                                     args.api_key, args.base_url))

        logger.info("Generating %d with_answer + %d blind traces...",
                    len(wa_tasks), len(bl_tasks))
        wa_outputs, bl_outputs = await asyncio.gather(
            asyncio.gather(*wa_tasks),
            asyncio.gather(*bl_tasks),
        )

    # Build results
    results = []
    for item, wa_raw, bl_raw in zip(all_items, wa_outputs, bl_outputs):
        results.append({
            "id":           item["id"],
            "dataset":      item["dataset"],
            "hypothesis":   item["hypothesis"],
            "proof_label":  item["answer"],         # ground truth answer
            "question":     item["premise"],         # context
            "with_answer": {
                "steps": parse_steps(wa_raw) if wa_raw else [],
                "raw":   wa_raw or "",
            },
            "blind": {
                "steps": parse_steps(bl_raw) if bl_raw else [],
                "raw":   bl_raw or "",
            },
            # keep reference explanations for esnli export
            "reference_explanations": item.get("reference_explanations", []),
        })

    # Save main output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    wa_ok = sum(1 for r in results if r["with_answer"]["steps"])
    bl_ok = sum(1 for r in results if r["blind"]["steps"])
    logger.info("Saved %d results → %s", len(results), out_path)
    logger.info("with_answer success: %d/%d", wa_ok, len(results))
    logger.info("blind success:       %d/%d", bl_ok, len(results))

    # Export for ROSCOE
    if args.export_dir:
        export_for_roscoe(results, args.export_dir)

        # Inline evaluation (end-to-end)
        if args.evaluate:
            logger.info("\nRunning ROSCOE evaluation inline...")
            roscoe_dir = Path(args.roscoe_parlai_dir) / "projects" / "roscoe"
            all_scores = run_roscoe_evaluation(
                export_dir=args.export_dir,
                roscoe_dir=str(roscoe_dir),
                transformer_model=args.roscoe_model,
                scores_output_dir=str(Path(args.export_dir) / "roscoe_scores"),
            )
            if all_scores:
                print_roscoe_results(all_scores)
                # Save aggregate JSON
                agg_path = Path(args.export_dir) / "roscoe_aggregate_scores.json"
                with open(agg_path, "w", encoding="utf-8") as f:
                    json.dump(all_scores, f, indent=2)
                logger.info("Aggregate scores saved → %s", agg_path)
        else:
            logger.info(
                "\nTo score with ROSCOE inline, re-run with --evaluate\n"
                "Or manually: cd %s && python projects/roscoe/roscoe.py "
                "-t sentence_transformer -m all-mpnet-base-v2 "
                "--dataset-path %s",
                args.roscoe_parlai_dir,
                str(Path(args.export_dir).resolve()),
            )


# ---------------------------------------------------------------------------
# Entailment Bank mode main (original)
# ---------------------------------------------------------------------------
async def run_generate(args: argparse.Namespace) -> None:
    items = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    if args.max_samples:
        items = items[: args.max_samples]
    logger.info("Loaded %d items from %s", len(items), args.input)

    semaphore = asyncio.Semaphore(args.concurrency)

    async with aiohttp.ClientSession() as session:
        wa_tasks, bl_tasks = [], []
        for item in items:
            premises   = extract_premises(item)
            hypothesis = item.get("hypothesis", "")
            proof_label = infer_proof_label(item)

            wa_tasks.append(call_llm(
                session, semaphore,
                SYSTEM_WITH_ANSWER,
                build_prompt_with_answer(hypothesis, premises, proof_label),
                args.model, args.temperature, args.api_key, args.base_url,
            ))
            bl_tasks.append(call_llm(
                session, semaphore,
                SYSTEM_BLIND,
                build_prompt_blind(hypothesis, premises),
                args.model, args.temperature, args.api_key, args.base_url,
            ))

        wa_outputs, bl_outputs = await asyncio.gather(
            asyncio.gather(*wa_tasks),
            asyncio.gather(*bl_tasks),
        )

    results = []
    label_correct = 0
    for item, wa_raw, bl_raw in zip(items, wa_outputs, bl_outputs):
        premises    = extract_premises(item)
        hypothesis  = item.get("hypothesis", "")
        gold_label  = infer_proof_label(item)
        predicted   = extract_predicted_label(bl_raw) if bl_raw else ""
        if predicted == gold_label:
            label_correct += 1

        results.append({
            "id":          item.get("id", ""),
            "dataset":     "entailment_bank",
            "hypothesis":  hypothesis,
            "proof_label": gold_label,
            "question":    " ".join(premises),
            "with_answer": {
                "steps": parse_steps(wa_raw) if wa_raw else [],
                "raw":   wa_raw or "",
            },
            "blind": {
                "steps": parse_steps(bl_raw) if bl_raw else [],
                "predicted_label": predicted,
                "raw":   bl_raw or "",
            },
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    n     = len(results)
    wa_ok = sum(1 for r in results if r["with_answer"]["steps"])
    bl_ok = sum(1 for r in results if r["blind"]["steps"])
    logger.info("Saved %d results → %s", n, out_path)
    logger.info("with_answer chain success: %d/%d", wa_ok, n)
    logger.info("blind chain success:       %d/%d", bl_ok, n)
    logger.info("blind label accuracy:      %d/%d (%.1f%%)",
                label_correct, n, 100 * label_correct / max(n, 1))

    if args.export_dir:
        export_for_receval(results, args.export_dir)


# ---------------------------------------------------------------------------
# ROSCOE Inline Evaluation
# ---------------------------------------------------------------------------

def run_roscoe_evaluation(
    export_dir: str,
    roscoe_dir: str,
    transformer_model: str = "all-mpnet-base-v2",
    scores_output_dir: str = None,
    discourse_batch: int = 64,
    coherence_batch: int = 16,
) -> dict:
    """Run ROSCOE scoring on the exported traces.

    Imports ROSCOE's Evaluator directly (no subprocess, no ParlAI CLI).
    Only uses sentence_transformer mode (no simcse required).

    Args:
        export_dir:         Directory containing {dataset}_{setting}.json files
        roscoe_dir:         Path to ParlAI/projects/roscoe/ directory
        transformer_model:  Sentence transformer model (default: all-mpnet-base-v2)
        scores_output_dir:  Where to save TSV score files (default: export_dir/scores/)
        discourse_batch:    Batch size for discourse metrics
        coherence_batch:    Batch size for coherence metrics

    Returns:
        dict: {setting: {dataset: {metric: mean_score}}}
    """
    import sys as _sys
    import os as _os

    roscoe_dir = Path(roscoe_dir).resolve()
    parlai_root = roscoe_dir.parent.parent  # ParlAI/
    _sys.path.insert(0, str(parlai_root))

    try:
        from projects.roscoe.score import (
            Evaluator,
            UNSUPERVISED_SCORES,
            REASONING_SCORES,
            SENT_TRANS,
            Chain,
        )
        from projects.roscoe.utils import split_gsm8k_gpt3_generations_to_steps
        from nltk.tokenize import sent_tokenize
    except ImportError as e:
        logger.error("Cannot import ROSCOE modules: %s", e)
        logger.error("Make sure ParlAI is installed: pip install parlai")
        return {}

    export_path = Path(export_dir)
    if scores_output_dir is None:
        scores_output_dir = str(export_path / "roscoe_scores")
    Path(scores_output_dir).mkdir(parents=True, exist_ok=True)

    # ── Inline ReasoningSteps (replicates roscoe.py's class) ───────────────
    class ReasoningSteps(Chain):
        def __init__(self, line: str, chain_type: str = "regular") -> None:
            self.chain = self._parse(line, chain_type)

        def _parse(self, chain: str, chain_type: str) -> list:
            if chain_type == "gsm8k_ref":
                return chain.split("IGNORE THIS. Ground truth here for reference. ")[1].split("\n")
            elif chain_type == "gsm8k_hypo":
                return split_gsm8k_gpt3_generations_to_steps(reasoning=chain)
            else:
                return sent_tokenize(chain)

    # ── Build evaluator once (model is loaded once and reused) ─────────────
    logger.info("Loading ROSCOE sentence transformer: %s", transformer_model)
    evaluator = Evaluator(
        hypos=[],
        context=[],
        references=[],
        model_type=SENT_TRANS,
        transformer_model=transformer_model,
        discourse_batch=discourse_batch,
        coherence_batch=coherence_batch,
    )

    all_results: dict = {}
    json_files = sorted(export_path.glob("*.json"))

    if not json_files:
        logger.warning("No .json files found in %s", export_path)
        return {}

    for json_file in json_files:
        fname = json_file.name  # e.g. "drop_with_answer.json"
        # Parse dataset name and setting from filename
        # Filename format: {dataset}_{setting}.json  (setting = with_answer or blind)
        stem = json_file.stem  # "drop_with_answer"
        if stem.endswith("_with_answer"):
            dataset = stem[: -len("_with_answer")]
            setting = "with_answer"
        elif stem.endswith("_blind"):
            dataset = stem[: -len("_blind")]
            setting = "blind"
        else:
            logger.warning("Skipping unrecognized filename: %s", fname)
            continue

        logger.info("Scoring %s / %s ...", dataset, setting)

        # ── Load items and build Chain objects ─────────────────────────────
        hypotheses, contexts, refs = [], [], []
        with open(json_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                jline = json.loads(line)
                trace = jline.get("gpt-3", "")
                premise = jline.get("premise", "")
                hypo = jline.get("hypothesis", "")

                if dataset == "gsm8k":
                    h_chain = ReasoningSteps(line=trace, chain_type="gsm8k_hypo")
                    ctx = ReasoningSteps(line=premise)
                    ref_text = jline.get("hypothesis", "")
                    r_chain = ReasoningSteps(line=ref_text, chain_type="gsm8k_ref")
                    refs.append(r_chain)
                else:
                    h_chain = ReasoningSteps(line=trace)
                    ctx = ReasoningSteps(line=premise + " " + hypo)
                    if dataset == "esnli":
                        ref_text = " ".join(filter(None, [
                            jline.get("explanation_1", ""),
                            jline.get("explanation_2", ""),
                            jline.get("explanation_3", ""),
                        ]))
                        refs.append(ReasoningSteps(line=ref_text))

                hypotheses.append(h_chain)
                contexts.append(ctx)

        # ── Choose score types (reference-based only when refs available) ──
        has_refs = len(refs) == len(hypotheses)
        score_types = REASONING_SCORES if has_refs else UNSUPERVISED_SCORES

        # ── Feed into evaluator ────────────────────────────────────────────
        evaluator.set_hypos(hypotheses)
        evaluator.set_context(contexts)
        evaluator.set_references(refs if has_refs else [])

        scores = evaluator.evaluate(score_types=score_types)

        # ── Save TSV ───────────────────────────────────────────────────────
        tsv_path = _os.path.join(scores_output_dir, f"scores_{stem}.tsv")
        score_list = list(scores.keys())
        with open(tsv_path, "w") as tf:
            header = "{:<8} ".format("ID") + " ".join("{:<15}".format(s) for s in score_list)
            tf.write(header + "\n")
            n = len(scores[score_list[0]])
            for i in range(n):
                row = "{:<8} ".format(i) + " ".join("{:<15}".format(scores[s][i]) for s in score_list)
                tf.write(row + "\n")
        logger.info("Scores saved → %s", tsv_path)

        # ── Compute per-metric mean (ignore "N/A") ─────────────────────────
        mean_scores = {}
        for metric, vals in scores.items():
            numeric = [v for v in vals if v != "N/A"]
            mean_scores[metric] = round(float(sum(numeric) / len(numeric)), 4) if numeric else None

        all_results.setdefault(setting, {})[dataset] = mean_scores

    return all_results


def print_roscoe_results(all_results: dict) -> None:
    """Print a comparison table: with_answer vs blind per dataset and metric."""
    if not all_results:
        return

    settings = list(all_results.keys())
    datasets  = sorted({ds for s in all_results.values() for ds in s})
    all_metrics = sorted({m for s in all_results.values()
                          for ds_scores in s.values()
                          for m in ds_scores})

    print()
    print("=" * 90)
    print("  ROSCOE EVALUATION RESULTS  (with_answer vs blind)")
    print("=" * 90)

    for dataset in datasets:
        print(f"\n  [{dataset.upper()}]")
        print(f"  {'Metric':<35}", end="")
        for s in settings:
            print(f"  {s:>14}", end="")
        if len(settings) == 2:
            print(f"  {'Δ(A-B)':>10}", end="")
        print()
        print("  " + "─" * (35 + len(settings) * 16 + 12))

        for metric in all_metrics:
            vals = []
            for s in settings:
                v = all_results.get(s, {}).get(dataset, {}).get(metric)
                vals.append(v)
            if all(v is None for v in vals):
                continue
            print(f"  {metric:<35}", end="")
            for v in vals:
                print(f"  {f'{v:.4f}' if v is not None else 'N/A':>14}", end="")
            if len(vals) == 2 and vals[0] is not None and vals[1] is not None:
                delta = vals[0] - vals[1]
                sign  = "+" if delta > 0 else ""
                print(f"  {sign+f'{delta:.4f}':>10}", end="")
            print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "ReCEval / ROSCOE — Generate reasoning traces (with_answer vs blind).\n"
            "Two modes: --roscoe_mode (ROSCOE datasets) or --input (Entailment Bank)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode selection
    parser.add_argument("--roscoe_mode", action="store_true",
                        help="Use ROSCOE datasets (drop/esnli/cosmos/gsm8k) instead of Entailment Bank")

    # ROSCOE mode args
    parser.add_argument("--roscoe_data_dir",
                        default="ParlAI/projects/roscoe/roscoe_data/generated",
                        help="Path to ROSCOE generated/ data directory")
    parser.add_argument("--samples_per_dataset", type=int, default=None,
                        help="Items to sample from each ROSCOE dataset")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--sampled_input", default=None,
                        help="Path to pre-saved sampled JSON (skips sampling, ensures reproducibility). "
                             "Use roscoe_results/roscoe_sampled.json")
    parser.add_argument("--sampled_output", default=None,
                        help="Where to save the sampled items JSON (default: next to --output)")

    # Entailment Bank mode args
    parser.add_argument("--input", default=None,
                        help="Entailment Bank JSONL input file (required without --roscoe_mode)")

    # Shared args
    parser.add_argument("--output", required=True,
                        help="Main output JSON path (relative paths resolve under the results root)")
    parser.add_argument("--export_dir", default=None,
                        help="Export per-setting JSONL for ROSCOE/ReCEval scoring "
                             "(relative paths resolve under the results root)")
    parser.add_argument("--model",       default=DEFAULT_MODEL)
    parser.add_argument("--base_url",    default=OPENAI_BASE_URL)
    parser.add_argument("--api_key",     default=OPENAI_API_KEY)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--concurrency", type=int,   default=10)
    parser.add_argument("--max_samples", type=int,   default=None,
                        help="Limit number of items (Entailment Bank mode only)")

    # ROSCOE evaluation args
    parser.add_argument("--evaluate", action="store_true",
                        help="Run ROSCOE evaluation inline after generating traces "
                             "(requires --export_dir and --roscoe_mode)")
    parser.add_argument("--roscoe_parlai_dir",
                        default=str(Path(__file__).resolve().parent / "ParlAI"),
                        help="Path to the ParlAI root directory (contains projects/roscoe/)")
    parser.add_argument("--roscoe_model", default="all-mpnet-base-v2",
                        help="Sentence transformer model for ROSCOE scoring "
                             "(default: all-mpnet-base-v2)")

    args = parser.parse_args()

    args.output = str(resolve_output(args.output))
    if args.export_dir:
        args.export_dir = str(resolve_output(args.export_dir))
    if args.sampled_output:
        args.sampled_output = str(resolve_output(args.sampled_output))
    if args.input:
        args.input = str(resolve_input(args.input))
    if args.sampled_input:
        args.sampled_input = str(resolve_input(args.sampled_input))

    if args.roscoe_mode:
        asyncio.run(run_roscoe(args))
    else:
        if not args.input:
            parser.error("--input is required when not using --roscoe_mode")
        asyncio.run(run_generate(args))


if __name__ == "__main__":
    main()
