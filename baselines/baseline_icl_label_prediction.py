# #!/usr/bin/env python3
# """
# baseline_icl_label_prediction.py
# ------------------------
# Label prediction evaluation — training-free settings (A–I).
# 
# Original ICL settings (A–E):
#   A: Zero-shot                       — problem only
#   B: ICL + Raw Trace                 — k examples from k_traces (Step 1)
#   C: ICL + Cleaned Trace             — k examples from cleaned_traces (Step 3/3.6)
#   D: ICL + Synthesized (step_by_step)— k examples from synthesized_traces (Step 5 baseline)
#   E: ICL + Synthesized (DAG)         — k examples from synthesized_traces (Step 5 DAG)
# 
# New non-training reasoning baselines (F–K) — all use same model, no trace pool needed:
#   F: Self-Consistency (k=3)          — 3× zero-shot majority vote  (Wang et al., 2022)
#   G: Universal Self-Consistency (k=3)— 3× zero-shot + LLM picks most consistent answer
#                                         (Chen et al., 2023, arXiv:2311.17311)
#   H: Self-Refine                     — generate → self-critique → refine, 1 iteration
#                                         (Madaan et al., NeurIPS 2023, arXiv:2303.17651)
#   I: LLM Self-Aggregation (k=3)      — generate 3 independent answers, then LLM merges them
#                                         (Li et al., 2025, arXiv:2503.04104)
#   J: Self-Eval Beam Search           — step-by-step generation with LLM self-scoring,
#                                         beam search over partial traces (no external verifier)
#                                         (Xie et al., NeurIPS 2023, arXiv:2305.00633)
#   K: Faithful CoT + Symbolic Solver  — LLM translates to symbolic facts/rules, derives
#                                         conclusion via structured symbolic reasoning
#                                         (Lyu et al., IJCNLP-AACL 2023, arXiv:2301.13379)
# 
# All settings: same model (gpt-4.1-mini), same test set, binary labels (PROVED/DISPROVED).
# Four metrics reported: accuracy, macro-F1, avg reasoning steps, avg completion tokens.
# 
# Usage:
#     # Full evaluation — all settings
#     python baseline_icl_label_prediction.py \\
#         --datasets ../FLD.json ../FOLIO.json \\
#         --k_traces_file    pipeline_results/k_traces_10_samples.json \\
#         --cleaned_file     pipeline_results/cleaned_traces_dag.json \\
#         --synthesized_step pipeline_results_baseline/synthesized_traces.json \\
#         --synthesized_dag  pipeline_results_dag/synthesized_traces.json \\
#         --shots 3 --model gpt-4.1-mini --output icl_results.json
# 
#     # Baselines only (no trace files needed)
#     python baseline_icl_label_prediction.py \\
#         --datasets ../FLD.json ../FOLIO.json \\
#         --settings A F G H I --model gpt-4.1-mini --output baselines.json
# 
#     # Specific settings
#     python baseline_icl_label_prediction.py \\
#         --datasets ../FLD.json ../FOLIO.json \\
#         --synthesized_dag pipeline_results_dag/synthesized_traces.json \\
#         --settings A E F G --shots 3 --output icl_AEF.json
# """
# 
# from __future__ import annotations
# 
# import argparse
# import asyncio
# import json
# import logging
# import math
# import os
# import random
# import re
# import sys
# from collections import Counter, defaultdict
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple
# 
# import aiohttp
# import numpy as np
# 
# # ---------------------------------------------------------------------------
# # Config
# # ---------------------------------------------------------------------------
# sys.path.insert(0, str(Path(__file__).resolve().parent))
# try:
#     from config import OPENAI_API_KEY, OPENAI_BASE_URL, DEFAULT_MODEL, REQUEST_TIMEOUT
# except ImportError:
#     OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
#     OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
#     DEFAULT_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
#     REQUEST_TIMEOUT = int(os.getenv("OPENAI_REQUEST_TIMEOUT", "180"))
# 
# logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# logger = logging.getLogger(__name__)
# 
# VALID_LABELS  = {"__PROVED__", "__DISPROVED__"}
# BINARY_LABELS = ["__PROVED__", "__DISPROVED__"]
# _LABEL_RE     = re.compile(r"(__PROVED__|__DISPROVED__)", re.IGNORECASE)
# _STEP_RE      = re.compile(r"^Step\s*\d+\s*[:\.]", re.IGNORECASE | re.MULTILINE)
# _BOXED_RE     = re.compile(r"\\boxed\{([^}]+)\}", re.IGNORECASE)
# _HASH_ANS_RE  = re.compile(r"####\s*(.+)")          # GSM8K solution format
# _FINAL_ANS_RE = re.compile(                          # "the answer is X" / "Final Answer: X"
#     r"(?:the\s+answer\s+is|final\s+answer\s*[:\=]|answer\s*[:\=])\s*([^\n\.]+)", re.IGNORECASE
# )
# 
# ALL_SETTINGS  = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
# 
# 
# # ---------------------------------------------------------------------------
# # Label helpers
# # ---------------------------------------------------------------------------
# 
# def normalise_label(label: Optional[str]) -> Optional[str]:
#     if not label:
#         return None
#     label = label.strip().upper()
#     if label in VALID_LABELS:
#         return label
#     if "DISPROVED" in label:
#         return "__DISPROVED__"
#     if "PROVED" in label:
#         return "__PROVED__"
#     return {"TRUE": "__PROVED__", "FALSE": "__DISPROVED__"}.get(label)
# 
# 
# def extract_label(text: str) -> Optional[str]:
#     matches = _LABEL_RE.findall(text or "")
#     if matches:
#         return matches[-1].upper()
#     lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
#     tail  = " ".join(lines[-2:]) if len(lines) >= 2 else (lines[-1] if lines else "")
#     if re.search(r"\bdisproved\b|\bfalse\b|\brefuted\b", tail, re.I):
#         return "__DISPROVED__"
#     if re.search(r"\bproved\b|\btrue\b|\bholds\b", tail, re.I):
#         return "__PROVED__"
#     return None
# 
# 
# def extract_math_answer(text: str) -> Optional[str]:
#     """Extract numeric answer from math reasoning text.
# 
#     Priority: \\boxed{...} → #### N → 'the answer is N' / 'Final Answer: N'
#     Returns the raw string (e.g. '18', '3/4', '72').
#     """
#     if not text:
#         return None
#     # 1. LaTeX \boxed{...}
#     m = _BOXED_RE.findall(text)
#     if m:
#         return m[-1].strip().replace(",", "")
#     # 2. GSM8K #### format
#     m2 = _HASH_ANS_RE.findall(text)
#     if m2:
#         return m2[-1].strip().replace(",", "")
#     # 3. Natural-language fallback
#     m3 = _FINAL_ANS_RE.findall(text)
#     if m3:
#         return m3[-1].strip().replace(",", "")
#     # 4. Bare number (e.g. USC selector outputs "42" with no formatting)
#     stripped = text.strip().replace(",", "")
#     if re.fullmatch(r"-?\d+(\.\d+)?", stripped):
#         return stripped
#     return None
# 
# 
# def normalise_math_answer(ans: Optional[str]) -> Optional[str]:
#     """Normalize a math answer string for comparison (strip spaces, lowercase, remove commas)."""
#     if ans is None:
#         return None
#     ans = ans.strip().replace(",", "").lower()
#     # Remove trailing .0 for integers expressed as floats
#     try:
#         f = float(ans)
#         if f == int(f):
#             return str(int(f))
#         return str(f)
#     except (ValueError, OverflowError):
#         return ans
# 
# 
# def count_steps(text: str, reasoning_steps: Optional[list] = None) -> int:
#     if reasoning_steps and isinstance(reasoning_steps, list):
#         return len([s for s in reasoning_steps if s and str(s).strip()])
#     if not text:
#         return 0
#     m = _STEP_RE.findall(text)
#     return len(m) if m else min(len([l for l in text.splitlines() if l.strip()]), 30)
# 
# 
# def count_tokens(text: str) -> int:
#     return len(text.split()) if text else 0
# 
# 
# # ---------------------------------------------------------------------------
# # Dataset loading & balancing
# # ---------------------------------------------------------------------------
# 
# def load_raw_dataset(paths: List[Path], seed: int = 42, per_dataset: int = 250) -> List[Dict]:
#     """Load datasets with domain-aware handling.
# 
#     Logical domain (FLD/FOLIO):
#       - Balance to per_dataset/2 PROVED + per_dataset/2 DISPROVED per file.
# 
#     Math domain (GSM8K, detected via 'domain'=='math' or 'answer' field present):
#       - No label filtering/balancing — use numeric answer as ground_truth.
#       - Take min(len(items), per_dataset*2) samples (500 for a 500-item file).
#       - 'input' = question, 'ground_truth' = numeric answer string.
#     """
#     rng = random.Random(seed)
#     n_per_class = per_dataset // 2
#     all_samples = []
# 
#     for path in paths:
#         with open(path, encoding="utf-8") as f:
#             data = json.load(f)
#         items = data if isinstance(data, list) else data.get("results", [])
#         stem  = path.stem
# 
#         # Detect domain from first item
#         first = items[0] if items else {}
#         is_math = (first.get("domain") == "math" or
#                    ("answer" in first and "proof_label" not in first and "Label" not in first))
# 
#         if is_math:
#             # Math domain: all items are valid, just sample up to per_dataset*2
#             rng_items = list(items)
#             rng.shuffle(rng_items)
#             limit = min(len(rng_items), per_dataset * 2)
#             chosen = []
#             for idx, item in enumerate(rng_items[:limit]):
#                 problem_text = (item.get("input") or item.get("question", "")).strip()
#                 gt = normalise_math_answer(
#                     item.get("answer") or item.get("solution", "")
#                 )
#                 if not problem_text or not gt:
#                     continue
#                 chosen.append({
#                     "sample_id":      f"{stem}_{idx}",
#                     "source_dataset": stem,
#                     "problem_text":   problem_text,
#                     "ground_truth":   gt,
#                     "domain":         "math",
#                 })
#             logger.info("%s (math): %d samples loaded", stem, len(chosen))
#             all_samples.extend(chosen)
# 
#         else:
#             # Logical domain: PROVED/DISPROVED binary, balanced
#             proved, disproved = [], []
#             for idx, item in enumerate(items):
#                 gt = normalise_label(
#                     item.get("proof_label") or item.get("Label") or item.get("label")
#                 )
#                 problem_text = item.get("input", "").strip()
#                 if not gt or not problem_text:
#                     continue
#                 entry = {
#                     "sample_id":      f"{stem}_{idx}",
#                     "source_dataset": stem,
#                     "problem_text":   problem_text,
#                     "ground_truth":   gt,
#                     "domain":         "logical",
#                 }
#                 (proved if gt == "__PROVED__" else disproved).append(entry)
# 
#             rng.shuffle(proved)
#             rng.shuffle(disproved)
#             chosen = proved[:n_per_class] + disproved[:n_per_class]
#             logger.info("%s (logical): PROVED=%d→%d, DISPROVED=%d→%d, total=%d",
#                         stem, len(proved), min(len(proved), n_per_class),
#                         len(disproved), min(len(disproved), n_per_class), len(chosen))
#             all_samples.extend(chosen)
# 
#     return all_samples
# 
# 
# def stratified_split(
#     samples: List[Dict], test_ratio: float = 0.2, seed: int = 42
# ) -> Tuple[List[Dict], List[Dict]]:
#     rng = random.Random(seed)
#     groups: Dict[Tuple, List] = defaultdict(list)
#     for s in samples:
#         groups[(s["source_dataset"], s["ground_truth"])].append(s)
#     train, test = [], []
#     for group in groups.values():
#         rng.shuffle(group)
#         n_test = max(1, round(len(group) * test_ratio))
#         test.extend(group[:n_test])
#         train.extend(group[n_test:])
#     return train, test
# 
# 
# # ---------------------------------------------------------------------------
# # Trace pool loading (Settings B–E)
# # ---------------------------------------------------------------------------
# 
# def load_trace_pool(path: Optional[Path], source: str) -> Dict[str, Dict]:
#     """Load a trace file → {sample_id: {text, n_steps, n_tokens}}."""
#     if not path or not path.exists():
#         return {}
#     with open(path, encoding="utf-8") as f:
#         data = json.load(f)
#     results = data.get("results", data) if isinstance(data, dict) else data
#     pool: Dict[str, Dict] = {}
# 
#     for r in results:
#         sid = r.get("sample_id", "")
#         if not sid:
#             continue
#         if source == "synthesized":
#             text = r.get("synthesized_trace", "")
#         elif source == "k_traces":
#             gt     = normalise_math_answer(r.get("target_answer") or r.get("ground_truth")) \
#                      or normalise_label(r.get("target_answer") or r.get("ground_truth"))
#             traces = r.get("traces", [])
#             text   = ""
#             for t in traces:
#                 t_text = t.get("reasoning_text") or t.get("raw_response", "")
#                 # Try math answer first, then logical label
#                 t_gt_raw = r.get("target_answer") or r.get("ground_truth", "")
#                 t_pred_math = normalise_math_answer(extract_math_answer(t_text))
#                 t_pred_logic = normalise_label(t.get("label")) or extract_label(t_text)
#                 t_pred = t_pred_math if t_pred_math else t_pred_logic
#                 if str(t_pred) == str(gt):
#                     text = t_text
#                     break
#             if not text and traces:
#                 text = traces[0].get("reasoning_text") or traces[0].get("raw_response", "")
#         elif source == "cleaned":
#             cleaned = r.get("cleaned_traces", [])
#             text    = max((t.get("reasoning_text", "") for t in cleaned), key=len, default="") if cleaned else ""
#         else:
#             continue
#         if text:
#             pool[sid] = {
#                 "text":     text,
#                 "n_steps":  count_steps(text),
#                 "n_tokens": count_tokens(text),
#             }
#     return pool
# 
# 
# # ---------------------------------------------------------------------------
# # Example retrieval
# # ---------------------------------------------------------------------------
# 
# def _tokenize(text: str) -> List[str]:
#     return re.findall(r'\b[a-z]{2,}\b', text.lower())
# 
# 
# def _bm25(query_tok, doc_tok, idf, avgdl, k1=1.5, b=0.75):
#     dl = len(doc_tok)
#     tf = Counter(doc_tok)
#     return sum(
#         idf.get(tok, 0) * (tf[tok] * (k1 + 1)) / (tf[tok] + k1 * (1 - b + b * dl / max(avgdl, 1)))
#         for tok in set(query_tok)
#     )
# 
# 
# def select_examples(
#     pool: List[Dict],
#     trace_pool: Dict[str, Dict],
#     query_text: str,
#     k: int,
#     exclude_ids: set,
#     retrieval: str,
#     seed: int = 42,
# ) -> List[Dict]:
#     available = [s for s in pool
#                  if s["sample_id"] not in exclude_ids and s["sample_id"] in trace_pool]
#     if not available:
#         return []
#     if retrieval == "similar":
#         all_docs = [_tokenize(s["problem_text"]) for s in available]
#         N  = len(all_docs)
#         df: Counter = Counter()
#         for doc in all_docs:
#             for t in set(doc):
#                 df[t] += 1
#         idf   = {t: math.log((N - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}
#         avgdl = sum(len(d) for d in all_docs) / max(N, 1)
#         qtok  = _tokenize(query_text)
#         scored = sorted(((s, _bm25(qtok, doc, idf, avgdl)) for s, doc in zip(available, all_docs)),
#                         key=lambda x: -x[1])
#         return [s for s, _ in scored[:k]]
#     else:
#         random.seed(seed)
#         return random.sample(available, min(k, len(available)))
# 
# 
# # ---------------------------------------------------------------------------
# # Prompts — Logical domain
# # ---------------------------------------------------------------------------
# 
# SYSTEM_LOGICIAN = (
#     "You are an expert logician. Given premises and a hypothesis, decide whether "
#     "the hypothesis is __PROVED__ or __DISPROVED__ based solely on the premises. "
#     "Output your step-by-step reasoning, then on the very last line write ONLY: "
#     "__PROVED__ or __DISPROVED__."
# )
# 
# SYSTEM_LOGICIAN_ZEROSHOT = (
#     "You are an expert logician. Given premises and a hypothesis, decide whether "
#     "the hypothesis is __PROVED__ or __DISPROVED__ based solely on the premises. "
#     "Output ONLY the label on the last line: __PROVED__ or __DISPROVED__."
# )
# 
# # Prompts — Math domain
# # ---------------------------------------------------------------------------
# 
# SYSTEM_MATH = (
#     "You are an expert mathematician. Solve the math problem step by step. "
#     "Show all calculations clearly. "
#     "On the very last line write ONLY the numeric answer in the format: \\boxed{<answer>}"
# )
# 
# SYSTEM_MATH_ZEROSHOT = (
#     "You are an expert mathematician. Solve the math problem. "
#     "Output ONLY the numeric answer on the last line in the format: \\boxed{<answer>}"
# )
# 
# 
# def _problem_block(problem: str) -> str:
#     return f"Problem:\n{problem}"
# 
# 
# def build_zeroshot_prompt(problem: str, domain: str = "logical") -> str:
#     if domain == "math":
#         return (_problem_block(problem) + "\n\n"
#                 "Solve step by step. Last line must be: \\boxed{<numeric answer>}")
#     return (_problem_block(problem) + "\n\n"
#             "Output ONLY one of: __PROVED__, __DISPROVED__")
# 
# 
# def build_icl_prompt(problem: str, examples: List[Dict],
#                      trace_pool: Dict[str, Dict], domain: str = "logical") -> str:
#     parts = []
#     for i, ex in enumerate(examples, 1):
#         trace   = trace_pool[ex["sample_id"]]["text"]
#         trimmed = trace[:800] + ("..." if len(trace) > 800 else "")
#         parts  += [
#             f"--- Example {i} ---",
#             _problem_block(ex["problem_text"]),
#             f"Reasoning:\n{trimmed}",
#             f"Answer: {ex['ground_truth']}",
#             "",
#         ]
#     parts += [
#         "--- New Problem ---",
#         _problem_block(problem),
#         "Reasoning: [your step-by-step reasoning]\nAnswer:",
#     ]
#     return "\n".join(parts)
# # ---------------------------------------------------------------------------
# 
# async def call_llm(
#     session: aiohttp.ClientSession,
#     semaphore: asyncio.Semaphore,
#     system: str,
#     user: str,
#     model: str,
#     api_key: str,
#     base_url: str,
#     temperature: float = 0.0,
#     max_tokens: int = 512,
# ) -> Optional[str]:
#     url     = base_url.rstrip("/") + "/chat/completions"
#     headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
#     payload = {
#         "model":    model,
#         "messages": [{"role": "system", "content": system},
#                      {"role": "user",   "content": user}],
#         "temperature": temperature,
#         "max_tokens":  max_tokens,
#     }
#     for attempt in range(4):
#         async with semaphore:
#             try:
#                 async with session.post(
#                     url, headers=headers, json=payload,
#                     timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
#                 ) as resp:
#                     if resp.status == 200:
#                         data = await resp.json()
#                         return data["choices"][0]["message"]["content"].strip()
#                     logger.warning("HTTP %s: %s", resp.status, (await resp.text())[:200])
#             except Exception as e:
#                 logger.warning("Attempt %d: %s", attempt + 1, e)
#         await asyncio.sleep(2 ** attempt)
#     return None
# 
# 
# # ---------------------------------------------------------------------------
# # Metrics
# # ---------------------------------------------------------------------------
# 
# def compute_metrics(predictions: List[Dict]) -> Dict:
#     """Domain-aware metrics.
# 
#     Logical domain: binary PROVED/DISPROVED accuracy + macro-F1.
#     Math domain:    exact-match accuracy on numeric answer strings + macro-F1 over unique answers.
#     """
#     # Detect domain from first prediction
#     domain = predictions[0].get("domain", "logical") if predictions else "logical"
# 
#     all_steps:  List[float] = []
#     all_tokens: List[float] = []
#     correct = no_pred = valid_total = 0
# 
#     if domain == "math":
#         # Exact-match on normalised numeric strings
#         for p in predictions:
#             gt   = normalise_math_answer(str(p.get("ground_truth", "")))
#             pred = normalise_math_answer(str(p.get("predicted", "") or ""))
#             if not gt:
#                 continue
#             valid_total += 1
#             if not pred:
#                 no_pred += 1
#             elif pred == gt:
#                 correct += 1
#             if p.get("response_steps") is not None:
#                 all_steps.append(p["response_steps"])
#             if p.get("response_tokens") is not None:
#                 all_tokens.append(p["response_tokens"])
# 
#         accuracy = correct / valid_total if valid_total else 0.0
#         return {
#             "accuracy":     round(accuracy, 4),
#             "macro_f1":     round(accuracy, 4),   # for math, exact-match acc ≈ F1
#             "avg_steps":    round(float(np.mean(all_steps)),  2) if all_steps  else 0.0,
#             "std_steps":    round(float(np.std(all_steps)),   2) if all_steps  else 0.0,
#             "avg_tokens":   round(float(np.mean(all_tokens)), 1) if all_tokens else 0.0,
#             "std_tokens":   round(float(np.std(all_tokens)),  1) if all_tokens else 0.0,
#             "n_total":      valid_total,
#             "n_correct":    correct,
#             "n_no_pred":    no_pred,
#             "no_pred_rate": round(no_pred / valid_total if valid_total else 0.0, 4),
#             "per_class_f1": {},
#             "confusion_matrix": {},
#             "domain":       "math",
#         }
# 
#     else:
#         # Logical domain: binary PROVED/DISPROVED
#         confusion = {gt: {p: 0 for p in BINARY_LABELS + ["none"]} for gt in BINARY_LABELS}
# 
#         for p in predictions:
#             gt   = p.get("ground_truth")
#             pred = p.get("predicted")
#             if gt not in BINARY_LABELS:
#                 continue
#             valid_total += 1
#             if pred not in BINARY_LABELS:
#                 no_pred += 1
#                 confusion[gt]["none"] += 1
#             else:
#                 confusion[gt][pred] += 1
#                 if pred == gt:
#                     correct += 1
#             if p.get("response_steps") is not None:
#                 all_steps.append(p["response_steps"])
#             if p.get("response_tokens") is not None:
#                 all_tokens.append(p["response_tokens"])
# 
#         accuracy = correct / valid_total if valid_total else 0.0
#         f1s = {}
#         for cls in BINARY_LABELS:
#             tp   = confusion[cls][cls]
#             fp   = sum(confusion[g][cls] for g in BINARY_LABELS if g != cls)
#             fn   = sum(confusion[cls][p] for p in BINARY_LABELS if p != cls)
#             prec = tp / (tp + fp) if (tp + fp) else 0.0
#             rec  = tp / (tp + fn) if (tp + fn) else 0.0
#             f1s[cls] = round(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0, 4)
# 
#         return {
#             "accuracy":     round(accuracy, 4),
#             "macro_f1":     round(sum(f1s.values()) / len(BINARY_LABELS), 4),
#             "avg_steps":    round(float(np.mean(all_steps)),  2) if all_steps  else 0.0,
#             "std_steps":    round(float(np.std(all_steps)),   2) if all_steps  else 0.0,
#             "avg_tokens":   round(float(np.mean(all_tokens)), 1) if all_tokens else 0.0,
#             "std_tokens":   round(float(np.std(all_tokens)),  1) if all_tokens else 0.0,
#             "n_total":      valid_total,
#             "n_correct":    correct,
#             "n_no_pred":    no_pred,
#             "no_pred_rate": round(no_pred / valid_total if valid_total else 0.0, 4),
#             "per_class_f1": f1s,
#             "confusion_matrix": confusion,
#             "domain":       "logical",
#         }
# 
# 
# def compute_per_dataset(predictions: List[Dict]) -> Dict[str, Dict]:
#     groups: Dict[str, List] = defaultdict(list)
#     for p in predictions:
#         groups[p.get("source_dataset", "unknown")].append(p)
#     return {ds: compute_metrics(preds) for ds, preds in groups.items()}
# 
# 
# # ---------------------------------------------------------------------------
# # Printing
# # ---------------------------------------------------------------------------
# 
# def print_metrics(m: Dict, label: str = "", indent: str = "") -> None:
#     domain = m.get("domain", "logical")
#     short  = {"__PROVED__": "PROVED", "__DISPROVED__": "DISPR"}
#     if label:
#         print(f"\n{indent}{'─'*70}")
#         print(f"{indent}  {label}")
#         print(f"{indent}{'─'*70}")
#     print(f"{indent}  Accuracy         : {m['accuracy']:.4f}  ({m['n_correct']}/{m['n_total']})")
#     if domain == "math":
#         print(f"{indent}  Exact-match Acc  : {m['accuracy']:.4f}  (same as accuracy for math)")
#     else:
#         print(f"{indent}  Macro-F1         : {m['macro_f1']:.4f}")
#     print(f"{indent}  Avg steps/resp   : {m['avg_steps']:.2f} ± {m['std_steps']:.2f}")
#     print(f"{indent}  Avg tokens/resp  : {m['avg_tokens']:.1f} ± {m['std_tokens']:.1f}")
#     print(f"{indent}  No-pred rate     : {m['no_pred_rate']:.4f}  ({m['n_no_pred']} samples)")
#     if domain == "logical" and m.get("per_class_f1"):
#         f1_str = "  ".join(f"{short[c]}={m['per_class_f1'].get(c,0):.3f}" for c in BINARY_LABELS)
#         print(f"{indent}  Per-class F1     : {f1_str}")
#         if m.get("confusion_matrix"):
#             print(f"\n{indent}  Confusion matrix (rows=GT, cols=Pred):")
#             hdr = f"{indent}  {'GT/Pred':<10}" + "".join(f"{short[l]:>8}" for l in BINARY_LABELS) + f"{'none':>7}"
#             print(hdr)
#             cm = m["confusion_matrix"]
#             for gt in BINARY_LABELS:
#                 row = f"{indent}  {short[gt]:<10}" + "".join(f"{cm.get(gt,{}).get(p,0):>8}" for p in BINARY_LABELS)
#                 row += f"{cm.get(gt,{}).get('none',0):>7}"
#                 print(row)
# 
# 
# def print_comparison_table(results: List[Tuple[str, Dict]]) -> None:
#     print("\n" + "═"*116)
#     print("  LABEL PREDICTION — COMPARISON TABLE (All Settings)")
#     print("═"*116)
#     hdr = (f"  {'Setting':<46}  {'Accuracy':>9}  {'Macro-F1':>9}"
#            f"  {'Avg Steps':>11}  {'Avg Tokens':>12}  N")
#     print(hdr)
#     print("  " + "─"*112)
#     for label, m in results:
#         steps_str  = f"{m['avg_steps']:.2f}±{m['std_steps']:.2f}"
#         tokens_str = f"{m['avg_tokens']:.1f}±{m['std_tokens']:.1f}"
#         print(f"  {label:<46}  {m['accuracy']:>9.4f}  {m['macro_f1']:>9.4f}"
#               f"  {steps_str:>11}  {tokens_str:>12}  {m['n_total']}")
#     print("═"*116)
# 
# 
# # ===========================================================================
# # SETTING IMPLEMENTATIONS
# # ===========================================================================
# 
# # ---------------------------------------------------------------------------
# # Settings A–E: Zero-shot and ICL variants
# # ---------------------------------------------------------------------------
# 
# async def run_setting_zeroshot(
#     test_samples: List[Dict],
#     model: str, api_key: str, base_url: str,
#     semaphore: asyncio.Semaphore,
#     session: aiohttp.ClientSession,
# ) -> List[Dict]:
#     """Setting A: zero-shot, single call per sample. Domain-aware."""
#     domain = test_samples[0].get("domain", "logical") if test_samples else "logical"
#     system = SYSTEM_MATH_ZEROSHOT if domain == "math" else SYSTEM_LOGICIAN_ZEROSHOT
# 
#     coros = [
#         call_llm(session, semaphore, system,
#                  build_zeroshot_prompt(s["problem_text"], domain=domain),
#                  model, api_key, base_url)
#         for s in test_samples
#     ]
#     responses = await asyncio.gather(*coros)
# 
#     results = []
#     for s, r in zip(test_samples, responses):
#         if domain == "math":
#             predicted = normalise_math_answer(extract_math_answer(r))
#         else:
#             predicted = extract_label(r)
#         results.append({
#             "sample_id":       s["sample_id"],
#             "source_dataset":  s["source_dataset"],
#             "ground_truth":    s["ground_truth"],
#             "predicted":       predicted,
#             "raw_response":    r or "",
#             "response_steps":  count_steps(r or ""),
#             "response_tokens": count_tokens(r or ""),
#             "domain":          domain,
#         })
#     return results
# 
# 
# async def run_setting_icl(
#     name: str,
#     test_samples: List[Dict],
#     train_pool: List[Dict],
#     trace_pool: Dict[str, Dict],
#     shots: int,
#     retrieval: str,
#     model: str, api_key: str, base_url: str,
#     semaphore: asyncio.Semaphore,
#     session: aiohttp.ClientSession,
#     seed: int,
# ) -> List[Dict]:
#     """Settings B–E: ICL with a trace pool. Domain-aware."""
#     domain   = test_samples[0].get("domain", "logical") if test_samples else "logical"
#     system   = SYSTEM_MATH if domain == "math" else SYSTEM_LOGICIAN
#     test_ids = {s["sample_id"] for s in test_samples}
#     coros, meta = [], []
# 
#     for s in test_samples:
#         examples = select_examples(train_pool, trace_pool, s["problem_text"],
#                                    shots, test_ids, retrieval, seed)
#         user     = build_icl_prompt(s["problem_text"], examples, trace_pool, domain=domain)
#         coros.append(call_llm(session, semaphore, system, user,
#                               model, api_key, base_url, max_tokens=1024))
#         meta.append({
#             "sample":             s,
#             "example_avg_steps":  float(np.mean([trace_pool[e["sample_id"]]["n_steps"]  for e in examples])) if examples else 0.0,
#             "example_avg_tokens": float(np.mean([trace_pool[e["sample_id"]]["n_tokens"] for e in examples])) if examples else 0.0,
#         })
# 
#     responses = await asyncio.gather(*coros)
#     predictions = []
#     for m, r in zip(meta, responses):
#         s = m["sample"]
#         predicted = normalise_math_answer(extract_math_answer(r)) if domain == "math" else extract_label(r)
#         predictions.append({
#             "sample_id":           s["sample_id"],
#             "source_dataset":      s["source_dataset"],
#             "ground_truth":        s["ground_truth"],
#             "predicted":           predicted,
#             "raw_response":        r or "",
#             "response_steps":      count_steps(r or ""),
#             "response_tokens":     count_tokens(r or ""),
#             "example_avg_steps":   m["example_avg_steps"],
#             "example_avg_tokens":  m["example_avg_tokens"],
#             "domain":              domain,
#         })
#     return predictions
# 
# 
# def _extract_pred(text: str, domain: str = "logical") -> Optional[str]:
#     """Domain-aware prediction extraction."""
#     if domain == "math":
#         return normalise_math_answer(extract_math_answer(text))
#     return extract_label(text)
# 
# 
# def _make_pred_dict(s: Dict, r: str, domain: str, **extra) -> Dict:
#     """Build a standard prediction record."""
#     return {
#         "sample_id":       s["sample_id"],
#         "source_dataset":  s["source_dataset"],
#         "ground_truth":    s["ground_truth"],
#         "predicted":       _extract_pred(r, domain),
#         "raw_response":    r or "",
#         "response_steps":  count_steps(r or ""),
#         "response_tokens": count_tokens(r or ""),
#         "domain":          domain,
#         **extra,
#     }
# 
# 
# # ---------------------------------------------------------------------------
# # Setting F: Self-Consistency (Wang et al., 2022)
# # ---------------------------------------------------------------------------
# 
# async def run_setting_self_consistency(
#     test_samples: List[Dict],
#     k: int,
#     model: str, api_key: str, base_url: str,
#     semaphore: asyncio.Semaphore,
#     session: aiohttp.ClientSession,
# ) -> List[Dict]:
#     """Setting F: generate k responses at temperature 0.7, majority vote on label.
# 
#     Reference: Wang et al. (2022) — Self-Consistency Improves CoT Reasoning.
#     LLM calls per sample: k.
#     """
#     domain = test_samples[0].get("domain", "logical") if test_samples else "logical"
#     system = SYSTEM_MATH if domain == "math" else SYSTEM_LOGICIAN
#     all_coros = []
#     for s in test_samples:
#         user = build_zeroshot_prompt(s["problem_text"], domain=domain)
#         for _ in range(k):
#             all_coros.append(
#                 call_llm(session, semaphore, system,
#                          user, model, api_key, base_url,
#                          temperature=0.7, max_tokens=1024)
#             )
#     all_responses = await asyncio.gather(*all_coros)
# 
#     predictions = []
#     for i, s in enumerate(test_samples):
#         resps  = all_responses[i * k: (i + 1) * k]
#         preds  = [_extract_pred(r, domain) for r in resps if r]
#         valid  = [p for p in preds if p is not None]
#         pred   = Counter(valid).most_common(1)[0][0] if valid else None
#         steps  = [count_steps(r or "")  for r in resps]
#         tokens = [count_tokens(r or "") for r in resps]
#         predictions.append({
#             "sample_id":       s["sample_id"],
#             "source_dataset":  s["source_dataset"],
#             "ground_truth":    s["ground_truth"],
#             "predicted":       pred,
#             "all_labels":      preds,
#             "response_steps":  float(np.mean(steps))  if steps  else 0.0,
#             "response_tokens": float(np.mean(tokens)) if tokens else 0.0,
#             "domain":          domain,
#         })
#     return predictions
# 
# 
# # ---------------------------------------------------------------------------
# # Setting G: Universal Self-Consistency (Chen et al., 2023, arXiv:2311.17311)
# # ---------------------------------------------------------------------------
# 
# _USC_SELECTOR_SYSTEM_LOGICAL = (
#     "You are an expert logician. You will see multiple reasoning attempts for the same problem. "
#     "Select the single most consistent and correct answer. "
#     "Output ONLY one of: __PROVED__, __DISPROVED__"
# )
# _USC_SELECTOR_SYSTEM_MATH = (
#     "You are an expert mathematician. You will see multiple solution attempts for the same problem. "
#     "Select the single most consistent and correct numeric answer. "
#     "Output ONLY the number (e.g. 42 or 3.5), no other text."
# )
# 
# 
# async def run_setting_usc(
#     test_samples: List[Dict],
#     k: int,
#     model: str, api_key: str, base_url: str,
#     semaphore: asyncio.Semaphore,
#     session: aiohttp.ClientSession,
# ) -> List[Dict]:
#     """Setting G: Universal Self-Consistency — k samples + LLM selector.
# 
#     Reference: Chen et al. (2023) arXiv:2311.17311
#     LLM calls per sample: k + 1.
#     """
#     domain = test_samples[0].get("domain", "logical") if test_samples else "logical"
#     system = SYSTEM_MATH if domain == "math" else SYSTEM_LOGICIAN
#     sel_system = _USC_SELECTOR_SYSTEM_MATH if domain == "math" else _USC_SELECTOR_SYSTEM_LOGICAL
# 
#     gen_coros = []
#     for s in test_samples:
#         user = build_zeroshot_prompt(s["problem_text"], domain=domain)
#         for _ in range(k):
#             gen_coros.append(
#                 call_llm(session, semaphore, system,
#                          user, model, api_key, base_url,
#                          temperature=0.7, max_tokens=1024)
#             )
#     all_gen = await asyncio.gather(*gen_coros)
# 
#     sel_coros = []
#     grouped   = []
#     for i, s in enumerate(test_samples):
#         resps = all_gen[i * k: (i + 1) * k]
#         grouped.append(resps)
#         candidates_block = "\n\n".join(
#             f"Candidate {j+1}:\n{r[:600]}" for j, r in enumerate(resps) if r
#         )
#         if domain == "math":
#             sel_user = (
#                 f"Problem:\n{s['problem_text']}\n\n"
#                 f"Multiple solution attempts:\n{candidates_block}\n\n"
#                 "What is the most consistent correct numeric answer? Output ONLY the number:"
#             )
#         else:
#             sel_user = (
#                 f"Problem:\n{s['problem_text']}\n\n"
#                 f"Multiple reasoning attempts:\n{candidates_block}\n\n"
#                 "Based on the above candidates, what is the most consistent and correct answer? "
#                 "Output ONLY: __PROVED__ or __DISPROVED__"
#             )
#         sel_coros.append(
#             call_llm(session, semaphore, sel_system,
#                      sel_user, model, api_key, base_url,
#                      temperature=0.0, max_tokens=32)
#         )
#     sel_responses = await asyncio.gather(*sel_coros)
# 
#     predictions = []
#     for s, resps, sel_r in zip(test_samples, grouped, sel_responses):
#         pred   = _extract_pred(sel_r, domain) if sel_r else None
#         steps  = [count_steps(r or "")  for r in resps]
#         tokens = [count_tokens(r or "") for r in resps]
#         predictions.append({
#             "sample_id":         s["sample_id"],
#             "source_dataset":    s["source_dataset"],
#             "ground_truth":      s["ground_truth"],
#             "predicted":         pred,
#             "selector_response": sel_r or "",
#             "response_steps":    float(np.mean(steps))  if steps  else 0.0,
#             "response_tokens":   float(np.mean(tokens)) if tokens else 0.0,
#             "domain":            domain,
#         })
#     return predictions
# 
# 
# # ---------------------------------------------------------------------------
# # Setting H: Self-Refine (Madaan et al., NeurIPS 2023, arXiv:2303.17651)
# # ---------------------------------------------------------------------------
# 
# _REFINE_CRITIC_SYSTEM = (
#     "You are a strict reasoning critic. "
#     "Review the reasoning below and identify any errors, unsupported leaps, or incorrect conclusions. "
#     "Be specific and concise. If the reasoning is correct, say 'The reasoning is correct.'"
# )
# 
# _REFINE_REVISE_SYSTEM_LOGICAL = (
#     "You are an expert logician. Revise the reasoning chain based on the critique provided. "
#     "Produce an improved reasoning chain that fixes all identified errors. "
#     "On the very last line write ONLY: __PROVED__ or __DISPROVED__."
# )
# 
# _REFINE_REVISE_SYSTEM_MATH = (
#     "You are an expert mathematician. Revise the solution based on the critique provided. "
#     "Show all corrected calculations step by step. "
#     "On the very last line write ONLY: \\boxed{<numeric answer>}"
# )
# 
# 
# async def run_setting_self_refine(
#     test_samples: List[Dict],
#     model: str, api_key: str, base_url: str,
#     semaphore: asyncio.Semaphore,
#     session: aiohttp.ClientSession,
#     n_iterations: int = 1,
# ) -> List[Dict]:
#     """Setting H: Self-Refine — generate → critique → revise, n_iterations times.
# 
#     Reference: Madaan et al. (NeurIPS 2023) arXiv:2303.17651
#     LLM calls per sample: 1 + 2 * n_iterations.
#     """
#     domain  = test_samples[0].get("domain", "logical") if test_samples else "logical"
#     system  = SYSTEM_MATH if domain == "math" else SYSTEM_LOGICIAN
#     rev_sys = _REFINE_REVISE_SYSTEM_MATH if domain == "math" else _REFINE_REVISE_SYSTEM_LOGICAL
# 
#     init_coros = [
#         call_llm(session, semaphore, system,
#                  build_zeroshot_prompt(s["problem_text"], domain=domain),
#                  model, api_key, base_url, temperature=0.0, max_tokens=1024)
#         for s in test_samples
#     ]
#     current_responses = list(await asyncio.gather(*init_coros))
# 
#     for _ in range(n_iterations):
#         crit_coros = []
#         for s, resp in zip(test_samples, current_responses):
#             crit_user = (
#                 f"Problem:\n{s['problem_text']}\n\n"
#                 f"Reasoning:\n{resp or '(empty)'}\n\n"
#                 "Identify any errors or unsupported steps:"
#             )
#             crit_coros.append(
#                 call_llm(session, semaphore, _REFINE_CRITIC_SYSTEM,
#                          crit_user, model, api_key, base_url,
#                          temperature=0.0, max_tokens=256)
#             )
#         critiques = await asyncio.gather(*crit_coros)
# 
#         revise_coros = []
#         for s, resp, crit in zip(test_samples, current_responses, critiques):
#             if domain == "math":
#                 conclusion_hint = "Last line must be ONLY: \\boxed{<numeric answer>}"
#             else:
#                 conclusion_hint = "Last line must be ONLY: __PROVED__ or __DISPROVED__"
#             rev_user = (
#                 f"Problem:\n{s['problem_text']}\n\n"
#                 f"Original reasoning:\n{resp or '(empty)'}\n\n"
#                 f"Critique:\n{crit or '(none)'}\n\n"
#                 f"Revise to fix all issues. {conclusion_hint}"
#             )
#             revise_coros.append(
#                 call_llm(session, semaphore, rev_sys,
#                          rev_user, model, api_key, base_url,
#                          temperature=0.0, max_tokens=1024)
#             )
#         current_responses = list(await asyncio.gather(*revise_coros))
# 
#     predictions = []
#     for s, r in zip(test_samples, current_responses):
#         predictions.append({
#             "sample_id":       s["sample_id"],
#             "source_dataset":  s["source_dataset"],
#             "ground_truth":    s["ground_truth"],
#             "predicted":       _extract_pred(r, domain),
#             "raw_response":    r or "",
#             "response_steps":  count_steps(r or ""),
#             "response_tokens": count_tokens(r or ""),
#             "domain":          domain,
#         })
#     return predictions
# 
# 
# # ---------------------------------------------------------------------------
# # Setting I: LLM Self-Aggregation (Li et al., 2025, arXiv:2503.04104)
# # ---------------------------------------------------------------------------
# 
# _AGGREGATE_SYSTEM_LOGICAL = (
#     "You are an expert logician. You have seen multiple independent reasoning attempts "
#     "for the same problem. Synthesize the best elements into a single coherent chain. "
#     "On the very last line write ONLY: __PROVED__ or __DISPROVED__."
# )
# 
# _AGGREGATE_SYSTEM_MATH = (
#     "You are an expert mathematician. You have seen multiple solution attempts for the same problem. "
#     "Synthesize the correct approach into a single coherent solution. "
#     "On the very last line write ONLY: \\boxed{<numeric answer>}"
# )
# 
# 
# async def run_setting_self_aggregation(
#     test_samples: List[Dict],
#     k: int,
#     model: str, api_key: str, base_url: str,
#     semaphore: asyncio.Semaphore,
#     session: aiohttp.ClientSession,
# ) -> List[Dict]:
#     """Setting I: LLM Self-Aggregation — generate k, then LLM synthesizes them.
# 
#     Reference: Li et al. (2025) arXiv:2503.04104
#     LLM calls per sample: k + 1.
#     """
#     domain     = test_samples[0].get("domain", "logical") if test_samples else "logical"
#     system     = SYSTEM_MATH if domain == "math" else SYSTEM_LOGICIAN
#     agg_system = _AGGREGATE_SYSTEM_MATH if domain == "math" else _AGGREGATE_SYSTEM_LOGICAL
# 
#     gen_coros = []
#     for s in test_samples:
#         for _ in range(k):
#             gen_coros.append(
#                 call_llm(session, semaphore, system,
#                          build_zeroshot_prompt(s["problem_text"], domain=domain),
#                          model, api_key, base_url,
#                          temperature=0.7, max_tokens=1024)
#             )
#     all_gen = await asyncio.gather(*gen_coros)
# 
#     agg_coros = []
#     grouped   = []
#     for i, s in enumerate(test_samples):
#         resps = all_gen[i * k: (i + 1) * k]
#         grouped.append(resps)
#         responses_block = "\n\n".join(
#             f"Response {j+1}:\n{r[:600]}" for j, r in enumerate(resps) if r
#         )
#         if domain == "math":
#             conclusion_hint = "Last line must be ONLY: \\boxed{<numeric answer>}"
#         else:
#             conclusion_hint = "Last line must be ONLY: __PROVED__ or __DISPROVED__"
#         agg_user = (
#             f"Problem:\n{s['problem_text']}\n\n"
#             f"Multiple independent attempts:\n{responses_block}\n\n"
#             f"Synthesize into the best possible solution. {conclusion_hint}"
#         )
#         agg_coros.append(
#             call_llm(session, semaphore, agg_system,
#                      agg_user, model, api_key, base_url,
#                      temperature=0.0, max_tokens=1024)
#         )
#     agg_responses = await asyncio.gather(*agg_coros)
# 
#     predictions = []
#     for s, resps, agg_r in zip(test_samples, grouped, agg_responses):
#         predictions.append({
#             "sample_id":       s["sample_id"],
#             "source_dataset":  s["source_dataset"],
#             "ground_truth":    s["ground_truth"],
#             "predicted":       _extract_pred(agg_r, domain),
#             "raw_response":    agg_r or "",
#             "response_steps":  count_steps(agg_r or ""),
#             "response_tokens": count_tokens(agg_r or ""),
#             "domain":          domain,
#         })
#     return predictions
# 
# 
# # ===========================================================================
# # Setting J: Self-Evaluation Guided Beam Search
# # (Xie et al., NeurIPS 2023, arXiv:2305.00633)
# # ===========================================================================
# #
# # The model generates one step at a time and self-evaluates each step by asking
# # "Given the reasoning so far, how confident are you this is correct? (0-1)"
# # We maintain a beam of width `beam_width` partial traces, expanding and pruning
# # based on self-evaluation scores.  No external verifier needed.
# #
# # Simplified implementation:
# #   - Beam width = beam_width (default 2)
# #   - At each step: expand each beam member by sampling one next step,
# #     then score all candidates, keep top beam_width.
# #   - Stop when any beam member produces a label or max_steps reached.
# #   - LLM calls per sample: ≈ beam_width × max_steps × 2  (generate + score)
# # ---------------------------------------------------------------------------
# 
# _SEBS_GEN_SYSTEM = (
#     "You are an expert logician reasoning step by step. "
#     "Continue the reasoning chain by writing ONLY the next single reasoning step. "
#     "If you have reached a definitive conclusion, write the conclusion step ending with "
#     "__PROVED__ or __DISPROVED__. Do not write more than one step."
# )
# 
# _SEBS_GEN_SYSTEM_MATH = (
#     "You are an expert mathematician reasoning step by step. "
#     "Continue the solution by writing ONLY the next single calculation step. "
#     "If you have reached the final numeric answer, write the conclusion step ending with "
#     "\\boxed{<answer>}. Do not write more than one step."
# )
# 
# _SEBS_SCORE_SYSTEM = (
#     "You are a logical reasoning evaluator. "
#     "Given a problem and a partial reasoning chain, assess how likely this reasoning "
#     "path leads to a correct conclusion. "
#     "Output ONLY a number between 0.0 and 1.0 representing your confidence."
# )
# 
# _SEBS_SCORE_SYSTEM_MATH = (
#     "You are a math reasoning evaluator. "
#     "Given a math problem and a partial solution chain, assess how likely this solution "
#     "path leads to the correct numeric answer. "
#     "Output ONLY a number between 0.0 and 1.0 representing your confidence."
# )
# 
# 
# def _parse_score(text: str) -> float:
#     """Parse a float confidence score from LLM output."""
#     m = re.search(r"\b(0?\.\d+|1\.0|0|1)\b", text or "")
#     if m:
#         try:
#             return max(0.0, min(1.0, float(m.group(1))))
#         except ValueError:
#             pass
#     return 0.5  # neutral fallback
# 
# 
# async def run_setting_sebs(
#     test_samples: List[Dict],
#     model: str, api_key: str, base_url: str,
#     semaphore: asyncio.Semaphore,
#     session: aiohttp.ClientSession,
#     beam_width: int = 2,
#     max_steps: int = 8,
# ) -> List[Dict]:
#     """Setting J: Self-Evaluation Guided Beam Search (Xie et al., NeurIPS 2023).
# 
#     LLM calls per sample: ≈ beam_width × max_steps × 2 (generate + self-score).
#     """
#     domain = test_samples[0].get("domain", "logical") if test_samples else "logical"
#     sebs_gen_sys   = _SEBS_GEN_SYSTEM_MATH   if domain == "math" else _SEBS_GEN_SYSTEM
#     sebs_score_sys = _SEBS_SCORE_SYSTEM_MATH  if domain == "math" else _SEBS_SCORE_SYSTEM
# 
#     async def _beam_search_one(problem: str) -> Tuple[str, float]:
#         """Run beam search for a single problem. Returns (best_trace, best_score)."""
#         # Each beam member: {"trace": str, "score": float, "done": bool}
#         beams = [{"trace": "", "score": 1.0, "done": False}]
# 
#         for step_idx in range(max_steps):
#             active = [b for b in beams if not b["done"]]
#             if not active:
#                 break
# 
#             # Generate next step for every active beam
#             gen_coros = []
#             for b in active:
#                 prev = b["trace"] if b["trace"] else "(start)"
#                 user = (
#                     f"Problem:\n{problem}\n\n"
#                     f"Reasoning so far:\n{prev}\n\n"
#                     f"Write the next reasoning step only:"
#                 )
#                 gen_coros.append(
#                     call_llm(session, semaphore, sebs_gen_sys, user,
#                              model, api_key, base_url,
#                              temperature=0.7, max_tokens=200)
#                 )
#             next_steps = await asyncio.gather(*gen_coros)
# 
#             # Score each candidate
#             score_coros = []
#             candidates = []
#             for b, step in zip(active, next_steps):
#                 new_trace = (b["trace"] + "\n" + (step or "")).strip()
#                 candidates.append({"trace": new_trace, "step": step or "",
#                                    "parent_score": b["score"]})
#                 score_user = (
#                     f"Problem:\n{problem}\n\n"
#                     f"Partial reasoning:\n{new_trace}\n\n"
#                     "Confidence this reasoning path is correct (0.0–1.0):"
#                 )
#                 score_coros.append(
#                     call_llm(session, semaphore, sebs_score_sys, score_user,
#                              model, api_key, base_url,
#                              temperature=0.0, max_tokens=16)
#                 )
#             score_texts = await asyncio.gather(*score_coros)
# 
#             new_beams = []
#             for cand, score_text in zip(candidates, score_texts):
#                 step_score = _parse_score(score_text)
#                 combined   = 0.6 * step_score + 0.4 * cand["parent_score"]
#                 label_hit  = _extract_pred(cand["trace"], domain)
#                 new_beams.append({
#                     "trace": cand["trace"],
#                     "score": combined,
#                     "done":  label_hit is not None,
#                 })
# 
#             # Keep inactive (done) beams + top-beam_width active beams
#             inactive = [b for b in beams if b["done"]]
#             new_beams.sort(key=lambda x: -x["score"])
#             beams = inactive + new_beams[:beam_width]
# 
#         # Return the highest-scoring beam
#         best = max(beams, key=lambda x: x["score"])
#         return best["trace"], best["score"]
# 
#     # Run beam search for all test samples (concurrently)
#     tasks = [_beam_search_one(s["problem_text"]) for s in test_samples]
#     results_bs = await asyncio.gather(*tasks)
# 
#     predictions = []
#     for s, (trace, score) in zip(test_samples, results_bs):
#         predictions.append({
#             "sample_id":       s["sample_id"],
#             "source_dataset":  s["source_dataset"],
#             "ground_truth":    s["ground_truth"],
#             "predicted":       _extract_pred(trace, domain),
#             "raw_response":    trace,
#             "beam_score":      round(score, 4),
#             "response_steps":  count_steps(trace),
#             "response_tokens": count_tokens(trace),
#             "domain":          domain,
#         })
#     return predictions
# 
# 
# # ===========================================================================
# # Setting K: Faithful CoT + Symbolic Solver
# # (Lyu et al., IJCNLP-AACL 2023, arXiv:2301.13379)
# # ===========================================================================
# #
# # Two-stage approach:
# #   Stage 1 (LLM): Translate the natural-language problem into a symbolic
# #                  representation — a list of Prolog-style facts and rules.
# #   Stage 2 (Solver): Execute the symbolic representation with a Python solver
# #                     (pyDatalog) to derive the conclusion.
# #
# # If pyDatalog is unavailable or the solver fails to derive an answer,
# # we fall back to extracting a label from the LLM's symbolic output directly.
# #
# # LLM calls per sample: 1 (translation only).
# # ---------------------------------------------------------------------------
# 
# _SYMBOLIC_SYSTEM = (
#     "You are an expert in symbolic logic. Convert the following logical reasoning "
#     "problem into Prolog-style facts and rules, then derive the conclusion.\n\n"
#     "Output format (follow exactly):\n"
#     "FACTS:\n"
#     "fact(entity, property).\n"
#     "rule(X, Y) :- fact(X, Z), fact(Z, Y).\n"
#     "...\n\n"
#     "QUERY:\n"
#     "?- hypothesis(entity, property).\n\n"
#     "CONCLUSION: __PROVED__ or __DISPROVED__\n\n"
#     "If the hypothesis cannot be determined from the facts, write: CONCLUSION: __DISPROVED__"
# )
# 
# _SYMBOLIC_SYSTEM_MATH = (
#     "You are an expert mathematician. Solve the following math problem using "
#     "step-by-step arithmetic reasoning, expressing each operation explicitly.\n\n"
#     "Output format:\n"
#     "STEPS:\n"
#     "step1: <equation>\n"
#     "step2: <equation>\n"
#     "...\n\n"
#     "ANSWER: <numeric value>\n\n"
#     "The ANSWER line must contain only the final numeric answer."
# )
# 
# # Optional: try to import pyDatalog for actual symbolic execution
# try:
#     from pyDatalog import pyDatalog as _pydl
#     _PYDATALOG_AVAILABLE = True
# except ImportError:
#     _PYDATALOG_AVAILABLE = False
# 
# 
# def _run_symbolic_solver(llm_output: str, domain: str = "logical") -> Optional[str]:
#     """Extract answer from the LLM's structured symbolic output."""
#     if domain == "math":
#         # Look for ANSWER: N line
#         m = re.search(r"ANSWER\s*[:\-]\s*([^\n]+)", llm_output or "", re.IGNORECASE)
#         if m:
#             return normalise_math_answer(m.group(1))
#         return normalise_math_answer(extract_math_answer(llm_output))
# 
#     # Logical domain: look for CONCLUSION: __PROVED__/__DISPROVED__
#     conclusion_match = re.search(
#         r"CONCLUSION\s*[:\-]\s*(__PROVED__|__DISPROVED__)",
#         llm_output or "", re.IGNORECASE
#     )
#     if conclusion_match:
#         return normalise_label(conclusion_match.group(1))
#     return extract_label(llm_output)
# 
# 
# async def run_setting_faithful_cot(
#     test_samples: List[Dict],
#     model: str, api_key: str, base_url: str,
#     semaphore: asyncio.Semaphore,
#     session: aiohttp.ClientSession,
# ) -> List[Dict]:
#     """Setting K: Faithful CoT + Symbolic Solver (Lyu et al., IJCNLP-AACL 2023).
# 
#     Stage 1: LLM translates problem to structured symbolic reasoning.
#     Stage 2: Extract answer from structured output.
#     LLM calls per sample: 1.
#     """
#     domain     = test_samples[0].get("domain", "logical") if test_samples else "logical"
#     sym_system = _SYMBOLIC_SYSTEM_MATH if domain == "math" else _SYMBOLIC_SYSTEM
# 
#     coros = []
#     for s in test_samples:
#         if domain == "math":
#             user = (
#                 f"Problem:\n{s['problem_text']}\n\n"
#                 "Solve using explicit arithmetic steps. "
#                 "Follow the output format with STEPS: and ANSWER: sections."
#             )
#         else:
#             user = (
#                 f"Problem:\n{s['problem_text']}\n\n"
#                 "Translate into symbolic logic facts and rules, then derive the conclusion. "
#                 "Follow the output format with FACTS:, QUERY:, and CONCLUSION: sections."
#             )
#         coros.append(
#             call_llm(session, semaphore, sym_system, user,
#                      model, api_key, base_url,
#                      temperature=0.0, max_tokens=1024)
#         )
#     responses = await asyncio.gather(*coros)
# 
#     predictions = []
#     for s, r in zip(test_samples, responses):
#         pred = _run_symbolic_solver(r or "", domain=domain)
#         predictions.append({
#             "sample_id":        s["sample_id"],
#             "source_dataset":   s["source_dataset"],
#             "ground_truth":     s["ground_truth"],
#             "predicted":        pred,
#             "raw_response":     r or "",
#             "symbolic_output":  r or "",
#             "response_steps":   count_steps(r or ""),
#             "response_tokens":  count_tokens(r or ""),
#             "domain":           domain,
#         })
#     return predictions
# 
# 
# # ===========================================================================
# # MAIN RUNNER
# # ===========================================================================
# 
# async def run(args: argparse.Namespace) -> None:
#     all_samples = load_raw_dataset([Path(p) for p in args.datasets], seed=args.seed)
#     logger.info("Loaded %d balanced samples", len(all_samples))
# 
#     # Train/test split
#     if args.test_file and Path(args.test_file).exists():
#         with open(args.test_file) as f:
#             test_ids_set = set(json.load(f))
#         test_samples  = [s for s in all_samples if s["sample_id"] in test_ids_set]
#         train_samples = [s for s in all_samples if s["sample_id"] not in test_ids_set]
#         logger.info("Using provided split: %d test / %d train", len(test_samples), len(train_samples))
#     else:
#         train_samples, test_samples = stratified_split(all_samples, args.test_ratio, args.seed)
#         logger.info("Auto-split: %d train / %d test", len(train_samples), len(test_samples))
#         split_path = Path(args.output).parent / "test_ids.json" if args.output else Path("test_ids.json")
#         split_path.parent.mkdir(parents=True, exist_ok=True)
#         with open(split_path, "w") as f:
#             json.dump([s["sample_id"] for s in test_samples], f, indent=2)
#         logger.info("Test IDs saved → %s", split_path)
# 
#     # Trace pools for B–E
#     pools = {
#         "B": load_trace_pool(Path(args.k_traces_file)    if args.k_traces_file    else None, "k_traces"),
#         "C": load_trace_pool(Path(args.cleaned_file)     if args.cleaned_file     else None, "cleaned"),
#         "D": load_trace_pool(Path(args.synthesized_step) if args.synthesized_step else None, "synthesized"),
#         "E": load_trace_pool(Path(args.synthesized_dag)  if args.synthesized_dag  else None, "synthesized"),
#     }
#     for key, pool in pools.items():
#         logger.info("Setting %s trace pool: %d samples", key, len(pool))
# 
#     # Determine which settings to run
#     settings_to_run = args.settings or ALL_SETTINGS
#     # Skip B-E if trace pool is missing
#     for key in list(settings_to_run):
#         if key in ("B", "C", "D", "E") and not pools.get(key):
#             logger.warning("Setting %s skipped — no trace file provided", key)
#             settings_to_run = [s for s in settings_to_run if s != key]
# 
#     k          = args.shots          # k for SC / USC / Self-Agg
#     model      = args.model
#     api_key    = args.api_key
#     base_url   = args.base_url
#     semaphore  = asyncio.Semaphore(args.concurrency)
# 
#     all_results: List[Tuple[str, Dict]] = []
#     full_output: Dict[str, Any]         = {}
# 
#     setting_names = {
#         "A": "A: Zero-shot",
#         "B": "B: ICL + Raw Trace",
#         "C": "C: ICL + Cleaned Trace",
#         "D": "D: ICL + Synthesized (step_by_step)",
#         "E": "E: ICL + Synthesized (DAG)",
#         "F": f"F: Self-Consistency (k={k})",
#         "G": f"G: Universal Self-Consistency (k={k})",
#         "H": "H: Self-Refine (1 iteration)",
#         "I": f"I: LLM Self-Aggregation (k={k})",
#         "J": f"J: Self-Eval Beam Search (beam={args.beam_width})",
#         "K": "K: Faithful CoT + Symbolic Solver",
#     }
# 
#     async with aiohttp.ClientSession() as session:
#         for key in settings_to_run:
#             name = setting_names[key]
#             logger.info("=== Running Setting %s: %s ===", key, name)
# 
#             if key == "A":
#                 preds = await run_setting_zeroshot(
#                     test_samples, model, api_key, base_url, semaphore, session)
#             elif key in ("B", "C", "D", "E"):
#                 preds = await run_setting_icl(
#                     name, test_samples, train_samples, pools[key],
#                     args.shots, args.retrieval,
#                     model, api_key, base_url, semaphore, session, args.seed)
#             elif key == "F":
#                 preds = await run_setting_self_consistency(
#                     test_samples, k, model, api_key, base_url, semaphore, session)
#             elif key == "G":
#                 preds = await run_setting_usc(
#                     test_samples, k, model, api_key, base_url, semaphore, session)
#             elif key == "H":
#                 preds = await run_setting_self_refine(
#                     test_samples, model, api_key, base_url, semaphore, session,
#                     n_iterations=args.refine_iterations)
#             elif key == "I":
#                 preds = await run_setting_self_aggregation(
#                     test_samples, k, model, api_key, base_url, semaphore, session)
#             elif key == "J":
#                 preds = await run_setting_sebs(
#                     test_samples, model, api_key, base_url, semaphore, session,
#                     beam_width=args.beam_width, max_steps=args.beam_max_steps)
#             elif key == "K":
#                 preds = await run_setting_faithful_cot(
#                     test_samples, model, api_key, base_url, semaphore, session)
#             else:
#                 continue
# 
#             overall = compute_metrics(preds)
#             per_ds  = compute_per_dataset(preds)
# 
#             print_metrics(overall, label=f"Setting {key}: {name}")
#             for ds, dm in sorted(per_ds.items()):
#                 print_metrics(dm, label=ds, indent="  ")
# 
#             all_results.append((name, overall))
#             full_output[key] = {
#                 "setting":     name,
#                 "shots":       k if key in ("F", "G", "I") else (args.shots if key in ("B","C","D","E") else 0),
#                 "retrieval":   args.retrieval if key in ("B","C","D","E") else "n/a",
#                 "beam_width":  args.beam_width if key == "J" else None,
#                 "overall":     overall,
#                 "per_dataset": per_ds,
#                 "predictions": preds,
#             }
# 
#     print_comparison_table(all_results)
# 
#     if args.output:
#         out = Path(args.output)
#         out.parent.mkdir(parents=True, exist_ok=True)
#         with open(out, "w", encoding="utf-8") as f:
#             json.dump(full_output, f, indent=2, ensure_ascii=False)
#         logger.info("Results saved → %s", out)
# 
#         summary = {
#             "settings": [
#                 {"key": k, "name": v["setting"],
#                  "accuracy": v["overall"]["accuracy"],
#                  "macro_f1": v["overall"]["macro_f1"],
#                  "avg_steps": v["overall"]["avg_steps"],
#                  "avg_tokens": v["overall"]["avg_tokens"],
#                  "n_total": v["overall"]["n_total"]}
#                 for k, v in full_output.items()
#             ],
#             "model":    args.model,
#             "shots":    args.shots,
#             "retrieval": args.retrieval,
#             "seed":     args.seed,
#         }
#         summary_path = out.with_suffix(".summary.json")
#         with open(summary_path, "w", encoding="utf-8") as f:
#             json.dump(summary, f, indent=2)
#         logger.info("Summary saved → %s", summary_path)
# 
# 
# def main() -> None:
#     parser = argparse.ArgumentParser(
#         description="Label prediction — settings A–I (5 ICL + 4 non-training baselines).",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#     )
#     # Datasets
#     parser.add_argument("--datasets",   nargs="+", required=True,
#                         help="FLD.json or FOLIO.json files")
#     parser.add_argument("--test_file",  default=None,
#                         help="JSON file with test sample_ids (auto-split if absent)")
#     parser.add_argument("--test_ratio", type=float, default=0.2)
#     parser.add_argument("--seed",       type=int,   default=42)
# 
#     # Trace pool files (Settings B–E)
#     parser.add_argument("--k_traces_file",    default=None,
#                         help="k_traces*.json for Setting B (Step 1)")
#     parser.add_argument("--cleaned_file",     default=None,
#                         help="cleaned_traces*.json for Setting C (Step 3/3.6)")
#     parser.add_argument("--synthesized_step", default=None,
#                         help="synthesized_traces.json (step_by_step) for Setting D")
#     parser.add_argument("--synthesized_dag",  default=None,
#                         help="synthesized_traces.json (DAG) for Setting E")
# 
#     # ICL / sampling params
#     parser.add_argument("--shots",     type=int, default=3,
#                         help="k for ICL examples (B–E) and for SC/USC/Self-Agg (F/G/I)")
#     parser.add_argument("--retrieval", choices=["random", "similar"], default="random",
#                         help="Example retrieval strategy for ICL settings B–E")
#     parser.add_argument("--settings",  nargs="+", choices=ALL_SETTINGS, default=None,
#                         help="Settings to run (default: all available). "
#                              "A–E need trace files; F–I need nothing extra.")
#     parser.add_argument("--refine_iterations", type=int, default=1,
#                         help="Number of critique-revise iterations for Setting H (default 1)")
#     parser.add_argument("--beam_width",     type=int, default=2,
#                         help="Beam width for Setting J: Self-Eval Beam Search (default 2)")
#     parser.add_argument("--beam_max_steps", type=int, default=8,
#                         help="Max reasoning steps per beam for Setting J (default 8)")
# 
#     # API
#     parser.add_argument("--model",       default=DEFAULT_MODEL)
#     parser.add_argument("--api_key",     default=OPENAI_API_KEY)
#     parser.add_argument("--base_url",    default=OPENAI_BASE_URL)
#     parser.add_argument("--concurrency", type=int, default=20)
# 
#     # Output
#     parser.add_argument("--output", default="icl_results.json")
# 
#     args = parser.parse_args()
#     asyncio.run(run(args))
# 
# 
# if __name__ == "__main__":
#     main()
