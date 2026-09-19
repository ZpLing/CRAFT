"""
ReCEval NLI-subset scorer (entail + contradict), batched version.

Uses the SAME DeBERTa NLI model as ReCEval and the SAME scoring formulas
(obtain_unit_entailment_score / obtain_contradiction_score). Each step is
treated as a single RCU (bypassing AllenNLP SRL, which is OOD on symbolic logic).

Performance: all NLI pairs for the whole input file are collected first,
then batched through the model with batch_size=--batch (default 64). On MPS
this is 10-30x faster than per-call inference.

Input : receval_inputs/*.json (from receval_adapter_craft.py):
    [{id, hypothesis, question,
      with_answer: {steps: [...]},  # CRAFT post
      blind:       {steps: [...]}}] # raw CoT

Output: JSON consumable by receval_build_table.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Batched NLI
# ---------------------------------------------------------------------------

class BatchNLI:
    def __init__(self, device, max_len: int = 512):
        self.device = device
        self.max_len = max_len
        logger.info("Loading %s on %s ...", MODEL_ID, device)
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device).eval()
        # labels order: [entailment, neutral, contradiction] per MoritzLaurer card
        self.entail_idx = 0
        self.contra_idx = 2

    @torch.no_grad()
    def probs(self, pairs: list[tuple[str, str]], batch_size: int = 64) -> np.ndarray:
        """Return (N, 3) softmax probabilities for each (premise, hypothesis) pair."""
        if not pairs:
            return np.zeros((0, 3), dtype=np.float32)
        out = np.zeros((len(pairs), 3), dtype=np.float32)
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            prems = [p for p, _ in chunk]
            hyps  = [h for _, h in chunk]
            enc = self.tok(prems, hyps, truncation=True, padding=True,
                           max_length=self.max_len, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits
            p = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            out[start:start + len(chunk)] = p
        return out


# ---------------------------------------------------------------------------
# Build all NLI pair jobs for one input file
# ---------------------------------------------------------------------------

def plan_sample(item: dict, setting: str, job_idx: int):
    """Return (entail_pairs, contradict_pairs, meta) where meta holds slicing info
    to reassemble per-step scores after batched inference.

    entail pair for step i (i>=1): premise = " ".join(steps[:i]), hypothesis = steps[i]
    contradict pairs for step i: for each p in (ctx_sents + prior_steps): (p, steps[i])
    """
    steps_raw = item[setting].get("steps", [])
    steps = [s.strip() for s in steps_raw if s and s.strip()]
    ctx = item.get("question", "")
    ctx_sents = sent_tokenize(ctx) if ctx else []

    entail_pairs: list[tuple[str, str]] = []
    entail_step_ids: list[int] = []   # which step index each pair corresponds to
    contradict_pairs: list[tuple[str, str]] = []
    contradict_slices: list[tuple[int, int, int]] = []  # (step_idx, start, end) in contradict_pairs

    for i, s in enumerate(steps):
        # entail: prior steps → current step
        if i > 0:
            entail_pairs.append((" ".join(steps[:i]), s))
            entail_step_ids.append(i)
        # contradict: for each premise unit
        prem_units = ctx_sents + steps[:i]
        if prem_units:
            start = len(contradict_pairs)
            for p in prem_units:
                contradict_pairs.append((p, s))
            end = len(contradict_pairs)
            contradict_slices.append((i, start, end))
        else:
            contradict_slices.append((i, -1, -1))

    meta = {
        "job_idx": job_idx,
        "setting": setting,
        "id": item.get("id", ""),
        "n_steps": len(steps),
        "entail_step_ids": entail_step_ids,
        "contradict_slices": contradict_slices,
    }
    return entail_pairs, contradict_pairs, meta


def assemble(meta, entail_probs, contradict_probs, ent_idx, con_idx):
    """Reconstruct per-step entail & contradict scores from batched output."""
    n = meta["n_steps"]
    entail_per = [1.0] * n  # step 0 and any with no premise → 1.0 (matches ReCEval)
    for pair_pos, step_i in enumerate(meta["entail_step_ids"]):
        entail_per[step_i] = float(entail_probs[pair_pos, ent_idx])
    contradict_per = [1.0] * n
    for step_i, start, end in meta["contradict_slices"]:
        if start == -1:
            continue
        max_c = float(contradict_probs[start:end, con_idx].max()) if end > start else 0.0
        contradict_per[step_i] = 1.0 - max_c
    return entail_per, contradict_per


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch", type=int, default=64, help="NLI batch size (try 32/64/128)")
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()

    with open(args.input) as f:
        items = json.load(f)
    if args.max_samples:
        items = items[: args.max_samples]

    device = pick_device()
    scorer = BatchNLI(device)

    # ---- Plan: collect all pairs across all samples × both settings ----
    entail_pairs_all: list[tuple[str, str]] = []
    contradict_pairs_all: list[tuple[str, str]] = []
    metas: list[dict] = []  # one per (sample, setting)
    ent_offsets: list[tuple[int, int]] = []  # (start, end) into entail_pairs_all
    con_offsets: list[tuple[int, int]] = []

    for idx, item in enumerate(items):
        for setting in ("with_answer", "blind"):
            e_pairs, c_pairs, meta = plan_sample(item, setting, job_idx=len(metas))
            es = len(entail_pairs_all); cs = len(contradict_pairs_all)
            entail_pairs_all.extend(e_pairs)
            contradict_pairs_all.extend(c_pairs)
            ent_offsets.append((es, len(entail_pairs_all)))
            con_offsets.append((cs, len(contradict_pairs_all)))
            metas.append(meta)

    total_pairs = len(entail_pairs_all) + len(contradict_pairs_all)
    logger.info("N=%d items | entail_pairs=%d | contradict_pairs=%d | total=%d",
                len(items), len(entail_pairs_all), len(contradict_pairs_all), total_pairs)

    # ---- Run NLI in batches ----
    t0 = time.time()
    logger.info("Running entail (batch=%d) ...", args.batch)
    entail_probs = scorer.probs(entail_pairs_all, batch_size=args.batch)
    logger.info("  entail done in %.1fs", time.time() - t0)

    t0 = time.time()
    logger.info("Running contradict (batch=%d) ...", args.batch)
    contradict_probs = scorer.probs(contradict_pairs_all, batch_size=args.batch)
    logger.info("  contradict done in %.1fs", time.time() - t0)

    # ---- Reassemble ----
    wa_results, bl_results = [], []
    for mi, meta in enumerate(metas):
        e_start, e_end = ent_offsets[mi]
        c_start, c_end = con_offsets[mi]
        e_probs = entail_probs[e_start:e_end]
        c_probs = contradict_probs[c_start:c_end]

        # The meta uses local indices into its own e_pairs / c_pairs; remap:
        # plan_sample returns local-indexed pair positions; after slicing they are 0-indexed again.
        entail_per, contradict_per = assemble(meta, e_probs, c_probs, scorer.entail_idx, scorer.contra_idx)

        rec = {
            "id": meta["id"],
            "entail":         float(min(entail_per))     if entail_per else None,
            "entail_mean":    float(np.mean(entail_per)) if entail_per else None,
            "contradict":     float(min(contradict_per)) if contradict_per else None,
            "contradict_mean":float(np.mean(contradict_per)) if contradict_per else None,
            "entail_per_step":     entail_per,
            "contradict_per_step": contradict_per,
        }
        (wa_results if meta["setting"] == "with_answer" else bl_results).append(rec)

    def agg(results, key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    keys = ["entail", "entail_mean", "contradict", "contradict_mean"]
    wa_agg = {k: agg(wa_results, k) for k in keys}
    bl_agg = {k: agg(bl_results, k) for k in keys}

    def delta(k):
        if wa_agg.get(k) is None or bl_agg.get(k) is None:
            return None
        return round(wa_agg[k] - bl_agg[k], 6)

    output = {
        "with_answer": {"per_sample": wa_results, "aggregate": wa_agg},
        "blind":       {"per_sample": bl_results, "aggregate": bl_agg},
        "comparison":  {f"{k}_delta": delta(k) for k in keys},
        "config": {
            "nli_model": MODEL_ID,
            "n_samples": len(items),
            "batch": args.batch,
            "note": "NLI-subset ReCEval: each step = one RCU (no SRL); DeBERTa-v3-large-mnli.",
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("Saved → %s", out)
    logger.info("=== %s  (N=%d) ===", Path(args.input).stem, len(items))
    for k in keys:
        if wa_agg.get(k) is None:
            continue
        logger.info("%-18s  raw=%.4f  craft=%.4f  delta=%+.4f",
                    k, bl_agg[k], wa_agg[k], delta(k))


if __name__ == "__main__":
    main()
