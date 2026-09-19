"""
ReCEval flat-text scorer (MPS-accelerated, AllenNLP-free).

Same input / output schema as receval_evaluate_traces.py, but:
  - Drops AllenNLP SRL.  RCU extraction is sentence-level: split each step
    with nltk.sent_tokenize(), last sentence = conclusion, prior = premises.
  - Runs DeBERTa NLI on MPS (Apple Silicon GPU) via torch 2.x.
  - Only entail + contradict (the two NLI-based metrics).  pvi / ll-info
    need fine-tuned ReCEval T5/GPT-2 checkpoints that aren't shipped.

Trade-off vs the SRL path: relative ordering of traces is usually preserved,
absolute scores will differ because RCUs are coarser.

Run inside the `ML` conda env (torch>=2.0, transformers>=4.30):
    python receval_evaluate_traces_mps.py \\
        --input  receval_inputs/fld_o4mini.json \\
        --output receval_scores/fld_o4mini_30_mps.json \\
        --score_keys entail contradict --K 0 --max_samples 30
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import nltk
from nltk import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# -- device selection ---------------------------------------------------------
_force = os.environ.get("RECEVAL_DEVICE", "").lower()
if _force in ("cuda", "mps", "cpu"):
    device = torch.device(_force)
elif torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
logger.info("Using device: %s", device)

NLI_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
ent_tokenizer = None
ent_model = None


def load_nli():
    global ent_tokenizer, ent_model
    if ent_model is not None:
        return
    logger.info("Loading DeBERTa NLI on %s ...", device)
    from huggingface_hub import snapshot_download
    local = snapshot_download(repo_id=NLI_NAME)
    ent_tokenizer = AutoTokenizer.from_pretrained(local)
    ent_model = AutoModelForSequenceClassification.from_pretrained(local).to(device).eval()


@torch.no_grad()
def nli_probs(premise: str, hypothesis: str) -> dict:
    inp = ent_tokenizer(premise, hypothesis, truncation=True, return_tensors="pt").to(device)
    logits = ent_model(**inp).logits[0]
    p = torch.softmax(logits, dim=-1).tolist()
    return {"entailment": p[0], "neutral": p[1], "contradiction": p[2]}


def entail_score(premise_units, conc_units):
    if not premise_units:
        return 1.0
    return nli_probs(" and ".join(premise_units), " and ".join(conc_units))["entailment"]


def contradict_score(premise_units, conc_units):
    """1 - max contradiction probability across premise units. Higher = less contradicted."""
    if not premise_units:
        return 1.0
    hyp = " and ".join(conc_units)
    cmax = 0.0
    for p in premise_units:
        c = nli_probs(p, hyp)["contradiction"]
        if c > cmax:
            cmax = c
    return 1.0 - cmax


def get_phrases(step: str) -> list[str]:
    """Sentence-level RCU: each sentence is one unit."""
    sents = [s.strip() for s in sent_tokenize(step) if s.strip()]
    return sents if sents else [step.strip()]


def score_chain(reasoning_steps, input_context, hypothesis, score_keys, K=0):
    input_ctx_sents = sent_tokenize(input_context)
    running_conc = []
    out = {k: [] for k in score_keys}

    for step in reasoning_steps:
        if not step.strip():
            continue
        units = get_phrases(step)
        premise_units = units[:-1]
        conc_units = [units[-1]]

        if "entail" in score_keys:
            ctx = premise_units + (running_conc[-K:] if K else [])
            out["entail"].append(entail_score(ctx, conc_units))

        if "contradict" in score_keys:
            ctx = input_ctx_sents + running_conc
            out["contradict"].append(contradict_score(ctx, conc_units))

        running_conc.extend(conc_units)

    res = {}
    for k in score_keys:
        s = out[k]
        res[k] = float(min(s)) if s else None
        res[f"{k}_per_step"] = [float(x) for x in s]
    return res


def main():
    ap = argparse.ArgumentParser(description="MPS-accelerated ReCEval scorer (no AllenNLP).")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--score_keys", nargs="+", default=["entail", "contradict"],
                    choices=["entail", "contradict"])
    ap.add_argument("--K", type=int, default=0)
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()

    nltk.download("punkt", quiet=True)

    items = json.load(open(args.input, "r", encoding="utf-8"))
    if args.max_samples:
        items = items[: args.max_samples]
    logger.info("Scoring %d items with %s", len(items), args.score_keys)

    load_nli()

    wa, bl = [], []
    for i, item in enumerate(items):
        if (i + 1) % 5 == 0 or i == 0:
            logger.info("[%d/%d] id=%s", i + 1, len(items), item.get("id", ""))
        hyp = item.get("hypothesis", "")
        ctx = item.get("question", "")

        for setting, container in (("with_answer", wa), ("blind", bl)):
            steps = item[setting].get("steps", [])
            if not steps:
                container.append({"id": item.get("id", ""), **{k: None for k in args.score_keys}})
                continue
            container.append({"id": item.get("id", ""), **score_chain(steps, ctx, hyp, args.score_keys, args.K)})

    def agg(xs, keys):
        a = {}
        for k in keys:
            v = [r[k] for r in xs if r.get(k) is not None]
            a[k] = float(np.mean(v)) if v else None
        return a

    wa_a, bl_a = agg(wa, args.score_keys), agg(bl, args.score_keys)
    cmp = {f"{k}_delta": (round(wa_a[k] - bl_a[k], 6) if wa_a.get(k) is not None and bl_a.get(k) is not None else None)
           for k in args.score_keys}

    output = {
        "with_answer": {"per_sample": wa, "aggregate": wa_a},
        "blind":       {"per_sample": bl, "aggregate": bl_a},
        "comparison":  cmp,
        "config": {"score_keys": args.score_keys, "K": args.K, "n_samples": len(items),
                   "scorer": "mps_no_srl", "device": str(device)},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(output, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    logger.info("Saved → %s", args.output)
    for k in args.score_keys:
        logger.info("%-12s with_answer=%.4f  blind=%.4f  delta=%+.4f",
                    k, wa_a.get(k) or float("nan"), bl_a.get(k) or float("nan"),
                    cmp.get(f"{k}_delta") or float("nan"))


if __name__ == "__main__":
    main()
