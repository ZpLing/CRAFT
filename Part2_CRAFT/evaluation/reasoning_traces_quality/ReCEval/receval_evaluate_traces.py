"""
ReCEval Flat-Text Evaluator — Score LLM-generated reasoning chains

This script implements the flat-text evaluation path that was commented out
in the original evaluate_receval.py (line 361):
    # reasoning_steps = sent_tokenize(entry['steps'])

It reuses ALL scoring functions from evaluate_receval.py unchanged:
    - get_phrases()          → SRL-based RCU extractor (AllenNLP)
    - obtain_entailment_scores()  → DeBERTa NLI
    - obtain_unit_pvi_score()     → PVI intra-step correctness (T5 fine-tuned)
    - obtain_contradiction_score() → global coherence
    - obtain_info_gain_score()    → LL informativeness (GPT-2-XL)

The only change: instead of reading EB tree structures and calling
get_reasoning_chain_text(), we take a flat list of step strings directly.

The two containers are named craft / raw: the CRAFT
comparison feeds this scorer through receval_adapter_craft.py, which puts CRAFT's
synthesized trace in "craft" and the raw CoT in "raw".
unmodified, and receval_build_table.py relabels them Raw / CRAFT in the table. They
are containers, not settings — Part 1's w/ Answer vs w/o Answer is a different axis.

Input JSON (output of receval_generate_traces.py):
    [{"id": "...", "hypothesis": "...", "question": "...",
      "craft": {"steps": ["Step 1 ...", "Step 2 ..."]},
      "raw":       {"steps": ["Step 1 ...", "Step 2 ..."]}}, ...]

Output JSON:
    {
      "craft": {
        "per_sample": [{"id": "...", "entail": 0.91, "ll_info": 0.12, ...}],
        "aggregate":  {"entail": 0.88, "ll_info": 0.09, ...}
      },
      "raw": { ... same structure ... },
      "comparison": {"entail_delta": 0.03, "ll_info_delta": 0.03, ...}
    }

Usage:
    python receval_evaluate_traces.py \\
        --input receval_traces.json \\
        --output receval_scores.json \\
        --score_keys entail ll-info contradict \\
        --K 1

Dependencies: same as evaluate_receval.py
    pip install supar allennlp allennlp-models transformers torch scipy nltk
    python -m nltk.downloader punkt averaged_perceptron_tagger
"""

import argparse
import json
import os
import sys
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re, string

# ---------------------------------------------------------------------------
# Import all model/scoring infrastructure from evaluate_receval.py
# We add ReCEval to path and import its functions directly to avoid duplication
# ---------------------------------------------------------------------------
RECEVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RECEVAL_DIR))

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
try:
    from config import resolve_input, resolve_output
except ImportError:
    resolve_input = resolve_output = Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Quiet the libraries ReCEval pulls in while loading models. Naming them directly
# rather than disabling every existing logger, which would also silence this module —
# and which would miss the ReCEval loggers anyway, since those are created by the
# imports below, after this point.
for _noisy in ("transformers", "allennlp", "supar", "filelock", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Re-import all scoring functions and models from evaluate_receval.py
# We do this by exec'ing the init portion up to the main loop.
# This ensures we use exactly the same models and functions.
# ---------------------------------------------------------------------------
def load_receval_components():
    """
    Load all models and functions from evaluate_receval.py.
    Returns a namespace dict with all the scoring functions.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "evaluate_receval",
        str(RECEVAL_DIR / "evaluate_receval.py"),
    )
    # We can't simply import it because the main loop runs at module level.
    # Instead, re-implement model loading + copy the pure functions.
    # This keeps strict parity with the original.
    pass


# ---- Model loading (mirrors evaluate_receval.py lines 37–93 exactly) ----
# The models load on demand rather than at import. Upstream loads all four
# unconditionally, which makes a run of the two metrics the paper reports
# (entail, contradict) depend on the two PVI checkpoints and GPT-2-XL, neither
# of which it needs: PVI is downloaded separately and is absent here, so the
# import raised before the first sample was ever scored. Each loader below
# fills the same module-level names the scoring functions already read, so the
# functions themselves are still upstream's, unchanged.

from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM,
)
from datasets import Dataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# NLI model — entail and contradict.
# AllenNLP pins transformers 4.20, which reads the Hub's redirect to a relative
# Location header as if it were a URL and so cannot download anything from
# huggingface.co any more. The weights are therefore fetched once with a current
# transformers (any environment with one) and read from disk here. RECEVAL_NLI_MODEL
# overrides the location; the Hub id is still the default, for a machine whose
# transformers can reach it.
_LOCAL_NLI = RECEVAL_DIR.parent / ".models" / "deberta-v3-large-mnli"
ent_model_name = os.getenv("RECEVAL_NLI_MODEL") or (
    str(_LOCAL_NLI) if (_LOCAL_NLI / "config.json").exists()
    else "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
)
ent_tokenizer = ent_model = None

# SRL predictor — RCU extraction, needed by every metric
SRL_MODEL_URL = ("https://storage.googleapis.com/allennlp-public-models/"
                 "structured-prediction-srl-bert.2020.12.15.tar.gz")
srl_predictor = None

# PVI models (fine-tuned T5 from ReCEval) — the pvi metric only
inp_model_dir     = str(RECEVAL_DIR / "PVI/inp_models/")
no_inp_model_dir  = str(RECEVAL_DIR / "PVI/noinp_models/")
info_gain_model_dir = str(RECEVAL_DIR / "PVI/infogain_models/")

max_input_length = 512
max_target_length = 64
padding = "max_length"
pad_token = "<pad>"
prefix = "Generate entailed sentence: "

inp_tokenizer = inp_config = inp_model = None
no_inp_tokenizer = no_inp_config = no_inp_model = None

# Info-gain model: GPT-2-XL (default in evaluate_receval.py) — the ll-info metric only
info_gain_mname = "gpt2"
ll_tokenizer = ll_model = None


def load_nli() -> None:
    """DeBERTa NLI, behind entail and contradict."""
    global ent_tokenizer, ent_model
    if ent_model is not None:
        return
    logger.info("Loading NLI model %s on %s ...", ent_model_name, device)
    ent_tokenizer = AutoTokenizer.from_pretrained(ent_model_name)
    ent_model = AutoModelForSequenceClassification.from_pretrained(ent_model_name).to(device)
    ent_model.eval()


def load_srl() -> None:
    """AllenNLP SRL, behind get_phrases(), which every metric goes through."""
    global srl_predictor
    if srl_predictor is not None:
        return
    logger.info("Loading SRL predictor ...")
    srl_predictor = Predictor.from_path(SRL_MODEL_URL)


def load_pvi() -> None:
    """The two fine-tuned T5 checkpoints, behind pvi."""
    global inp_tokenizer, inp_config, inp_model
    global no_inp_tokenizer, no_inp_config, no_inp_model
    if inp_model is not None:
        return
    missing = [d for d in (inp_model_dir, no_inp_model_dir) if not Path(d).is_dir()]
    if missing:
        raise FileNotFoundError(
            "The pvi metric needs ReCEval's fine-tuned T5 checkpoints, which are "
            "downloaded separately and are not in this tree: "
            + ", ".join(missing)
            + ". Drop them there, or leave 'pvi' out of --score_keys."
        )
    logger.info("Loading PVI models on %s ...", device)
    inp_tokenizer = AutoTokenizer.from_pretrained(inp_model_dir)
    inp_config    = AutoConfig.from_pretrained(inp_model_dir)
    inp_model     = AutoModelForSeq2SeqLM.from_pretrained(inp_model_dir, config=inp_config).to(device).eval()

    no_inp_tokenizer = AutoTokenizer.from_pretrained(no_inp_model_dir)
    no_inp_config    = AutoConfig.from_pretrained(no_inp_model_dir)
    no_inp_model     = AutoModelForSeq2SeqLM.from_pretrained(no_inp_model_dir, config=no_inp_config).to(device).eval()


def load_ll() -> None:
    """GPT-2-XL, behind ll-info."""
    global ll_tokenizer, ll_model
    if ll_model is not None:
        return
    logger.info("Loading %s / gpt2-xl on %s ...", info_gain_mname, device)
    ll_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    ll_model     = AutoModelForCausalLM.from_pretrained("gpt2-xl").eval().to(device)
    ll_tokenizer.padding_side = "left"
    ll_tokenizer.pad_token    = ll_tokenizer.eos_token
    ll_model.config.pad_token_id = ll_model.config.eos_token_id


def load_models(score_keys) -> None:
    """Load exactly the models the requested metrics use."""
    load_srl()
    if {"entail", "contradict"} & set(score_keys):
        load_nli()
    if "pvi" in score_keys:
        load_pvi()
    if "ll-info" in score_keys:
        load_ll()
    logger.info("Models loaded for metrics: %s", ", ".join(score_keys))

# ---- Scoring functions (copied verbatim from evaluate_receval.py) ----

def detokenize(tokens):
    return TreebankWordDetokenizer().detokenize(tokens)

def obtain_entailment_scores(premise, hypothesis):
    inp = ent_tokenizer(premise, hypothesis, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        output = ent_model(inp["input_ids"].to(device))
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    label_names = ["entailment", "neutral", "contradiction"]
    prediction = {name: float(pred) for pred, name in zip(prediction, label_names)}
    return prediction["entailment"]

def obtain_contradiction_scores(premise, hypothesis):
    inp = ent_tokenizer(premise, hypothesis, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        output = ent_model(inp["input_ids"].to(device))
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    label_names = ["entailment", "neutral", "contradiction"]
    prediction = {name: float(pred) for pred, name in zip(prediction, label_names)}
    return prediction["contradiction"]

def obtain_unit_entailment_score(prem_units, conc_units):
    if len(prem_units):
        premise    = " and ".join(prem_units)
        hypothesis = " and ".join(conc_units)
        score = obtain_entailment_scores(premise, hypothesis)
    else:
        score = 1
    return score

def obtain_contradiction_score(prem_units, conc_units):
    pair_scores = []
    hypothesis  = " and ".join(conc_units)
    for premise in prem_units:
        pair_scores.append(obtain_contradiction_scores(premise, hypothesis))
    if len(pair_scores):
        score = 1 - max(pair_scores)
    else:
        score = 1
    return score

def verb_modifiers(desc):
    filtered_mods = []
    mods = re.findall(r"\[ARGM.*?\]", desc)
    if not len(mods): return filtered_mods
    for mod in mods:
        phrase = mod.split(": ")[1].rstrip("]")
        verb_match = ["VB" in k[1] for k in nltk.pos_tag(word_tokenize(phrase))]
        if sum(verb_match) and len(phrase.split()) > 2:
            filtered_mods.append(phrase)
    return filtered_mods

def remove_modifiers(sent, modifiers):
    if not len(modifiers): return sent
    for mod in modifiers:
        sent = sent.replace(mod, "")
        sent = re.sub(" +", " ", sent)
        sent = sent.strip(string.punctuation + " ")
    return sent

def extract_frame(tags, words, desc):
    start, end = None, None
    if len(set(tags)) == 1: return ""
    tags = [t if "C-ARG" not in t else "O" for t in tags]
    for w in range(len(words)):
        if "B-" in tags[w] and start is None: start = w
        if tags[len(words) - w - 1] != "O" and end is None: end = len(words) - w - 1
    if end is None: end = start
    sent = detokenize(words[start: end + 1]).rstrip(".")
    return sent

# SRL's encoder is BERT, so a step longer than its 512 positions overflows the
# position embeddings and takes the whole run down with it. The logical sets'
# steps never come close; a competition-maths step does, because a block of
# LaTeX is many wordpieces per word. Halving is used rather than a word cap
# because no word count predicts how a formula tokenises.
SRL_SHRINK_TRIES = 4
SRL_MIN_WORDS = 8


def srl_parse(sent: str):
    """(srl_out, the text it parsed), or (None, sent) if SRL could not parse it.

    A step that will not fit is cut down until it does, and the caller scores the
    part that fitted. One that still will not parse is treated as a single unit,
    which is what get_phrases already does for a sentence SRL finds no frame in.
    """
    text = sent
    for _ in range(SRL_SHRINK_TRIES):
        try:
            return srl_predictor.predict(text), text
        except RuntimeError as exc:
            if "size of tensor" not in str(exc):
                raise
            words = text.split()
            if len(words) < 2 * SRL_MIN_WORDS:
                break
            text = " ".join(words[: len(words) // 2])
            logger.warning("Step too long for SRL; retrying on its leading %d words",
                           len(text.split()))
    logger.warning("SRL could not parse a %d-word step; treating it as one unit",
                   len(sent.split()))
    return None, sent


def get_phrases(sent):
    phrases = []
    history = ""
    srl_out, sent = srl_parse(sent)
    if srl_out is None:
        return [sent.rstrip(".")]
    words   = srl_out["words"]
    frames  = [s["tags"]        for s in srl_out["verbs"]]
    descs   = [s["description"] for s in srl_out["verbs"]]
    mod_sent = detokenize(words).rstrip(".")
    for frame, desc in zip(frames, descs):
        phrase = extract_frame(frame, words, desc)
        if phrase == mod_sent: phrase = remove_modifiers(phrase, verb_modifiers(desc))
        phrases.append(phrase)
    phrases.sort(key=lambda s: len(s), reverse=True)
    filtered_phrases = []
    for p in phrases:
        if p not in history:
            history += " " + p
            filtered_phrases.append(p)
    if len(filtered_phrases):
        filtered_phrases.sort(key=lambda s: mod_sent.find(s))
        left = mod_sent
        mod_filt = False
        for fp in filtered_phrases: left = left.replace(fp, "#").strip(string.punctuation + " ")
        for l in left.split("#"):
            l = l.strip(string.punctuation + " ")
            if len(l.split()) >= 4 and l not in " ".join(filtered_phrases):
                verb_match = ["VB" in k[1] for k in nltk.pos_tag(word_tokenize(l))]
                if sum(verb_match):
                    filtered_phrases.append(l)
                    mod_filt = True
        if mod_filt: filtered_phrases.sort(key=lambda s: mod_sent.find(s))
        return filtered_phrases
    else:
        return [sent.rstrip(".")]

def preprocess_and_convert(premise_units, conc_units):
    data = {"inputs": [], "labels": []}
    parent_text = " & ".join(premise_units) + " ->"
    child_text  = " " + conc_units[0]
    data["inputs"].append(parent_text)
    data["labels"].append(child_text)
    return data

def postprocess_test_data(examples):
    inputs = [prefix + text for text in examples["inputs"]]
    model_inputs = inp_tokenizer(inputs, max_length=max_input_length,
                                  padding=padding, truncation=True, return_tensors="pt")
    with inp_tokenizer.as_target_tokenizer():
        targets = [pad_token + label for label in examples["labels"]]
        labels  = inp_tokenizer(targets, max_length=max_target_length,
                                padding=padding, truncation=True, return_tensors="pt")
    model_inputs["decoder_input_ids"]      = labels["input_ids"]
    model_inputs["decoder_attention_mask"] = labels["attention_mask"]
    return model_inputs

def noinp_postprocess_test_data(examples):
    inputs = [prefix + "None ->" for _ in examples["inputs"]]
    model_inputs = no_inp_tokenizer(inputs, max_length=max_input_length,
                                     padding=padding, truncation=True, return_tensors="pt")
    with no_inp_tokenizer.as_target_tokenizer():
        targets = [pad_token + label for label in examples["labels"]]
        labels  = no_inp_tokenizer(targets, max_length=max_target_length,
                                   padding=padding, truncation=True, return_tensors="pt")
    model_inputs["decoder_input_ids"]      = labels["input_ids"]
    model_inputs["decoder_attention_mask"] = labels["attention_mask"]
    return model_inputs

def obtain_log_prob(predict_dataset, model, tokenizer):
    logits = model(
        input_ids=torch.Tensor(predict_dataset["input_ids"]).long().to(device),
        attention_mask=torch.Tensor(predict_dataset["attention_mask"]).long().to(device),
        decoder_input_ids=torch.Tensor(predict_dataset["decoder_input_ids"]).long().to(device),
        decoder_attention_mask=torch.Tensor(predict_dataset["decoder_attention_mask"]).long().to(device),
    ).logits
    all_logprobs = torch.log(torch.softmax(logits, dim=-1))
    labels = tokenizer(predict_dataset["labels"], max_length=max_target_length).input_ids
    filter_sums = []
    for row, label in zip(all_logprobs, labels):
        label.pop()
        row = row[: len(label), :].detach().cpu().numpy()
        vocab_size = row.shape[-1]
        loc = F.one_hot(torch.tensor(label), num_classes=vocab_size).numpy().astype(bool)
        filter_sums.append(np.sum(row, where=loc) / len(label))
    return np.array(filter_sums)

def obtain_unit_pvi_score(premise_units, conc_units):
    dataset        = Dataset.from_dict(preprocess_and_convert(premise_units, conc_units))
    inp_dataset    = dataset.map(postprocess_test_data,       batched=True, remove_columns=["inputs"])
    no_inp_dataset = dataset.map(noinp_postprocess_test_data, batched=True, remove_columns=["inputs"])
    inp_logprob    = obtain_log_prob(inp_dataset,    inp_model,    inp_tokenizer)[0]
    no_inp_logprob = obtain_log_prob(no_inp_dataset, no_inp_model, no_inp_tokenizer)[0]
    return inp_logprob - no_inp_logprob

def slice_select_logits(all_logprobs, label):
    row        = all_logprobs[-len(label):, :].detach().cpu().numpy()  # GPT-2 path
    vocab_size = row.shape[-1]
    loc = F.one_hot(torch.tensor(label), num_classes=vocab_size).numpy().astype(bool)
    return np.array([np.sum(row, where=loc) / len(label)])

def obtain_info_gain_score(prev_steps, current_step, conc_units, target, model, tokenizer):
    """LL-info path (GPT-2-XL), verbatim from evaluate_receval.py lines 319–334."""
    target = " " + target
    input_text = " " + " ".join(prev_steps + [current_step]) + " Therefore," + target
    if len(prev_steps):
        ref_text = " " + " ".join(prev_steps) + " Therefore," + target
    else:
        ref_text = " Therefore," + target
    labels     = tokenizer(target).input_ids
    input_ids  = tokenizer(input_text,  return_tensors="pt").input_ids.to(device)
    ref_ids    = tokenizer(ref_text,    return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        inp_logits = model(input_ids=input_ids,  return_dict=True).logits.detach().cpu()
        ref_logits = model(input_ids=ref_ids,    return_dict=True).logits.detach().cpu()
    all_inp = torch.log(torch.softmax(inp_logits, dim=-1))
    all_ref = torch.log(torch.softmax(ref_logits, dim=-1))
    filtered_inp = slice_select_logits(all_inp[0, :-1, :], labels)[0]
    filtered_ref = slice_select_logits(all_ref[0, :-1, :], labels)[0]
    return filtered_inp - filtered_ref


# ---------------------------------------------------------------------------
# Flat-text scoring (the commented-out path from evaluate_receval.py line 361)
# ---------------------------------------------------------------------------

def score_chain(
    reasoning_steps: list[str],
    input_context: str,
    hypothesis: str,
    score_keys: list[str],
    K: int = 0,
) -> dict:
    """
    Score a flat list of reasoning step strings using ReCEval metrics.

    This is the implementation of:
        reasoning_steps = sent_tokenize(entry['steps'])
    from evaluate_receval.py line 361 (the commented-out flat-text branch).

    Per-step scores are aggregated as min() over the chain (same as original).
    """
    input_context_sentences = sent_tokenize(input_context)
    running_conc = []

    step_scores = {k: [] for k in score_keys}

    for sid, step in enumerate(reasoning_steps):
        if not step.strip():
            continue
        # RCU extraction via SRL (same as original get_phrases call)
        units = get_phrases(step)
        premise_units = units[:-1]
        conc_units    = [units[-1]]

        if "entail" in score_keys:
            step_scores["entail"].append(
                obtain_unit_entailment_score(premise_units + running_conc[-K:] if K else premise_units, conc_units)
            )
        if "pvi" in score_keys:
            step_scores["pvi"].append(
                obtain_unit_pvi_score(premise_units + running_conc[-K:] if K else premise_units, conc_units)
            )
        if "contradict" in score_keys:
            step_scores["contradict"].append(
                obtain_contradiction_score(input_context_sentences + running_conc, conc_units)
            )
        if "ll-info" in score_keys:
            step_scores["ll-info"].append(
                obtain_info_gain_score(
                    reasoning_steps[:sid], step, conc_units,
                    hypothesis, ll_model, ll_tokenizer,
                )
            )

        running_conc.extend(conc_units)

    # Aggregate: min score over all steps (same as evaluate_receval.py lines 396–400)
    result = {}
    for k in score_keys:
        scores = step_scores[k]
        result[k] = float(min(scores)) if scores else None
        result[f"{k}_per_step"] = [float(s) for s in scores]

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "ReCEval flat-text evaluator.\n"
            "Scores LLM-generated reasoning chains from receval_generate_traces.py\n"
            "using the same metrics as evaluate_receval.py (entail, pvi, contradict, ll-info).\n"
            "Implements the commented-out flat-text path from evaluate_receval.py line 361."
        )
    )
    parser.add_argument("--input",  required=True,
                        help="receval_traces.json from generate script "
                             "(relative paths resolve under the results root)")
    parser.add_argument("--output", default="CRAFT_results/reasoning_traces_quality/ReCEval/receval_scores.json",
                        help="Output JSON with scores "
                             "(relative paths resolve under the results root)")
    parser.add_argument(
        "--score_keys", nargs="+",
        default=["ll-info"],
        choices=["entail", "pvi", "contradict", "ll-info"],
        help="Metrics to compute (default: ll-info, same as evaluate_receval.py default)",
    )
    parser.add_argument("--K", type=int, default=0,
                        help="How many prior conclusions to include as context (default: 0)")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    args.input  = str(resolve_input(args.input))
    args.output = str(resolve_output(args.output))

    with open(args.input, "r", encoding="utf-8") as f:
        items = json.load(f)

    if args.max_samples:
        items = items[: args.max_samples]
    logger.info("Scoring %d items with metrics: %s", len(items), args.score_keys)
    load_models(args.score_keys)

    wa_results, bl_results = [], []

    for i, item in enumerate(items):
        logger.info("[%d/%d] id=%s", i + 1, len(items), item.get("id", ""))
        hypothesis = item.get("hypothesis", "")
        question   = item.get("question", "")  # joined premises as input context

        for setting, container in (("craft", wa_results), ("raw", bl_results)):
            steps = item[setting].get("steps", [])
            if not steps:
                logger.warning("Empty steps for id=%s setting=%s", item.get("id"), setting)
                container.append({"id": item.get("id", ""), **{k: None for k in args.score_keys}})
                continue
            scores = score_chain(steps, question, hypothesis, args.score_keys, args.K)
            container.append({"id": item.get("id", ""), **scores})

    def aggregate(results: list[dict], keys: list[str]) -> dict:
        agg = {}
        for k in keys:
            vals = [r[k] for r in results if r.get(k) is not None]
            agg[k] = float(np.mean(vals)) if vals else None
        return agg

    wa_agg = aggregate(wa_results, args.score_keys)
    bl_agg = aggregate(bl_results, args.score_keys)
    comparison = {
        f"{k}_delta": (
            round(wa_agg[k] - bl_agg[k], 6)
            if wa_agg.get(k) is not None and bl_agg.get(k) is not None
            else None
        )
        for k in args.score_keys
    }

    output = {
        "craft": {"per_sample": wa_results, "aggregate": wa_agg},
        "raw":       {"per_sample": bl_results, "aggregate": bl_agg},
        "comparison":  comparison,
        "config": {"score_keys": args.score_keys, "K": args.K, "n_samples": len(items)},
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("Saved scores → %s", out_path)
    logger.info("=== ReCEval Score Comparison ===")
    for k in args.score_keys:
        logger.info(
            "%-12s  craft=%.4f  raw=%.4f  delta=%+.4f",
            k,
            wa_agg.get(k) or float("nan"),
            bl_agg.get(k) or float("nan"),
            comparison.get(f"{k}_delta") or float("nan"),
        )


if __name__ == "__main__":
    main()
