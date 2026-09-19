"""
roscoe_score.py — Score exported traces with ROSCOE, and report the comparison.

Imports ROSCOE's Evaluator out of a ParlAI checkout directly rather than shelling
out to its CLI, and runs it in sentence_transformer mode, so the checkout needs no
patching: the simcse import upstream insists on is stubbed out below.
"""
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


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
    import types as _types

    roscoe_dir = Path(roscoe_dir).resolve()
    parlai_root = roscoe_dir.parent.parent  # ParlAI/
    _sys.path.insert(0, str(parlai_root))

    # Upstream score.py raises at import time when the simcse package is absent,
    # although the sentence_transformer path taken here never builds a SimCSE model.
    # A stub satisfies that import without vendoring a patched copy of ParlAI; it
    # raises if anything actually reaches for SimCSE, so a wrong model type fails
    # loudly instead of scoring with something else.
    if "simcse" not in _sys.modules:
        _stub = _types.ModuleType("simcse")

        def _simcse_unavailable(*_args, **_kwargs):
            raise ImportError(
                "The simcse package is not installed. ROSCOE scoring here runs in "
                "sentence_transformer mode (all-mpnet-base-v2); install simcse only "
                "if you need the sim_sce model type."
            )

        _stub.SimCSE = _simcse_unavailable
        _sys.modules["simcse"] = _stub

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
        # Filename format: {dataset}_{setting}.json  (setting = with_answer or wout_answer)
        stem = json_file.stem  # "drop_with_answer"
        if stem.endswith("_with_answer"):
            dataset = stem[: -len("_with_answer")]
            setting = "with_answer"
        elif stem.endswith("_wout_answer"):
            dataset = stem[: -len("_wout_answer")]
            setting = "wout_answer"
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
    """Print a comparison table: with_answer vs wout_answer per dataset and metric."""
    if not all_results:
        return

    settings = list(all_results.keys())
    datasets  = sorted({ds for s in all_results.values() for ds in s})
    all_metrics = sorted({m for s in all_results.values()
                          for ds_scores in s.values()
                          for m in ds_scores})

    print()
    print("=" * 90)
    print("  ROSCOE EVALUATION RESULTS  (with_answer vs wout_answer)")
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
