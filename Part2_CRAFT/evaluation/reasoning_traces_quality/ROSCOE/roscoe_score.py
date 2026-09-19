"""
roscoe_score.py — Score exported traces with ROSCOE, and report the comparison.

Part 2's copy: the two settings it compares are `raw` (the first of the K candidate
traces) and `craft` (the synthesized trace), not Part 1's w/ and w/o Answer. Copied
rather than imported so the two parts stay independent.

Imports ROSCOE's Evaluator out of a ParlAI checkout directly rather than shelling
out to its CLI, and runs it in sentence_transformer mode, so the checkout needs no
patching: the simcse import upstream insists on is stubbed out below.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ROSCOE's scorer lives in the ParlAI repository, not in the pip package, and the
# repository is far too large to vendor for two files. They are fetched on first
# use into the directory --roscoe_parlai_dir points at, which is gitignored: the
# scorer is upstream's code, pinned by URL rather than copied into this one.
# ---------------------------------------------------------------------------
_ROSCOE_RAW = "https://raw.githubusercontent.com/facebookresearch/ParlAI/main/projects/roscoe"
_ROSCOE_FILES = ("score.py", "utils.py")


def ensure_roscoe_sources(roscoe_dir: Path) -> None:
    """Make `projects.roscoe` importable from roscoe_dir's parent tree.

    Downloads score.py and utils.py when absent. Both are plain torch/numpy/nltk
    code — nothing in them needs ParlAI itself installed — and the __init__.py
    files are what let them win over the `projects` package that ships with the
    parlai wheel.
    """
    import urllib.request

    roscoe_dir = Path(roscoe_dir)
    missing = [f for f in _ROSCOE_FILES if not (roscoe_dir / f).exists()]
    if missing:
        roscoe_dir.mkdir(parents=True, exist_ok=True)
        for name in missing:
            url = f"{_ROSCOE_RAW}/{name}"
            logger.info("Fetching ROSCOE %s from ParlAI", name)
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    (roscoe_dir / name).write_bytes(r.read())
            except Exception as e:
                raise RuntimeError(
                    f"Could not fetch {url}: {e}\n"
                    f"Clone ParlAI and point --roscoe_parlai_dir at it instead."
                ) from e
    for pkg in (roscoe_dir, roscoe_dir.parent):
        init = pkg / "__init__.py"
        if not init.exists():
            init.touch()


def set_model_cache(cache_dir: str | None) -> None:
    """Point every model download at one directory, before anything loads a model.

    ROSCOE pulls roughly 5 GB — all-mpnet-base-v2 and its mpnet-base tokenizer,
    gpt2-large for perplexity, roberta-large-cola for grammar — and the libraries
    each read a different variable for where to put it. Setting them here keeps
    the weights off the machine's default cache and on whatever volume the run
    was given, which matters when the run is on a shared server.
    """
    if not cache_dir:
        return
    path = str(Path(cache_dir).expanduser().resolve())
    for var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
                "SENTENCE_TRANSFORMERS_HOME", "TORCH_HOME"):
        os.environ[var] = path
    logger.info("Model cache → %s", path)


def run_roscoe_evaluation(
    export_dir: str,
    roscoe_dir: str,
    transformer_model: str = "all-mpnet-base-v2",
    scores_output_dir: str = None,
    discourse_batch: int = 64,
    coherence_batch: int = 16,
    model_cache_dir: str = None,
    datasets: list = None,
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

    set_model_cache(model_cache_dir or os.environ.get("CRAFT_MODEL_CACHE"))
    roscoe_dir = Path(roscoe_dir).resolve()
    ensure_roscoe_sources(roscoe_dir)
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
    # .jsonl is what this pipeline writes; .json is what a ParlAI roscoe_data
    # checkout calls the same newline-delimited content.
    json_files = sorted(list(export_path.glob("*.jsonl")) + list(export_path.glob("*.json")))
    if datasets:
        wanted = set(datasets)
        json_files = [f for f in json_files
                      if f.stem.rsplit("_", 2)[0] in wanted]

    if not json_files:
        logger.warning("No .json files found in %s", export_path)
        return {}

    for json_file in json_files:
        fname = json_file.name  # e.g. "drop_craft.jsonl"
        # Parse dataset name and setting from filename
        # Filename format: {dataset}_{setting}.jsonl  (setting = raw or craft)
        stem = json_file.stem  # "drop_craft"
        if stem.endswith("_raw"):
            dataset = stem[: -len("_raw")]
            setting = "raw"
        elif stem.endswith("_craft"):
            dataset = stem[: -len("_craft")]
            setting = "craft"
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
        # Counting is not enough: an item whose reference text is missing yields an
        # empty chain, and the reference-based scores would then be computed against
        # nothing and reported as if they meant something. Every chain must hold text.
        has_refs = (len(refs) == len(hypotheses)
                    and all(getattr(r, "chain", None) for r in refs))
        if refs and not has_refs:
            logger.warning("%s/%s: %d of %d reference chains are empty — scoring "
                           "without the reference-based metrics",
                           dataset, setting,
                           sum(1 for r in refs if not getattr(r, "chain", None)), len(refs))
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


def merge_shards_when_complete(export_dir: Path) -> Path | None:
    """Fold the per-dataset summaries into one as soon as they are all present.

    Splitting a model across datasets means four jobs write four shards into one
    directory, and whichever finishes last can see that the set is complete and
    merge it — so a run ends with the summary it would have had if one job had
    done the work, and nobody has to remember a merge step. Concurrent jobs may
    both find the set complete and both merge: the inputs are the same, so the
    result is too, and the write goes through a temporary file to keep a reader
    from seeing a half-written one.
    """
    datasets = sorted({f.stem.rsplit("_", 2)[0]
                       for f in export_dir.glob("*_answer.jsonl")})
    if not datasets:
        return None
    shards = {ds: export_dir / f"evaluation_results.{ds}.json" for ds in datasets}
    if not all(p.exists() and p.stat().st_size for p in shards.values()):
        return None

    merged: dict = {"datasets": {}, "meta": {}}
    for ds, path in shards.items():
        part = json.loads(path.read_text(encoding="utf-8"))
        merged["datasets"].update(part.get("datasets", {}))
        merged["meta"] = part.get("meta", {})
    merged["meta"]["n_datasets"] = len(merged["datasets"])
    merged["meta"]["metrics"] = sorted({m for d in merged["datasets"].values()
                                        for side in d["metrics"].values() for m in side})

    out = export_dir / "evaluation_results.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out)
    for path in shards.values():
        path.unlink(missing_ok=True)
    return out


def build_evaluation_summary(scores: dict, export_dir: Path) -> dict:
    """Write what ROSCOE measured: each metric, per dataset, per setting.

    Nothing derived goes in here. Turning the two settings into the reported
    deltas is roscoe_build_table.py's job; the per-item scores stay in
    roscoe_scores/. What this adds to the scorer's own output is the item count
    each mean rests on.
    """
    settings = ("raw", "craft")
    datasets = sorted({ds for s in scores.values() for ds in s})

    per_dataset = {}
    for ds in datasets:
        counts = {}
        for setting in settings:
            path = export_dir / f"{ds}_{setting}.jsonl"
            # Traces carry curly quotes and the like; a machine whose locale is
            # ASCII decodes them only if the encoding is named.
            counts[setting] = (sum(1 for _ in path.open(encoding="utf-8"))
                               if path.exists() else 0)
        per_dataset[ds] = {
            "n_traces": counts,
            "metrics": {st: scores.get(st, {}).get(ds, {}) for st in settings},
        }

    return {
        "datasets": per_dataset,
        "meta": {
            "settings": list(settings),
            "n_datasets": len(datasets),
            "metrics": sorted({m for s in scores.values()
                               for ds_scores in s.values() for m in ds_scores}),
        },
    }


def print_roscoe_results(all_results: dict) -> None:
    """Print a comparison table: raw vs craft per dataset and metric."""
    if not all_results:
        return

    settings = list(all_results.keys())
    datasets  = sorted({ds for s in all_results.values() for ds in s})
    all_metrics = sorted({m for s in all_results.values()
                          for ds_scores in s.values()
                          for m in ds_scores})

    print()
    print("=" * 90)
    print("  ROSCOE EVALUATION RESULTS  (raw CoT vs CRAFT)")
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
# CLI — score traces that were generated earlier, without re-generating them.
# Generation needs an API and scoring needs ~5 GB of local models, so the two
# halves usually run on different machines; this entry point is the second half.
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Score ROSCOE exports (<run>/<dataset>_{raw,craft}.jsonl).")
    parser.add_argument("--export_dir", nargs="+", required=True,
                        help="One or more directories of exports. A relative path "
                             "resolves under this part's results root.")
    parser.add_argument("--roscoe_parlai_dir",
                        default=str(Path(__file__).resolve().parent / "ParlAI"),
                        help="Tree holding projects/roscoe/; fetched there if absent")
    parser.add_argument("--model_cache_dir", default=None,
                        help="Where the scoring models are downloaded "
                             "(default: CRAFT_MODEL_CACHE, else the HF cache)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Score only these datasets (default: all in the directory). "
                             "A subset writes evaluation_results.<dataset>.json, so "
                             "jobs splitting one model across datasets cannot overwrite "
                             "each other's summary; merge them once all have run.")
    parser.add_argument("--roscoe_model", default="all-mpnet-base-v2")
    parser.add_argument("--discourse_batch", type=int, default=64)
    parser.add_argument("--coherence_batch", type=int, default=16)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from config import RESULTS_ROOT
    except ImportError:
        RESULTS_ROOT = Path(__file__).resolve().parents[3] / "results"

    roscoe_dir = Path(args.roscoe_parlai_dir) / "projects" / "roscoe"
    for raw in args.export_dir:
        export_dir = Path(raw)
        if not export_dir.is_absolute() and not export_dir.exists():
            export_dir = RESULTS_ROOT / raw
        if not export_dir.exists():
            logger.error("No such export directory: %s", export_dir)
            continue

        logger.info("=== Scoring %s ===", export_dir)
        scores = run_roscoe_evaluation(
            export_dir=str(export_dir),
            roscoe_dir=str(roscoe_dir),
            transformer_model=args.roscoe_model,
            scores_output_dir=str(export_dir / "roscoe_scores"),
            discourse_batch=args.discourse_batch,
            coherence_batch=args.coherence_batch,
            model_cache_dir=args.model_cache_dir,
            datasets=args.datasets,
        )
        if not scores:
            continue
        print_roscoe_results(scores)
        suffix = f".{args.datasets[0]}" if args.datasets and len(args.datasets) == 1 else ""
        out = export_dir / f"evaluation_results{suffix}.json"
        # Build it before opening the file: opening for write truncates, so a
        # failure while building would leave an empty file that looks like a run
        # that produced nothing rather than one that crashed.
        summary = json.dumps(build_evaluation_summary(scores, export_dir),
                             indent=2, ensure_ascii=False)
        out.write_text(summary, encoding="utf-8")
        logger.info("Summary → %s", out)
        if suffix:
            merged = merge_shards_when_complete(export_dir)
            if merged:
                logger.info("All datasets scored — merged → %s", merged)


if __name__ == "__main__":
    main()
