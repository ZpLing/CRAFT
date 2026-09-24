#!/usr/bin/env python3
"""
ablation_study.py — the ablation table of §4, one row per removed component.

    Setting                      what it removes            read from
    ---------------------------------------------------------------------------
    CRAFT (full)                 nothing                    the run's synthesized trace
    w/o CRAFT                    the whole pipeline         a single-call run
    w/o RKG                      Module II's graph          --synthesis_strategy step_by_step
    w/o Weighted Edges Fusion    the lambda term in W(e)    build_rkg --edge_lambda 0
    Embedding Cosine Similarity  Jaccard in the edge weight an embedding-similarity run
    w/o Rollout (K=1)            the other K-1 traces       the pipeline run on one trace
    w/o Steps Filtering          Module I's z-score filter  steps_filter --z_score_threshold -1000
    w/o Edge & Node Filtering    Module II's thresholds     build_rkg --consensus_threshold 0 --node_threshold 0

Scoring is not reimplemented here: the rows are read with the same loaders and
scored with the same metric as the main table, so an ablation row and a main-table
cell can never disagree about what a run achieved.

Three settings need their own run and are passed in with --variant NAME=PATH.
Settings that were not run are printed as absent, so a partial table cannot be
mistaken for a complete one.

'w/o CRAFT' needs no run at all. Removing the whole pipeline leaves one call to
the backbone, which is what the baselines' zero-shot CoT setting already is over
the same samples and the same model, so --baseline_cot reads that run's
traces.jsonl and writes the row from it rather than spending the calls again to
get the same thing, then scores that file like every other row. Only the trace text crosses over: the label is re-derived
with the extractor every other row is scored with, so a baseline and an ablation
row that disagree are disagreeing about the run and not about how two scorers
read a \\boxed{}. Samples the CRAFT run does not cover are dropped, and samples
it covers that the baseline is missing are reported rather than silently
shortening the row. --zero_shot takes a file already in that shape instead.

The results are laid out the way the table reads: one folder per backbone, one
folder per setting inside it, and in that folder one file per dataset,
CRAFT_results/Ablation_Study/<model>/<setting>/<dataset>.json, each carrying the
row's score, the file it was scored from and its change from the full pipeline.
A setting that was not run on a dataset simply has no file there.

Usage:
    python ablation_study.py --craft_dir craft_runs/olympiad_gemini \\
        --variant "w/o RKG=craft_runs/olympiad_gemini/synthesized_step_by_step.json" \\
        --variant "w/o Weighted Edges Fusion=craft_runs/olympiad_gemini_lam0/synthesized.json" \\
        --baseline_cot results/baseline_results/gemini-3.1-flash-lite/cot/traces.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from config import PART_ROOT, resolve_input, resolve_output, run_model
except ImportError:
    resolve_input = resolve_output = Path
    PART_ROOT = Path(__file__).resolve().parents[2]

    def run_model(*paths):  # noqa: D103
        return "unknown-model"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "evaluation" / "label_prediction"))
from evaluate_accuracy import LOADERS, compute_metrics

FULL = "CRAFT (full)"
ROLLOUT = "w/o Rollout (K=1)"
VARIANT_ROWS = ("w/o RKG", "w/o Weighted Edges Fusion", "Embedding Cosine Similarity",
                ROLLOUT, "w/o Steps Filtering", "w/o Edge & Node Filtering")
# The single-trace row used to be called "w/o Consensus"; the old name is still accepted.
ALIASES = {"w/o Consensus (K=1)": ROLLOUT}
ROW_ORDER = (FULL, "w/o CRAFT", ROLLOUT, "w/o Steps Filtering", "w/o RKG",
             "w/o Edge & Node Filtering",
             "w/o Weighted Edges Fusion", "Embedding Cosine Similarity")


def slug(setting: str) -> str:
    """A setting's folder name: 'w/o Edge & Node Filtering' -> 'wout_Edge_and_Node_Filtering'."""
    s = setting.replace("w/o", "wout").replace("&", "and").replace("=", "")
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def find_one(run_dir: Path, pattern: str) -> Optional[Path]:
    hits = sorted(run_dir.glob(pattern))
    return hits[0] if hits else None


def rel(path: Path) -> str:
    """A run path as the repo spells it, so a tracked table stays followable.

    The absolute path is right on the machine that produced the file and wrong
    everywhere else, and it carries a local username into a tracked result. A
    path under the part root is written relative to it; anything outside is left
    alone rather than guessed at.
    """
    try:
        return str(Path(path).resolve().relative_to(Path(PART_ROOT).resolve()))
    except (ValueError, TypeError):
        return str(path)


def build_zero_shot(baseline: Path, covered: Path, out: Path) -> List[str]:
    """Write the 'w/o CRAFT' row from the baseline run that already exists.

    It writes a file rather than scoring in memory so that row goes through
    load_synthesized like every other one: the moment this scored its own rows,
    the table would hold one number produced by a second reader of a trace, and
    a disagreement with the baselines could no longer be pinned on the run.

    Returns the sample ids the CRAFT run covers that the baseline does not, so a
    row resting on fewer problems than the rows above it says so instead of
    quietly averaging over a different set.
    """
    like = json.loads(Path(covered).read_text(encoding="utf-8"))
    like = like.get("results", like) if isinstance(like, dict) else like
    wanted = {r.get("sample_id") for r in like}

    rows: List[Dict] = []
    seen = set()
    with Path(baseline).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec.get("sample_id")
            if sid not in wanted or sid in seen:
                continue
            seen.add(sid)
            traces = rec.get("traces") or []
            rows.append({"sample_id": sid,
                         "source_dataset": rec.get("source_dataset"),
                         "domain": rec.get("domain"),
                         "ground_truth": rec.get("ground_truth"),
                         "synthesized_trace": traces[0] if traces else ""})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return sorted(wanted - seen)


def paired(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Accuracy of two rows on the problems both of them scored.

    A variant run that failed on some samples has a smaller denominator than
    the full run, and the two accuracies then move for a reason that has
    nothing to do with the component. The change a row reports is therefore
    taken on the common problems, and the count is written next to it.
    """
    ia = {s["sample_id"]: s for s in a if str(s.get("ground_truth") or "").strip()}
    ib = {s["sample_id"]: s for s in b if str(s.get("ground_truth") or "").strip()}
    common = [i for i in ia if i in ib]
    if not common:
        return {"n_common": 0}
    ma = compute_metrics([ia[i] for i in common])
    mb = compute_metrics([ib[i] for i in common])
    return {"n_common": len(common),
            "accuracy_common": round(100.0 * ma["accuracy"], 1),
            "full_accuracy_common": round(100.0 * mb["accuracy"], 1),
            "delta_paired": round(100.0 * (ma["accuracy"] - mb["accuracy"]), 1)}


def score(path: Path, source: str) -> Dict[str, Any]:
    """One row, scored exactly as evaluate_accuracy would score it.

    n_samples is the row's own denominator, not the slice's size. They are not
    always the same number: a sample whose consensus RKG came out empty has no
    graph for Module III to walk and the pipeline drops it, so a synthesis row
    can rest on fewer samples than a row that reads the traces directly and
    keeps all 500. A drop of a tenth of a point across
    denominators that differ by five is inside that difference, so the column
    carries the count rather than leaving it to be assumed.
    """
    samples = LOADERS[source](path)
    metrics = compute_metrics(samples)
    return {"accuracy": round(100.0 * metrics["accuracy"], 1),
            "macro_f1": metrics["macro_f1"],
            "avg_steps": metrics.get("avg_steps"),
            "n_samples": metrics.get("n_total"),
            "_samples": samples}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--craft_dir", required=True,
                    help="The full CRAFT run: k_traces, cleaned traces, synthesized trace")
    ap.add_argument("--baseline_cot", default=None,
                    help="'w/o CRAFT': traces.jsonl from the baselines' zero-shot CoT "
                         "setting for this model. The row is built from it, so the "
                         "single call is not paid for twice")
    ap.add_argument("--zero_shot", default=None,
                    help="'w/o CRAFT' from a file already in the synthesized schema, "
                         "instead of --baseline_cot. Given with --baseline_cot it is "
                         "where that row is written; the default is "
                         "zero_shot_from_cot.json in --craft_dir. Without either, the "
                         "row is absent")
    ap.add_argument("--variant", action="append", default=[], metavar="NAME=PATH",
                    help=f"Synthesis output for one of: {', '.join(VARIANT_ROWS)}. Repeatable")
    ap.add_argument("--synth_file", default=None,
                    help="The full run's synthesis output (default: synthesized*.json in --craft_dir)")
    ap.add_argument("--dataset", default=None,
                    help="The dataset's name in the file names (default: the run "
                         "directory's name up to its first underscore)")
    ap.add_argument("--output", default=None,
                    help="Root of the ablation results, default CRAFT_results/Ablation_Study "
                         "under the results root; each row is written to "
                         "<root>/<model>/<setting>/<dataset>.json with the model read from "
                         "the run's own metadata. The old per-dataset file "
                         "<root>/<model>/<dataset>.json is also accepted here and read as "
                         "that root, model and dataset")
    args = ap.parse_args()

    run_dir = Path(resolve_input(args.craft_dir))
    k_path = find_one(run_dir, "k_traces_*_samples.json")
    if k_path is None:
        raise FileNotFoundError(f"No k_traces_*_samples.json in {run_dir}")
    synth_path = (Path(resolve_input(args.synth_file)) if args.synth_file
                  else find_one(run_dir, "synthesized*.json"))
    if synth_path is None:
        raise FileNotFoundError(f"No synthesized*.json in {run_dir}")

    rows: Dict[str, Dict[str, Any]] = {FULL: score(synth_path, "synthesized")}
    sources: Dict[str, Path] = {FULL: synth_path}
    if args.baseline_cot:
        # Not zero_shot.json: several runs already carry a file by that name,
        # and a default that silently overwrites one of a run's own inputs is a
        # default that loses data the first time it is used.
        zs_path = Path(resolve_input(args.zero_shot)) if args.zero_shot \
            else run_dir / "zero_shot_from_cot.json"
        missing = build_zero_shot(Path(resolve_input(args.baseline_cot)),
                                  synth_path, zs_path)
        rows["w/o CRAFT"] = score(zs_path, "synthesized")
        sources["w/o CRAFT"] = zs_path
        if missing:
            print(f"  w/o CRAFT: {len(missing)} of the run's samples are absent from "
                  f"the baseline and are left out of that row: {missing[:5]}")
    elif args.zero_shot:
        sources["w/o CRAFT"] = Path(resolve_input(args.zero_shot))
        rows["w/o CRAFT"] = score(sources["w/o CRAFT"], "synthesized")
    for spec in args.variant:
        if "=" not in spec:
            raise SystemExit(f"--variant takes NAME=PATH, got {spec!r}")
        # Split on the last "=" so a variant name may itself contain one ("w/o Rollout (K=1)").
        name, raw = (s.strip() for s in spec.rsplit("=", 1))
        name = ALIASES.get(name, name)
        if name not in VARIANT_ROWS:
            raise SystemExit(f"Unknown variant {name!r}; expected one of {VARIANT_ROWS}")
        sources[name] = Path(resolve_input(raw))
        rows[name] = score(sources[name], "synthesized")

    full_acc = rows[FULL]["accuracy"]
    print()
    print(f"  {'Ablation Setting':<32} {'Acc(%)':>8} {'d(%)':>8} {'steps':>7} {'n':>6}")
    print("  " + "-" * 65)
    for name in ROW_ORDER:
        row = rows.get(name)
        if row is None:
            print(f"  {name:<32} {'not run':>8}")
            continue
        delta = "" if name == FULL else f"{row['accuracy'] - full_acc:+.1f}"
        steps = f"{row['avg_steps']:.1f}" if row.get("avg_steps") else "-"
        n = row.get("n_samples")
        print(f"  {name:<32} {row['accuracy']:>8.1f} {delta:>8} {steps:>7} "
              f"{n if n is not None else '-':>6}")
    print()
    absent = [n for n in ROW_ORDER if n not in rows]
    if absent:
        print("  Not run: " + ", ".join(absent))
        print("  w/o RKG: synthesize with --synthesis_strategy step_by_step.")
        print("  w/o Weighted Edges Fusion: build_rkg --edge_lambda 0, then synthesize.")
        print("  w/o CRAFT: pass --baseline_cot (or --zero_shot).")

    model = run_model(synth_path, k_path)
    dataset = args.dataset or run_dir.name.split("_")[0]
    root = Path(resolve_output(args.output or "CRAFT_results/Ablation_Study"))
    if root.suffix == ".json":
        # The old per-dataset file <root>/<model>/<dataset>.json names all three.
        dataset, model, root = root.stem, root.parent.name, root.parent.parent
    for name in ROW_ORDER:
        if name not in rows:
            continue
        out = root / model / slug(name) / f"{dataset}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        row = {k: v for k, v in rows[name].items() if k != "_samples"}
        pair = {} if name == FULL else paired(rows[name]["_samples"], rows[FULL]["_samples"])
        out.write_text(json.dumps(
            {"model": model, "dataset": dataset, "setting": name,
             "source_file": rel(sources[name]), "craft_dir": rel(run_dir),
             **row, "full_accuracy": full_acc,
             "delta": None if name == FULL else round(rows[name]["accuracy"] - full_acc, 1),
             **pair},
            indent=2), encoding="utf-8")
    print(f"  Saved: {root / model}/<setting>/{dataset}.json for "
          f"{sum(n in rows for n in ROW_ORDER)} settings")


if __name__ == "__main__":
    main()
