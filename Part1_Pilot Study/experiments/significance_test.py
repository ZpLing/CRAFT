#!/usr/bin/env python3
"""
significance_test.py
====================
Paired Wilcoxon signed-rank tests + bootstrap 95% CI + Cohen's d
for PRMBench and ROSCOE GT vs No-GT conditions.

Outputs:
  - significance_testing_results.json — all stats, one file per benchmark per model
  - the LaTeX table, printed

Figures are drawn elsewhere from these numbers; this file only produces them.

Usage:
    python significance_test.py
"""

import json, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

BENCH_DIR    = Path(__file__).resolve().parent    # Part1_Pilot Study/experiments/
PART_ROOT    = BENCH_DIR.parent                   # Part1_Pilot Study/
REPO_ROOT    = PART_ROOT.parent
RESULTS_ROOT = PART_ROOT / "results"             # <results>/<model>/{prmbench,roscoe}/

# A model's stats live with that model's runs — <results>/<model>/ already holds
# everything else it produced — so there is no separate directory of aggregates to
# keep in step with them. The LaTeX table spans all four models and is paper
# material rather than a run output, so it is printed rather than written.


# Which sections of the stats belong to which benchmark. A model's results live
# in one directory per benchmark, and its significance stats land in the same
# place under one name, so a benchmark's evidence — the runs and the test over
# them — is never split across directories.
BENCHMARK_SECTIONS = {
    "prmbench": ("prmbench", "prmbench_metrics", "prmbench_dims_combined"),
    "roscoe":   ("roscoe", "roscoe_combined"),
}
STATS_FILENAME = "significance_testing_results.json"


def model_stats_path(model_name: str, benchmark: str) -> Path:
    """Where one model's stats for one benchmark land, beside that benchmark's runs."""
    return RESULTS_ROOT / MODELS[model_name] / benchmark / STATS_FILENAME


def split_by_model(stats: dict) -> dict:
    """Turn {section: {model: rows}} into {model: {benchmark: {section: rows}}}.

    A benchmark with nothing to report — ROSCOE before its traces are scored —
    is left out rather than written as an empty file that looks like a result.
    """
    per_model: dict[str, dict] = {}
    for benchmark, sections in BENCHMARK_SECTIONS.items():
        for section in sections:
            for model, rows in stats.get(section, {}).items():
                if not rows:
                    continue
                per_model.setdefault(model, {}).setdefault(benchmark, {})[section] = rows
    return per_model

# ──────────────────────────────────────────────────────────────
# Model config
# ──────────────────────────────────────────────────────────────
# Display name → the model's directory under the results root. Everything a model
# produced lives there: prmbench/<dim>_<api>_results.jsonl and roscoe/roscoe_scores/.
MODELS = {
    "GPT-o4-mini":           "o4-mini",
    "GPT-5.4-nano":          "gpt-5.4-nano",
    "DeepSeek-V4-Flash":     "deepseek-v4-flash",
    "Gemini-3.1-Flash-Lite": "gemini-3.1-flash-lite",
}

MODEL_COLORS = {
    "GPT-o4-mini":           "#B4DEB6",   # light sage
    "GPT-5.4-nano":          "#7BC6BE",   # mid teal
    "DeepSeek-V4-Flash":     "#439CC4",   # steel blue
    "Gemini-3.1-Flash-Lite": "#09554D",   # dark teal
}

DIM_MAP = {"simplicity": "Simplicity", "soundness": "Soundness",
           "sensitivity": "Sensitivity"}
ROSCOE_DATASETS = ["cosmos", "drop", "esnli", "gsm8k"]
ROSCOE_DS_LABELS = {"cosmos": "CosmosQA", "drop": "DROP",
                    "esnli": "eSNLI", "gsm8k": "GSM8K"}

PRM_METRIC    = "f1"            # primary PRMBench metric (kept for table)
ROSCOE_METRIC = "faithfulness"  # primary ROSCOE metric

# For the 7-panel plot
PRM_PLOT_METRICS = ["total_step_acc", "first_error_acc", "f1"]
PRM_METRIC_LABELS = {
    "total_step_acc":  "Step Acc",
    "first_error_acc": "1st Err Acc",
    "f1":              "F1",
}

# PRMBench dimensions for top-row panels
PRM_DIMS = ["simplicity", "soundness", "sensitivity"]
PRM_DIM_LABELS = {"simplicity": "Simplicity", "soundness": "Soundness",
                  "sensitivity": "Sensitivity"}

# ROSCOE metrics to aggregate per dataset (bottom-row panels)
ROSCOE_AGG_METRICS = ["faithfulness", "informativeness_step",
                      "informativeness_chain", "coherence_step_vs_step"]

# How each model is named in the LaTeX table; a model not listed keeps its full name.
MODEL_DISPLAY = {
    "GPT-o4-mini":           "o4-mini",
    "GPT-5.4-nano":          "GPT-5.4-nano",
    "DeepSeek-V4-Flash":     "DeepSeek-V4-Flash",
    "Gemini-3.1-Flash-Lite": "Gemini-3.1-Flash-Lite",
}


# ──────────────────────────────────────────────────────────────
# Statistics helpers
# ──────────────────────────────────────────────────────────────
def cohen_d(a, b):
    """Paired Cohen's d (mean diff / pooled SD of differences)."""
    diff = np.array(a) - np.array(b)
    return diff.mean() / (diff.std(ddof=1) + 1e-12)


def bootstrap_ci(a, b, n_boot=5000, ci=0.95, seed=42):
    """Bootstrap 95% CI on mean(a − b)."""
    rng  = np.random.default_rng(seed)
    diff = np.array(a) - np.array(b)
    boot = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    lo   = np.percentile(boot, 100 * (1 - ci) / 2)
    hi   = np.percentile(boot, 100 * (1 + ci) / 2)
    return float(diff.mean()), float(lo), float(hi)


def wilcoxon(a, b):
    """Wilcoxon signed-rank test; returns p-value (two-sided)."""
    diff = np.array(a) - np.array(b)
    if np.all(diff == 0):
        return 1.0
    try:
        _, p = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
        return float(p)
    except Exception:
        return 1.0


def fmt2(x, signed=True):
    """Two decimals, rounded — the table's number format.

    A value that rounds to zero is printed unsigned: at two decimals there is no
    direction left in it, and "$-0.00$" only looks like a typo.
    """
    if not signed or round(x, 2) == 0:
        return f"{abs(x):.2f}" if round(x, 2) == 0 else f"{x:.2f}"
    return f"{x:+.2f}"


def fmt_p(p):
    """Two decimals, except where that would print a real p-value as 0.00."""
    return "<0.01" if p < 0.005 else f"{p:.2f}"


def sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────
def prm_items(model_name):
    """Yield (dim, item) for one model's PRMBench runs.

    The dimension comes from the file the item was scored in — the dataset is split
    into simplicity/soundness/sensitivity.jsonl and each run writes back under that
    name — so this cannot drift from how the items were sampled or scored.
    """
    base = RESULTS_ROOT / MODELS[model_name] / "prmbench"
    for dim in ("simplicity", "soundness", "sensitivity"):
        sides = {}
        for setting in ("with_answer", "wout_answer"):
            path = base / f"{dim}_{setting}.jsonl"
            if not path.exists():
                continue
            with path.open() as f:
                sides[setting] = {json.loads(l)["idx"]: json.loads(l)
                                  for l in f if l.strip()}
        if len(sides) != 2:
            continue
        # The two settings are separate files; an item is whatever both hold for
        # the same idx, so a comparison never pairs one item with another's score.
        for idx, wa in sides["with_answer"].items():
            bl = sides["wout_answer"].get(idx)
            if bl is None:
                continue
            yield dim, {"idx": idx, "classification": wa.get("classification"),
                        "with_answer": wa, "wout_answer": bl}


def load_prm(model_name):
    """Load one model's PRMBench runs → dict[dim] → (with_answer, wout_answer)."""
    dims = {"simplicity": [], "soundness": [], "sensitivity": []}
    for dim, item in prm_items(model_name):
        wa = (item.get("with_answer") or {}).get("metrics") or {}
        bl = (item.get("wout_answer") or {}).get("metrics") or {}
        gt, no_gt = wa.get(PRM_METRIC), bl.get(PRM_METRIC)
        if gt is not None and no_gt is not None:
            dims[dim].append((gt, no_gt))
    # also build "total" = all items
    all_pairs = [p for ps in dims.values() for p in ps]
    dims["total"] = all_pairs
    return {d: ([x[0] for x in ps], [x[1] for x in ps]) for d, ps in dims.items() if ps}


def load_prm_metrics(model_name):
    """Load one model's PRMBench runs → dict[metric] → (gt, no_gt) across ALL items."""
    buckets = {m: [] for m in PRM_PLOT_METRICS}
    for _dim, item in prm_items(model_name):
        wa = (item.get("with_answer") or {}).get("metrics") or {}
        bl = (item.get("wout_answer")       or {}).get("metrics") or {}
        for m in PRM_PLOT_METRICS:
            g, b = wa.get(m), bl.get(m)
            if g is not None and b is not None:
                buckets[m].append((g, b))
    return {m: ([x[0] for x in ps], [x[1] for x in ps])
            for m, ps in buckets.items() if ps}


def load_roscoe(model_name):
    """Load per-item ROSCOE TSV scores → dict[dataset] → (gt_scores, wout_answer_scores)."""
    base = RESULTS_ROOT / MODELS[model_name] / "roscoe" / "roscoe_scores"
    result = {}
    for ds in ROSCOE_DATASETS:
        p_gt  = base / f"scores_{ds}_with_answer.tsv"
        p_bl  = base / f"scores_{ds}_wout_answer.tsv"
        if not p_gt.exists() or not p_bl.exists():
            continue
        df_gt = pd.read_csv(p_gt, sep=r"\s+", engine="python")
        df_bl = pd.read_csv(p_bl, sep=r"\s+", engine="python")
        if ROSCOE_METRIC not in df_gt.columns or ROSCOE_METRIC not in df_bl.columns:
            continue
        n = min(len(df_gt), len(df_bl))
        gt_raw = df_gt[ROSCOE_METRIC].values[:n]
        bl_raw = df_bl[ROSCOE_METRIC].values[:n]
        # Drop pairs where either value is NaN
        import numpy as _np
        mask = ~(_np.isnan(gt_raw) | _np.isnan(bl_raw))
        gt = gt_raw[mask].tolist()
        bl = bl_raw[mask].tolist()
        if len(gt) < 5:  # skip if too few valid pairs
            continue
        result[ds] = (gt, bl)
    return result


def load_prm_dims_combined(model_name):
    """Load PRMBench JSONL → dict[dim] → (gt_diffs_pooled, bl_diffs_pooled).

    For each dimension, pool paired differences across ALL 3 metrics
    (step_acc, first_error_acc, f1) so one panel = one dimension.
    """
    dims = {d: [] for d in PRM_DIMS}
    for dim, item in prm_items(model_name):
        wa = (item.get("with_answer") or {}).get("metrics") or {}
        bl = (item.get("wout_answer")       or {}).get("metrics") or {}
        for m in PRM_PLOT_METRICS:
            g, b = wa.get(m), bl.get(m)
            if g is not None and b is not None:
                dims[dim].append((g, b))
    return {d: ([x[0] for x in ps], [x[1] for x in ps])
            for d, ps in dims.items() if ps}


def load_roscoe_combined(model_name):
    """Load ROSCOE scores → dict[dataset] → (gt_pooled, bl_pooled).

    For each dataset, pool paired values across ROSCOE_AGG_METRICS
    so one panel = one dataset aggregating multiple metrics.
    """
    base = RESULTS_ROOT / MODELS[model_name] / "roscoe" / "roscoe_scores"
    result = {}
    for ds in ROSCOE_DATASETS:
        p_gt = base / f"scores_{ds}_with_answer.tsv"
        p_bl = base / f"scores_{ds}_wout_answer.tsv"
        if not p_gt.exists() or not p_bl.exists():
            continue
        df_gt = pd.read_csv(p_gt, sep=r"\s+", engine="python")
        df_bl = pd.read_csv(p_bl, sep=r"\s+", engine="python")
        n = min(len(df_gt), len(df_bl))
        gt_all, bl_all = [], []
        for metric in ROSCOE_AGG_METRICS:
            if metric not in df_gt.columns or metric not in df_bl.columns:
                continue
            gt_raw = df_gt[metric].values[:n]
            bl_raw = df_bl[metric].values[:n]
            mask = ~(np.isnan(gt_raw) | np.isnan(bl_raw))
            gt_all.extend(gt_raw[mask].tolist())
            bl_all.extend(bl_raw[mask].tolist())
        if len(gt_all) >= 5:
            result[ds] = (gt_all, bl_all)
    return result


# ──────────────────────────────────────────────────────────────
# Run all tests
# ──────────────────────────────────────────────────────────────
def _compute_stat(gt, bl, tag=""):
    mean_diff, lo, hi = bootstrap_ci(gt, bl)
    p = wilcoxon(gt, bl)
    d = cohen_d(gt, bl)
    if tag:
        print(f"  {tag}: diff={mean_diff:+.4f} [{lo:+.4f},{hi:+.4f}]"
              f" p={p:.4f} d={d:.4f} {sig_label(p)}")
    return {
        "n":          len(gt),
        "gt_mean":    round(float(np.mean(gt)), 4),
        "wout_answer_mean": round(float(np.mean(bl)), 4),
        "mean_diff":  round(mean_diff, 4),
        "ci_lo":      round(lo, 4),
        "ci_hi":      round(hi, 4),
        "p_wilcoxon": round(p, 4),
        "cohen_d":    round(d, 4),
        "sig":        sig_label(p),
    }


def run_all_tests():
    all_stats = {"prmbench": {}, "prmbench_metrics": {},
                 "prmbench_dims_combined": {}, "roscoe": {},
                 "roscoe_combined": {}}

    for model in MODELS:
        all_stats["prmbench"][model] = {}
        all_stats["prmbench_metrics"][model] = {}
        all_stats["prmbench_dims_combined"][model] = {}
        # --- per-dimension (for LaTeX table) ---
        try:
            for dim, (gt, bl) in load_prm(model).items():
                all_stats["prmbench"][model][dim] = _compute_stat(
                    gt, bl, f"PRMBench {model} {dim}")
        except Exception as e:
            print(f"  [PRMBench dim] {model}: {e}")
        # --- per-metric (for 7-panel plot, kept for reference) ---
        try:
            for metric, (gt, bl) in load_prm_metrics(model).items():
                all_stats["prmbench_metrics"][model][metric] = _compute_stat(
                    gt, bl, f"PRMBench {model} {metric}")
        except Exception as e:
            print(f"  [PRMBench metric] {model}: {e}")
        # --- per-dimension, all metrics pooled (for new plot) ---
        try:
            for dim, (gt, bl) in load_prm_dims_combined(model).items():
                all_stats["prmbench_dims_combined"][model][dim] = _compute_stat(
                    gt, bl, f"PRMBench {model} {dim} (combined)")
        except Exception as e:
            print(f"  [PRMBench dim combined] {model}: {e}")

    for model in MODELS:
        all_stats["roscoe"][model] = {}
        all_stats["roscoe_combined"][model] = {}
        try:
            for ds, (gt, bl) in load_roscoe(model).items():
                all_stats["roscoe"][model][ds] = _compute_stat(
                    gt, bl, f"ROSCOE   {model} {ds}")
        except Exception as e:
            print(f"  [ROSCOE] {model}: {e}")
        # --- per-dataset, all metrics pooled (for new plot) ---
        try:
            for ds, (gt, bl) in load_roscoe_combined(model).items():
                all_stats["roscoe_combined"][model][ds] = _compute_stat(
                    gt, bl, f"ROSCOE   {model} {ds} (combined)")
        except Exception as e:
            print(f"  [ROSCOE combined] {model}: {e}")

    return all_stats


# ──────────────────────────────────────────────────────────────
# LaTeX table  (compact, suitable for appendix)
# ──────────────────────────────────────────────────────────────
def make_latex_table(stats):
    lines = [
        r"\begin{table}[!ht]",
        r"\centering\small",
        r"\caption{Paired Wilcoxon signed-rank test results (GT vs.\ No-GT). "
        r"$\Delta$ = mean(GT$-$NoGT), 95\% CI via bootstrap, $d$ = Cohen's $d$.}",
        r"\label{tab:significance}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrrrl}",
        r"\toprule",
        r"Benchmark & Condition & $\Delta$ & CI$_{\text{lo}}$ & CI$_{\text{hi}}$ & $p$ & $d$ & \\",
        r"\midrule",
    ]
    for model in MODELS:
        short = MODEL_DISPLAY.get(model, model).replace("\n", "")
        # PRMBench total
        row = stats["prmbench"].get(model, {}).get("total")
        if row:
            lines.append(
                rf"\multirow{{5}}{{*}}{{\rotatebox{{90}}{{{short}}}}}"
                rf" & PRMBench (Total)"
                rf" & ${fmt2(row['mean_diff'])}$"
                rf" & ${fmt2(row['ci_lo'])}$"
                rf" & ${fmt2(row['ci_hi'])}$"
                rf" & ${fmt_p(row['p_wilcoxon'])}$"
                rf" & ${fmt2(row['cohen_d'])}$"
                rf" & {row['sig']} \\"
            )
        for dim in ["simplicity", "soundness", "sensitivity"]:
            row = stats["prmbench"].get(model, {}).get(dim)
            if row:
                lines.append(
                    rf" & \ \ {DIM_MAP[dim]}"
                    rf" & ${fmt2(row['mean_diff'])}$"
                    rf" & ${fmt2(row['ci_lo'])}$"
                    rf" & ${fmt2(row['ci_hi'])}$"
                    rf" & ${fmt_p(row['p_wilcoxon'])}$"
                    rf" & ${fmt2(row['cohen_d'])}$"
                    rf" & {row['sig']} \\"
                )
        # ROSCOE avg
        roscoe_rows = [stats["roscoe"].get(model, {}).get(ds) for ds in ROSCOE_DATASETS]
        roscoe_rows = [r for r in roscoe_rows if r]
        if roscoe_rows:
            avg_diff = np.mean([r["mean_diff"] for r in roscoe_rows])
            avg_p    = np.mean([r["p_wilcoxon"] for r in roscoe_rows])
            avg_d    = np.mean([r["cohen_d"] for r in roscoe_rows])
            lines.append(
                rf" & ROSCOE (avg Faith.)"
                rf" & ${fmt2(avg_diff)}$"
                rf" & —"
                rf" & —"
                rf" & ${fmt_p(avg_p)}$"
                rf" & ${fmt2(avg_d)}$"
                rf" & {sig_label(avg_p)} \\"
            )
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running significance tests...\n")
    stats = run_all_tests()

    # One JSON per model per benchmark, beside that benchmark's runs
    print()
    written = split_by_model(stats)
    for model, by_benchmark in written.items():
        for benchmark, sections in by_benchmark.items():
            out = model_stats_path(model, benchmark)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(sections, f, indent=2)
            print(f"Saved: {out}")
    for benchmark in BENCHMARK_SECTIONS:
        if not any(benchmark in b for b in written.values()):
            print(f"No {benchmark} statistics — its runs have not been scored yet.")

    # The LaTeX table is printed, not written: it is one rendering of the stats
    # above for whoever is editing the paper, and a copy on disk would be a third
    # place for the same numbers to go stale.
    print("\nLaTeX table:\n")
    print(make_latex_table(stats))
