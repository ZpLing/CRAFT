#!/usr/bin/env python
"""
finelogic_build_table.py — raw CoT vs CRAFT under FineLogic.

Reads the two detailed_*.json files finelogic_eval_steps.py writes and produces:
  - overall step-weighted: valid / necessary / atomic
  - sample-level perfection: all_valid / all_necessary / all_atomic / all_three
  - same metrics restricted to FLD samples with original_data.steps in [10, 20]
    (the subset FineLogic reports on in Table 3)
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
try:
    from config import resolve_input
except ImportError:
    resolve_input = Path


def aggregate(samples, gt_steps_filter=None):
    tot = Counter(valid=0, necessary=0, atomic=0, total=0)
    perf = Counter(all_valid=0, all_necessary=0, all_atomic=0, all_three=0, n=0)

    for samp in samples:
        if samp is None or samp.get("error"):
            continue
        gt = samp.get("ground_truth_steps")
        if gt_steps_filter is not None:
            try:
                gt_int = int(gt)
            except (TypeError, ValueError):
                continue
            lo, hi = gt_steps_filter
            if not (lo <= gt_int <= hi):
                continue
        non_skip = [st for st in samp.get("steps", []) if not st.get("skip")]
        if not non_skip:
            continue
        perf["n"] += 1
        if all(st["valid"] for st in non_skip):
            perf["all_valid"] += 1
        if all(st["necessary"] for st in non_skip):
            perf["all_necessary"] += 1
        if all(st["atomic"] for st in non_skip):
            perf["all_atomic"] += 1
        if (all(st["valid"] for st in non_skip)
                and all(st["necessary"] for st in non_skip)
                and all(st["atomic"] for st in non_skip)):
            perf["all_three"] += 1
        tot["valid"]     += sum(st["valid"]     for st in non_skip)
        tot["necessary"] += sum(st["necessary"] for st in non_skip)
        tot["atomic"]    += sum(st["atomic"]    for st in non_skip)
        tot["total"]     += len(non_skip)

    n = perf["n"] or 1
    t = tot["total"] or 1
    return {
        "n_samples":          perf["n"],
        "n_steps":            tot["total"],
        "step_valid":         round(tot["valid"]     / t * 100, 1),
        "step_necessary":     round(tot["necessary"] / t * 100, 1),
        "step_atomic":        round(tot["atomic"]    / t * 100, 1),
        "sample_all_valid":     round(perf["all_valid"]     / n * 100, 1),
        "sample_all_necessary": round(perf["all_necessary"] / n * 100, 1),
        "sample_all_atomic":    round(perf["all_atomic"]    / n * 100, 1),
        "sample_all_three":     round(perf["all_three"]     / n * 100, 1),
    }


def show(rows, headers, label):
    print(f"\n{'='*100}")
    print(f"  {label}")
    print('='*100)
    col_widths = [max(len(str(r[i])) for r in [headers] + rows) + 2 for i in range(len(headers))]
    for r in [headers] + rows:
        print("".join(str(c).ljust(w) for c, w in zip(r, col_widths)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="detailed_*.json from the raw-CoT run")
    ap.add_argument("--craft", required=True, help="detailed_*.json from the CRAFT-trace run")
    ap.add_argument("--label", default="FLD")
    args = ap.parse_args()

    base = json.load(open(resolve_input(args.raw)))
    ours = json.load(open(resolve_input(args.craft)))

    headers = ["Subset", "n_samp", "n_steps",
               "step.V%", "step.N%", "step.A%",
               "samp.AllV%", "samp.AllN%", "samp.AllA%", "samp.All3%"]

    subsets = [
        ("ALL",     None),
        ("[10-20]", (10, 20)),
        ("[1-9]",   (1, 9)),
        ("[5-15]",  (5, 15)),
    ]
    rows = []
    for subset_name, subset_filter in subsets:
        for tag, data in [("BEFORE", base), ("AFTER", ours)]:
            a = aggregate(data, gt_steps_filter=subset_filter)
            rows.append([f"{subset_name}_{tag}", a["n_samples"], a["n_steps"],
                         a["step_valid"], a["step_necessary"], a["step_atomic"],
                         a["sample_all_valid"], a["sample_all_necessary"],
                         a["sample_all_atomic"], a["sample_all_three"]])

    show(rows, headers, f"BEFORE vs AFTER on {args.label} (CRAFT pipeline)")
    print("\nAllV / AllN / AllA are the paper's All Valid / All Relevant / All Atomic.")
    print("Note: FineLogic's Table 3 reports SAMPLE-level perfection rates (samp.AllV%, AllN%, AllA%) on FLD steps∈[10,20].")
    print("      Compare the 'samp.*' columns under the [10-20] subset rows against their paper.")


if __name__ == "__main__":
    main()
