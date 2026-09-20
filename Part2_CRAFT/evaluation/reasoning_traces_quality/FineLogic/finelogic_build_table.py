#!/usr/bin/env python
"""
finelogic_build_table.py — raw CoT vs CRAFT under FineLogic.

Reads the two detailed_*.json files finelogic_eval_steps.py writes and produces:
  - overall step-weighted: valid / necessary / atomic
  - sample-level perfection: all_valid / all_necessary / all_atomic / all_three
  - the same metrics restricted to a band of gold step counts, which is what
    FineLogic's own Table 3 reports on

Only a dataset that annotates a gold proof length has bands at all: FLD's run
1-7 and ProofWriter's are 5, while neither mathematical set annotates one, so
their samples appear in the overall row and in no band. FineLogic's own band,
[10, 20], is empty on this benchmark; --bands says which to report.

A band holding no samples prints as an em dash. It used to print 0.0 for every
metric, which reads exactly like a band where nothing was valid.
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
try:
    from config import resolve_input
except ImportError:
    resolve_input = Path


def parse_band(spec):
    """'10-20' -> (10, 20)."""
    try:
        lo, hi = (int(x) for x in spec.split("-", 1))
    except ValueError:
        raise SystemExit(f"--bands takes LO-HI, got {spec!r}")
    return lo, hi


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

    # No samples means no measurement, not a measurement of zero.
    def per_step(key):
        return round(tot[key] / tot["total"] * 100, 1) if tot["total"] else None

    def per_sample(key):
        return round(perf[key] / perf["n"] * 100, 1) if perf["n"] else None

    return {
        "n_samples":            perf["n"],
        "n_steps":              tot["total"],
        "step_valid":           per_step("valid"),
        "step_necessary":       per_step("necessary"),
        "step_atomic":          per_step("atomic"),
        "sample_all_valid":     per_sample("all_valid"),
        "sample_all_necessary": per_sample("all_necessary"),
        "sample_all_atomic":    per_sample("all_atomic"),
        "sample_all_three":     per_sample("all_three"),
    }


def cell(value):
    return "—" if value is None else value


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
    ap.add_argument("--bands", nargs="*", default=["1-9", "10-20"], metavar="LO-HI",
                    help="Gold-step-count bands to report beside the overall row "
                         "(default: 1-9 10-20). Pass none to report only overall")
    args = ap.parse_args()

    base = json.load(open(resolve_input(args.raw)))
    ours = json.load(open(resolve_input(args.craft)))

    headers = ["Subset", "n_samp", "n_steps",
               "step.V%", "step.N%", "step.A%",
               "samp.AllV%", "samp.AllN%", "samp.AllA%", "samp.All3%"]

    subsets = [("ALL", None)]
    subsets += [(f"[{b}]", parse_band(b)) for b in args.bands]

    rows = []
    for subset_name, subset_filter in subsets:
        for tag, data in [("BEFORE", base), ("AFTER", ours)]:
            a = aggregate(data, gt_steps_filter=subset_filter)
            rows.append([f"{subset_name}_{tag}", a["n_samples"], a["n_steps"],
                         cell(a["step_valid"]), cell(a["step_necessary"]),
                         cell(a["step_atomic"]), cell(a["sample_all_valid"]),
                         cell(a["sample_all_necessary"]), cell(a["sample_all_atomic"]),
                         cell(a["sample_all_three"])])

    show(rows, headers, f"BEFORE vs AFTER on {args.label} (CRAFT pipeline)")
    print("\nAllV / AllN / AllA are the paper's All Valid / All Relevant / All Atomic.")
    print("A band with no samples prints —: only FLD and ProofWriter annotate a gold")
    print("step count, and FLD's run 1-7, so FineLogic's own [10,20] band is empty here.")


if __name__ == "__main__":
    main()
