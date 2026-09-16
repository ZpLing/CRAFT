#!/usr/bin/env python3
"""
rebuild_consensus.py
────────────────────
Rebuild consensus_dag in an existing rkg JSON file using the current
build_consensus_rkg implementation (no LLM calls — uses cached trace_dags).
"""
import argparse, json, copy, sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from config import resolve_input
import module2_rkg_filtering.build_rkg as _b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="rkg JSON to rewrite in place; relative paths resolve under the results root")
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--proved_threshold", type=float, default=None,
                   help="Asymmetric voting τ; predict PROVED if PROVED-weight ratio ≥ τ. Default uses plain MV.")
    p.add_argument("--weight_by", choices=["uniform","step_count"], default="uniform",
                   help="Trace weighting scheme. 'step_count' favors longer/more-thorough traces.")
    p.add_argument("--gt_file", help="cleaned_with_problem.json for accuracy diagnostics")
    args = p.parse_args()
    args.input = str(resolve_input(args.input))
    if args.gt_file:
        args.gt_file = str(resolve_input(args.gt_file))

    with open(args.input) as f:
        rd = json.load(f)
    res = rd.get("results", rd) if isinstance(rd, dict) else rd

    gt_map = {}
    if args.gt_file and Path(args.gt_file).exists():
        with open(args.gt_file) as f:
            cd = json.load(f)
        gt_map = {s["sample_id"]: s.get("target_answer") for s in cd.get("results", cd)}

    correct = 0; total = 0; matrix = Counter(); no_conc = 0
    for r in res:
        td = r.get("trace_dags", [])
        valid = [copy.deepcopy(t) for t in td if t.get("extraction_method") != "error"]
        if not valid:
            r["consensus_dag"] = {"nodes": [], "edges": []}
            continue
        new_consensus = _b.build_consensus_rkg(valid, consensus_threshold=args.threshold,
                                                proved_threshold=args.proved_threshold,
                                                weight_by=args.weight_by)
        r["consensus_dag"] = new_consensus

        # diagnostic
        if gt_map:
            label = None
            for n in new_consensus.get("nodes", []):
                if n.get("type") == "conclusion":
                    t = n.get("text", "")
                    if "__PROVED__" in t: label = "__PROVED__"; break
                    if "__DISPROVED__" in t: label = "__DISPROVED__"; break
            sid = r.get("sample_id"); gt = gt_map.get(sid)
            if not label:
                no_conc += 1
            elif gt:
                total += 1
                if label == gt: correct += 1
                matrix[(label, gt)] += 1

    # write back
    with open(args.input, "w") as f:
        json.dump({"results": res} if isinstance(rd, dict) and "results" in rd else res, f)
    print(f"Wrote: {args.input}")
    if total:
        print(f"Conclusion-node accuracy: {correct}/{total} = {correct/total:.3f}")
        print(f"No conclusion: {no_conc}")
        print(f"Matrix: {dict(matrix)}")


if __name__ == "__main__":
    main()
