#!/usr/bin/env python3
"""One table, one formula: CRAFT against every baseline on the same samples.

The numbers in this project have been compared across three different
denominators. The logical scorer balances to 125 per class and reports 250; a
CRAFT run drops the samples Module III could not produce a trace for and
reports 496; a baseline keeps a sample it could not read an answer from, counts
it wrong, and reports 500. Read side by side those are not comparisons, and the
sign of a 0.1 gap is decided by which of them a cell happened to use.

So this computes every cell the same way:

  * the answer comparison is the dataset's own adapter from answer_match, the
    same one the scorer uses -- match_label for FLD and ProofWriter, symbolic
    equivalence for OlympiadBench and Omni-MATH;
  * a method scores on the sample ids it shares with CRAFT for that cell, so
    both sides answer the same questions and neither is charged for a sample
    the other never saw;
  * a sample a method has no answer for stays in its denominator and counts
    against it, because refusing to answer is not the same as not being asked.

Which CRAFT file a cell uses is read from
results/CRAFT_results/label_prediction/<model>/<dataset>_results.json, so the
table follows whatever the pipeline last wrote rather than a path repeated here.

    python sota_table.py                      # the full table
    python sota_table.py --models gpt-5.4-nano
    python sota_table.py --baseline self_consistency
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from answer_match import ADAPTERS          # noqa: E402
from extract_label import extract_pred     # noqa: E402

_P2 = _HERE.parents[1]
CRAFT_DIR    = _P2 / "results" / "CRAFT_results" / "label_prediction"
BASELINE_DIR = _P2 / "results" / "baseline_results"

MODELS   = ["gemini-3.1-flash-lite", "gpt-5.4-nano"]
DATASETS = ["FLD", "ProofWriter", "OmniMATH", "OlympiadBench"]
DOMAIN   = {"FLD": "logical", "ProofWriter": "logical",
            "OmniMATH": "math", "OlympiadBench": "math"}


def _rows(obj: Any) -> List[Dict]:
    return obj["results"] if isinstance(obj, dict) and "results" in obj else obj


def _load(path: Path) -> List[Dict]:
    return _rows(json.loads(path.read_text(encoding="utf-8")))


def craft_predictions(model: str, dataset: str
                      ) -> Tuple[Dict[str, Optional[str]], Path, List[str]]:
    """{sample_id: predicted answer} for the file this cell currently reports.

    Also returns the samples the pipeline refused rather than answered. A
    sample whose consensus RKG came out empty has no graph for Module III to
    walk, and the pipeline's rule is to report the error and drop it rather
    than fall back to a graph-free strategy. Dropping it from CRAFT alone would
    hand CRAFT an easier paper, so the caller drops it from every method.
    """
    meta_path = CRAFT_DIR / model / f"{dataset}_results.json"
    if not meta_path.exists():
        meta_path = CRAFT_DIR / model / f"{dataset}_500.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    synth = Path(meta["file"])
    if not synth.is_absolute():
        synth = _P2 / synth
    domain = DOMAIN[dataset]
    out: Dict[str, Optional[str]] = {}
    refused: List[str] = []
    for r in _load(synth):
        sid = r.get("sample_id")
        if sid is None:
            continue
        if r.get("error"):
            refused.append(sid)
            continue
        out[sid] = extract_pred(r.get("synthesized_trace", ""), domain)
    return out, synth, refused


def gold_map(model: str, dataset: str) -> Dict[str, Tuple[str, Optional[str]]]:
    """{sample_id: (gold, answer_type)} taken from the run's own k_traces."""
    meta_path = CRAFT_DIR / model / f"{dataset}_results.json"
    if not meta_path.exists():
        meta_path = CRAFT_DIR / model / f"{dataset}_500.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    run = Path(meta["file"]).parent
    if not run.is_absolute():
        run = _P2 / run
    kt = next(iter(sorted(run.glob("k_traces*.json"))), None)
    if kt is None:
        return {}
    return {r["sample_id"]: (r.get("target_answer"), r.get("answer_type"))
            for r in _load(kt) if r.get("target_answer") is not None}


def baseline_predictions(model: str, dataset: str) -> Dict[str, Dict[str, Optional[str]]]:
    """{setting: {sample_id: predicted}} for every baseline that ran this cell."""
    out: Dict[str, Dict[str, Optional[str]]] = {}
    root = BASELINE_DIR / model
    if not root.is_dir():
        return out
    for setting_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        res = setting_dir / "results.json"
        if not res.exists():
            continue
        try:
            obj = json.loads(res.read_text(encoding="utf-8"))
        except Exception:
            continue
        preds = {}
        for r in obj.get("predictions", []):
            if r.get("source_dataset", "").replace(".json", "") != dataset:
                continue
            preds[r["sample_id"]] = r.get("predicted")
        if preds:
            out[setting_dir.name] = preds
    return out


def _hits(preds: Dict[str, Optional[str]], gold: Dict[str, Tuple[str, Optional[str]]],
          ids: List[str], dataset: str) -> List[bool]:
    match = ADAPTERS[dataset]
    out = []
    for sid in ids:
        p = preds.get(sid)
        g, at = gold[sid]
        if not p:
            out.append(False)             # no answer counts against the method
            continue
        try:
            out.append(bool(match(p, g, at)))
        except Exception:
            out.append(False)
    return out


def score(preds: Dict[str, Optional[str]], gold: Dict[str, Tuple[str, Optional[str]]],
          ids: List[str], dataset: str) -> float:
    h = _hits(preds, gold, ids, dataset)
    return sum(h) / len(h) if h else float("nan")


def mcnemar(a: List[bool], b: List[bool]) -> Tuple[int, int, float]:
    """Exact two-sided McNemar on the samples the two methods disagree about.

    A gap of a few samples on 496 is inside the standard error of either
    method, so a table that ranks cells by the gap alone will rank noise. This
    reports the discordant pairs it rests on and the p-value, which is the only
    honest way to write "ahead" next to a difference of four samples.
    """
    from math import comb
    b_only = sum(1 for x, y in zip(a, b) if x and not y)
    c_only = sum(1 for x, y in zip(a, b) if y and not x)
    n = b_only + c_only
    if n == 0:
        return b_only, c_only, 1.0
    k = min(b_only, c_only)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return b_only, c_only, min(p, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--baseline", default=None,
                    help="Compare against this one baseline instead of the "
                         "strongest per cell (e.g. self_consistency)")
    ap.add_argument("--keep_refused", action="store_true",
                    help="Score the samples whose consensus RKG came out empty "
                         "as CRAFT errors instead of dropping them from every "
                         "method. The pipeline's rule is to drop them; this "
                         "shows what they cost")
    ap.add_argument("--json", type=Path, default=None, help="Also write the table here")
    a = ap.parse_args()

    table = []
    for model in a.models:
        for dataset in a.datasets:
            try:
                cp, cfile, refused = craft_predictions(model, dataset)
            except FileNotFoundError:
                continue
            gold = gold_map(model, dataset)
            if not gold:
                continue
            bl = baseline_predictions(model, dataset)
            # the questions every method was asked, minus the ones the pipeline
            # refused outright, which nobody is then scored on
            asked = set(gold) & set(cp)
            if bl:
                asked &= set.intersection(*[set(p) for p in bl.values()])
            if a.keep_refused:
                asked |= (set(gold) & set(refused))
            ids = sorted(asked - (set() if a.keep_refused else set(refused)))
            if not ids:
                continue
            craft_hits = _hits(cp, gold, ids, dataset)
            craft = sum(craft_hits) / len(craft_hits)
            hits = {s: _hits(p, gold, ids, dataset) for s, p in bl.items()}
            scored = {s: sum(h) / len(h) for s, h in hits.items()}
            if a.baseline:
                best_name, best = a.baseline, scored.get(a.baseline, float("nan"))
            else:
                best_name, best = max(scored.items(), key=lambda kv: kv[1]) \
                    if scored else ("—", float("nan"))
            won, lost, pval = (mcnemar(craft_hits, hits[best_name])
                               if best_name in hits else (0, 0, 1.0))
            table.append({
                "model": model, "dataset": dataset, "n": len(ids),
                "refused": len(refused),
                "craft": craft, "best_baseline": best_name, "baseline": best,
                "delta": craft - best, "craft_file": cfile.name,
                "only_craft": won, "only_baseline": lost, "p": pval,
                "all_baselines": scored,
            })

    w = max((len(f"{r['model']}/{r['dataset']}") for r in table), default=20)
    print(f"\n{'cell':<{w}} {'n':>5} {'CRAFT':>7} {'baseline':>9} "
          f"{'Δ':>7} {'only/only':>10} {'p':>7}  {'strongest baseline':<22}")
    print("-" * (w + 66))
    wins = sig = ties = 0
    for r in table:
        ahead = r["delta"] > 0
        wins += ahead
        strong = r["p"] < 0.05
        sig += ahead and strong
        ties += not strong
        mark = "*" if strong else " "
        print(f"{r['model']+'/'+r['dataset']:<{w}} {r['n']:>5} "
              f"{r['craft']*100:>7.1f} {r['baseline']*100:>9.1f} "
              f"{r['delta']*100:>+7.1f} {str(r['only_craft'])+'/'+str(r['only_baseline']):>10} "
              f"{r['p']:>7.3f}{mark} {r['best_baseline']:<22}")
    print(f"\n  CRAFT ahead on {wins}/{len(table)} cells; "
          f"{sig} of those significant at p<0.05 (*), "
          f"{ties} cells statistically tied")
    print("  only/only = samples only CRAFT got right / only the baseline did; "
          "p is exact two-sided McNemar on those")

    if a.json:
        a.json.write_text(json.dumps(table, indent=2), encoding="utf-8")
        print(f"  → {a.json}")


if __name__ == "__main__":
    main()
