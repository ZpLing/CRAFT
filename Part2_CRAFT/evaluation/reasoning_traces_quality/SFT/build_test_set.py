#!/usr/bin/env python3
"""A test set of problems the pipeline never saw, drawn from the source datasets.

Holding out part of the run's own 500 would cost training data and would test
on problems drawn by the same selection that produced the traces. The source
datasets are much larger than the slice Part 2 uses, so the held-out problems
come from what was left behind: same benchmark, same configuration, no overlap.

How much is left behind differs by an order of magnitude, and the smallest one
sets the size:

    FLD             80,000 in the source, 500 used
    Omni-MATH        4,428 in the source, ~500 used
    OlympiadBench      674 in OE_TO_maths_en_COMP, 500 used  ->   174 left
    ProofWriter        104 left at RelNeg-OWA-D5 with a binary answer

ProofWriter is the binding one. Its mirrors carry 3,000 problems but most are
other configurations (RelNoneg, AttNoneg) or other depths or the three-way
answer with Unknown, and matching the slice Part 2 runs on leaves 104. Taking
the others would test on a different task and call it the same benchmark.

Overlap is checked on the problem text, not on an index: the mirrors do not
carry the source's own ids, and two of the four datasets were selected by row
rather than by id.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import config as _cfg  # noqa: E402

INSTRUCTION = {
    "logical": ("Decide whether the hypothesis follows from the facts and rules. "
                "Reason step by step, then end with __PROVED__ or __DISPROVED__."),
    "math": ("Solve the problem. Reason step by step, then give the final answer "
             "in \\boxed{}."),
}


def norm(text) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()[:300]


def used_texts(dataset_dir: Path, name: str, fields) -> set:
    raw = json.loads((dataset_dir / f"{name}.json").read_text(encoding="utf-8"))
    rows = raw if isinstance(raw, list) else raw.get("data") or raw.get("samples")
    return {tuple(norm(r.get(f)) for f in fields) for r in rows}


def fld(used: set, cap: int) -> List[Dict]:
    from huggingface_hub import hf_hub_download, list_repo_files
    import pandas as pd
    out = []
    for f in sorted(list_repo_files("hitachi-nlp/FLD.v2", repo_type="dataset")):
        if not f.endswith(".parquet"):
            continue
        df = pd.read_parquet(hf_hub_download("hitachi-nlp/FLD.v2", f, repo_type="dataset"))
        for _, r in df.iterrows():
            label = str(r.get("world_assump_label") or r.get("proof_label") or "")
            if label not in ("PROVED", "DISPROVED"):
                continue
            facts, concl = r.get("facts") or r.get("context"), r.get("hypothesis")
            if (norm(facts), norm(concl)) in used:
                continue
            out.append({"dataset": "FLD", "domain": "logical",
                        "problem": f"Facts:\n{facts}\n\nHypothesis:\n{concl}",
                        "answer": f"__{label}__"})
            if len(out) >= cap:
                return out
    return out


def proofwriter(used: set, cap: int) -> List[Dict]:
    from huggingface_hub import hf_hub_download
    import pandas as pd
    out = []
    for sp in ("train", "dev", "test"):
        df = pd.read_parquet(hf_hub_download(
            "smoorsmith/proofwriter", f"data/{sp}-00000-of-00001.parquet",
            repo_type="dataset"))
        sel = df[(df["depth"] == 5)
                 & (df["answer"].astype(str).isin(["A", "B"]))
                 & (df["id"].astype(str).str.contains("RelNeg-OWA-D5"))]
        for _, r in sel.iterrows():
            if (norm(r["context"]), norm(r["question"])) in used:
                continue
            # A is the first option and these are (True, False) throughout.
            label = "__PROVED__" if str(r["answer"]) == "A" else "__DISPROVED__"
            out.append({"dataset": "ProofWriter", "domain": "logical",
                        "problem": f"Facts:\n{r['context']}\n\nHypothesis:\n{r['question']}",
                        "answer": label})
            if len(out) >= cap:
                return out
    return out


def omnimath(used: set, cap: int) -> List[Dict]:
    from huggingface_hub import hf_hub_download
    out = []
    p = hf_hub_download("KbsdJames/Omni-MATH", "test.jsonl", repo_type="dataset")
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        prob, ans = r.get("problem"), r.get("answer")
        if not prob or not ans or (norm(prob),) in used:
            continue
        out.append({"dataset": "OmniMATH", "domain": "math",
                    "problem": str(prob), "answer": str(ans)})
        if len(out) >= cap:
            break
    return out


def olympiadbench(used: set, cap: int) -> List[Dict]:
    from huggingface_hub import hf_hub_download
    import pandas as pd
    df = pd.read_parquet(hf_hub_download(
        "Hothan/OlympiadBench",
        "OlympiadBench/OE_TO_maths_en_COMP/OE_TO_maths_en_COMP.parquet",
        repo_type="dataset"))
    out = []
    for _, r in df.iterrows():
        prob = r.get("question")
        ans = r.get("final_answer")
        if hasattr(ans, "__len__") and not isinstance(ans, str):
            ans = ans[0] if len(ans) else None
        if not prob or ans is None or (norm(prob),) in used:
            continue
        out.append({"dataset": "OlympiadBench", "domain": "math",
                    "problem": str(prob), "answer": str(ans)})
        if len(out) >= cap:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_dir", default="dataset")
    ap.add_argument("--out", default="results/CRAFT_results/other_results/trace_utility/data/test.jsonl")
    ap.add_argument("--cap", type=int, default=174,
                    help="At most this many per benchmark. 174 is what "
                         "OlympiadBench has left; ProofWriter yields fewer and "
                         "the counts are reported rather than padded")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[3]
    dd = root / args.dataset_dir

    builders = [
        ("FLD", fld, used_texts(dd, "FLD", ("Facts", "Conclusion"))),
        ("ProofWriter", proofwriter, used_texts(dd, "ProofWriter", ("Facts", "Conclusion"))),
        ("OmniMATH", omnimath, used_texts(dd, "OmniMATH", ("input",))),
        ("OlympiadBench", olympiadbench, used_texts(dd, "OlympiadBench", ("input",))),
    ]
    rows: List[Dict] = []
    for name, fn, used in builders:
        got = fn(used, args.cap)
        for i, r in enumerate(got):
            r["sample_id"] = f"heldout_{name}_{i}"
            r["instruction"] = INSTRUCTION[r["domain"]]
        rows += got
        print(f"  {name:<16}{len(got):>5} held-out problems")

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  {len(rows)} problems -> {out}")


if __name__ == "__main__":
    main()
