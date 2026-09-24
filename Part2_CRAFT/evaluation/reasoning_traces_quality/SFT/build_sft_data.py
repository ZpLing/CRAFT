#!/usr/bin/env python3
"""Paired fine-tuning data for asking whether a CRAFT trace is better to learn from.

A judge scoring steps can only report what a step looks like. A per-step
atomicity score is the clearest case: it penalises a step for being long, which
is what CRAFT's steps are by construction — 178 words a step against raw CoT's
12 on OlympiadBench — and says nothing about whether the reasoning in them is
any good. Training says something a judge cannot. Fine-tune
one student on CRAFT's traces and another on the raw chains over the same
problems, change nothing else, and whichever student answers held-out problems
better learned from the better traces.

The comparison only means that if correctness is held fixed. CRAFT's traces
carry more right answers than the raw chains do — 205 against 121 on nano's
Omni-MATH — so a student trained on all of them would win by being shown better
answers, which is the label-prediction result restated, not a statement about
traces. So the pairs here are the problems where BOTH sides are right: same
problems, same count, same answers, differing only in how the reasoning is laid
out.

    split      500 problems per cell, 80/20 by sample_id, seeded and stable
    train      the pairs in the 80% where both traces reach the gold answer
    test       the whole 20%, at natural difficulty, unseen by either student

Writes train_craft.jsonl, train_raw.jsonl (same sample_ids, same length) and
test.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List

MODELS = ["gemini-3.1-flash-lite", "gpt-5.4-nano"]
DATASETS = ["FLD", "ProofWriter", "OmniMATH", "OlympiadBench"]

LABEL_RE = re.compile(r"__(?:PROVED|DISPROVED)__", re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\s*\{")
# PROVED / DISPROVED as a word, with or without the underscores the prompt asks
# for, which is the form the baselines' own extractor accepts.
BARE_LABEL_RE = re.compile(r"\b(?:PROVED|DISPROVED)\b")


def has_conclusion(trace: str, domain: str) -> bool:
    """Whether a trajectory states an answer at all, rather than being cut off.

    A logical answer is written three ways across these runs. CRAFT's synthesis
    is told to emit __PROVED__ and does; the Raw CoT baseline was told
    the same and often ends with a bare PROVED instead, which is what their own
    extractor reads and scores. Matching only the underscored form called 276
    of nano's 500 ProofWriter chains unfinished when every one of them ends in a
    stated conclusion — the reading that would have dropped them from the pairs,
    or had them regenerated against a baseline that is not in fact broken.
    """
    if not trace:
        return False
    if domain != "logical":
        return bool(BOXED_RE.search(trace))
    return bool(LABEL_RE.search(trace) or BARE_LABEL_RE.search(trace))


def strip_answer(trace: str, domain: str) -> str:
    """Cut the trace off before it states its answer.

    With the answer in the training target the student can learn which endings
    go with which problems instead of learning to reason to them, and the two
    sides carry the SAME answer here — a pair is only kept where both were
    right — so the answer is shared text that dilutes the one thing this
    experiment varies. Removing it leaves the reasoning, which is the thing
    being compared.

    The cut is at the start of the line that states the answer, so the line is
    removed whole rather than leaving "Therefore, the hypothesis is" dangling.
    Lines after it, which are usually blank or a restatement, go too.

    Not used by default. The default trains on the trajectory the pipeline
    produced, reasoning and answer together, which is the artefact the paper
    claims is better; a pair is only kept where both sides reach the same
    answer, so that shared ending cannot favour either student.
    """
    if not trace:
        return ""
    lines = trace.rstrip().split("\n")
    pat = LABEL_RE if domain == "logical" else BOXED_RE
    # The cut is at the FIRST line that states the answer, not the last. CRAFT's
    # traces restate their conclusion as they go — the consensus graph gives the
    # conclusion node's text to the steps that lead to it — so cutting at the
    # last occurrence leaves the answer sitting in an earlier step. Measured on
    # this data that left it in 332 of 1596 CRAFT traces against 11 raw ones,
    # which is worse than not stripping at all: it would have leaked the answer
    # to one student and not the other.
    for i, line in enumerate(lines):
        if pat.search(line):
            return "\n".join(lines[:i]).rstrip()
    return trace.rstrip()


INSTRUCTION = {
    "logical": ("Decide whether the hypothesis follows from the facts and rules. "
                "Reason step by step, then end with __PROVED__ or __DISPROVED__."),
    "math": ("Solve the problem. Reason step by step, then give the final answer "
             "in \\boxed{}."),
}


def in_test_split(sample_id: str, frac: float, seed: int) -> bool:
    """A stable 20% by sample_id.

    Hashing the id rather than shuffling a list keeps a problem on the same side
    of the split no matter which cell is being built or what order the records
    were written in, which asyncio does not fix.
    """
    h = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    return (int.from_bytes(h[:4], "big") % 10_000) < frac * 10_000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", default="results/CRAFT_results/Raw_Output")
    ap.add_argument("--baselines", default="results/baseline_results")
    ap.add_argument("--k_runs", default="results/craft_runs/full500")
    ap.add_argument("--out_dir", default="results/CRAFT_results/other_results/trace_utility/data")
    ap.add_argument("--test_frac", type=float, default=0.0,
                    help="Hold out this share of the run's own problems. 0 by "
                         "default: the test set is built from problems the "
                         "pipeline never touched, by build_test_set.py, so the "
                         "run's 500 per cell can all be trained on")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all_pairs", action="store_true", default=False,
                    help="Keep every problem both sides wrote a concluded trace "
                         "for, right or wrong, instead of only the problems both "
                         "got right. This asks a different question -- whether "
                         "a pipeline's whole output is the better thing to learn "
                         "from, answers included -- so its runs are named apart")
    ap.add_argument("--max_tokens", type=int, default=0,
                    help="Drop a pair when either side's prompt + trace would not "
                         "fit this many tokens of --tokenizer, so the students "
                         "train on whole traces and on the same problems. 0 keeps all")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--strip_answer", action="store_true", default=False,
                    help="Cut each training trace off before it states its "
                         "answer. Off by default — a student is trained on the "
                         "trajectory as the pipeline produced it, reasoning and "
                         "answer together. Both sides of a pair reach the same "
                         "answer, so including it cannot favour either student")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[3]
    out = root / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    train_craft: List[Dict] = []
    train_raw: List[Dict] = []
    test: List[Dict] = []
    test_ids: set = set()
    stats = []

    for model in MODELS:
        raw_by_id = {}
        for line in (root / args.baselines / model / "cot" / "traces.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            raw_by_id[r["sample_id"]] = r

        for ds in DATASETS:
            rows = [json.loads(l) for l in
                    (root / args.traces / model / f"{ds}_Output.jsonl").open(encoding="utf-8")]
            kpath = root / args.k_runs / f"{ds}_{model}" / "k_traces_500_samples.json"
            if not kpath.exists() and ds == "ProofWriter" and model.startswith("gemini"):
                kpath = root / args.k_runs / "ProofWriter_gd" / "k_traces_500_samples.json"
            kraw = json.loads(kpath.read_text(encoding="utf-8"))
            krows = kraw.get("results", kraw) if isinstance(kraw, dict) else kraw
            problems = {r["sample_id"]: r.get("problem_text") or "" for r in krows}

            n_tr = n_te = 0
            for r in rows:
                sid = r["sample_id"]
                problem = problems.get(sid, "")
                if not problem:
                    continue
                domain = r["domain"]
                rr = raw_by_id.get(sid)
                if rr is None:
                    continue

                if args.test_frac > 0 and in_test_split(sid, args.test_frac, args.seed):
                    # A problem is one problem. It appears once per backbone in
                    # the traces, because both were run over the same 500, and a
                    # student answers it once — counting it twice would weight
                    # these problems double and report a smaller interval than
                    # the evidence supports.
                    if sid not in test_ids:
                        test_ids.add(sid)
                        test.append({"sample_id": sid, "dataset": ds,
                                     "domain": domain, "instruction": INSTRUCTION[domain],
                                     "problem": problem, "answer": r["ground_truth"]})
                        n_te += 1
                    continue

                craft_ok = r["predicted"] == r["ground_truth"]
                raw_ok = rr.get("predicted") == rr.get("ground_truth")
                if not args.all_pairs and not (craft_ok and raw_ok):
                    continue
                raw_trace = (rr.get("traces") or [""])[0] or ""
                if not r.get("trace") or not raw_trace:
                    continue
                common = {"sample_id": sid, "dataset": ds, "source_model": model,
                          "domain": domain, "instruction": INSTRUCTION[domain],
                          "problem": problem, "answer": r["ground_truth"]}
                # A pair enters only if both trajectories actually reach a
                # stated conclusion. 295 of the raw chains — 18.5% of them, all
                # logical — run out of tokens mid-sentence and never state one,
                # while none of CRAFT's do; the baseline still recorded them as
                # correct. Training a student on those teaches it to stop
                # mid-sentence, and it would then lose for a reason that has
                # nothing to do with how well the trace reasons. Dropping the
                # pair keeps both trajectories exactly as the pipelines wrote
                # them and keeps the two sides comparable.
                if not (has_conclusion(r["trace"], domain)
                        and has_conclusion(raw_trace, domain)):
                    continue
                c_body = strip_answer(r["trace"], domain) if args.strip_answer else r["trace"]
                r_body = strip_answer(raw_trace, domain) if args.strip_answer else raw_trace
                # Whole trajectories by default; the guard below only bites when
                # --strip_answer has cut one down to nothing.
                # A trace that is only its answer has no reasoning to learn from,
                # and stripping leaves it empty on one side and not the other,
                # which would make the two training sets different sizes.
                if args.strip_answer and (len(c_body.split()) < 10 or len(r_body.split()) < 10):
                    continue
                train_craft.append({**common, "trace": c_body, "side": "craft"})
                train_raw.append({**common, "trace": r_body, "side": "raw"})
                n_tr += 1
            stats.append((model, ds, n_tr, n_te))

    assert [r["sample_id"] for r in train_craft] == [r["sample_id"] for r in train_raw], \
        "the two training sets must cover the same problems in the same order"

    if args.max_tokens:
        # The same budget the trainer is given; a pair goes when either side
        # would not fit, so neither student sees a trace with its middle cut.
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)

        def fits(rec: Dict) -> bool:
            prompt = f"{rec['instruction']}\n\n{rec['problem']}\n\nReasoning:\n"
            n = (len(tok(prompt, add_special_tokens=False)["input_ids"])
                 + len(tok(rec["trace"], add_special_tokens=False)["input_ids"]) + 1)
            return n <= args.max_tokens

        keep = [fits(c) and fits(r) for c, r in zip(train_craft, train_raw)]
        dropped = [c["sample_id"] for c, k in zip(train_craft, keep) if not k]
        train_craft = [c for c, k in zip(train_craft, keep) if k]
        train_raw = [r for r, k in zip(train_raw, keep) if k]
        print(f"  over {args.max_tokens} tokens on either side, dropped from both: "
              f"{len(dropped)} {dropped[:6]}{' ...' if len(dropped) > 6 else ''}")
    # A problem solved by both backbones contributes twice, once per backbone's
    # trace. That is left in: it is how the traces were produced, and it falls on
    # both sides equally, so it cannot favour either student.

    for name, rows in (("train_craft", train_craft), ("train_raw", train_raw), ("test", test)):
        p = out / f"{name}.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  {'cell':<38}{'train':>7}{'test':>7}")
    for model, ds, a, b in stats:
        print(f"  {model.split('-')[0] + '/' + ds:<38}{a:>7}{b:>7}")
    print(f"\n  train_craft.jsonl / train_raw.jsonl: {len(train_craft)} pairs, same ids")
    print(f"  test.jsonl: {len(test)} held-out problems")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
