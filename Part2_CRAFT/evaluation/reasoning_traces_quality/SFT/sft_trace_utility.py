#!/usr/bin/env python3
"""Fine-tune one student on one side of the paired traces, then answer held-out problems.

Two runs of this differing only in --train_file are the experiment: whichever
student answers the held-out problems better learned from the better traces.
Everything a run can vary is therefore fixed by argument and logged, and the
seed is set for torch, numpy and python so two runs of the same side differ
only where the hardware does.

Loss is taken on the trace alone. With the prompt in the loss the student is
partly learning to reproduce problem statements, which are identical on both
sides, and that dilutes exactly the difference being measured.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList, Trainer,
                          TrainingArguments)
from peft import LoraConfig, get_peft_model

def _add_scorer_path() -> None:
    """Put label_prediction on the path from wherever this file was copied to.

    The runs happen on a cluster, where this script sits beside its data rather
    than in the tree, so a path built from __file__'s parents finds nothing and
    the import kills the job after the backbone has loaded. Each candidate is
    tried and the first that holds the modules wins.
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "label_prediction",   # in the repo
        here.parent / "label_prediction",       # copied with its scorers beside it
        here.parent,                            # copied flat
    ]
    for c in candidates:
        if all((c / m).exists() for m in
               ("answer_match.py", "extract_label.py", "step_count.py")):
            sys.path.insert(0, str(c))
            return
    raise SystemExit(
        "answer_match.py, extract_label.py and step_count.py were not found "
        f"next to this script or at {candidates[0]}; copy them beside it.")


_add_scorer_path()

from answer_match import answers_match  # noqa: E402
from step_count import count_steps, count_tokens  # noqa: E402

from extract_label import extract_label, extract_math_answer  # noqa: E402

# The reader and the comparison a held-out problem is marked with are the ones
# the main table uses, so a student's accuracy here means what accuracy means
# everywhere else in this project.
_READER = {"FLD": extract_label, "ProofWriter": extract_label,
           "OmniMATH": extract_math_answer, "OlympiadBench": extract_math_answer}


_LOGICAL = {"FLD", "ProofWriter"}
_BINARY = ["__PROVED__", "__DISPROVED__"]


def _metrics(rows: list) -> dict:
    """Accuracy and average steps for every set, plus macro-F1 for a logical one.

    The paper's table reads three numbers off a logical column and two off a
    mathematical one, because F1 needs classes to average over and a maths
    answer is not a class. The formula is evaluate_accuracy's, so a number here
    means what the same name means in the main table.
    """
    n = len(rows)
    out = {
        "n": n,
        "accuracy": round(sum(r["correct"] for r in rows) / max(n, 1), 4),
        "avg_steps": round(sum(r["n_steps"] for r in rows) / max(n, 1), 2),
        "avg_tokens": round(sum(r["n_tokens"] for r in rows) / max(n, 1), 1),
    }
    if rows and rows[0]["dataset"] not in _LOGICAL:
        out["macro_f1"] = None       # no classes to average over
        return out

    confusion = {g: {p: 0 for p in _BINARY} for g in _BINARY}
    for r in rows:
        gold, pred = r["answer"], r["predicted"]
        if gold in confusion and pred in _BINARY:
            confusion[gold][pred] += 1
    f1s = {}
    for cls in _BINARY:
        tp = confusion[cls][cls]
        fp = sum(confusion[g][cls] for g in _BINARY if g != cls)
        fn = sum(confusion[cls][p] for p in _BINARY if p != cls)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s[cls] = round(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0, 4)
    out["macro_f1"] = round(sum(f1s.values()) / len(_BINARY), 4)
    out["per_class_f1"] = f1s
    return out


ANSWER_PAT = re.compile(r"__(?:PROVED|DISPROVED)__|\b(?:PROVED|DISPROVED)\b"
                        r"|\\boxed\s*\{[^{}]*\}")


class AnswerStated(StoppingCriteria):
    """Stop each sequence once it has stated an answer.

    Generation is what makes this job long, and a student that has written its
    conclusion has nothing left to say that is scored. The rule is per
    sequence: a finished row is padded from then on while the rest of the
    batch goes on, so nothing is appended after a stated answer -- the first
    version stopped on the whole batch, and a student that had answered kept
    writing until the slowest row in its batch did, which put a second
    \\boxed{} after the first on a few problems and moved their mark.
    """

    def __init__(self, tok, prompt_len: int):
        self.tok, self.prompt_len = tok, prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        done = [bool(ANSWER_PAT.search(
                    self.tok.decode(row[self.prompt_len:], skip_special_tokens=True)))
                for row in input_ids]
        return torch.tensor(done, dtype=torch.bool, device=input_ids.device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def build_prompt(rec: Dict) -> str:
    return (f"{rec['instruction']}\n\n{rec['problem']}\n\nReasoning:\n")


def over_budget(rows: List[Dict], tok, max_len: int) -> List[str]:
    """sample_ids whose prompt + trace + eos would not fit in max_len."""
    out = []
    for rec in rows:
        n = (len(tok(build_prompt(rec), add_special_tokens=False)["input_ids"])
             + len(tok(rec["trace"] + tok.eos_token, add_special_tokens=False)["input_ids"]))
        if n > max_len:
            out.append(rec["sample_id"])
    return out


class TraceSFT(Dataset):
    """prompt -> trace, with the prompt masked out of the loss."""

    def __init__(self, rows: List[Dict], tok, max_len: int):
        self.rows, self.tok, self.max_len = rows, tok, max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        rec = self.rows[i]
        p_ids = self.tok(build_prompt(rec), add_special_tokens=False)["input_ids"]
        t_ids = self.tok(rec["trace"] + self.tok.eos_token,
                         add_special_tokens=False)["input_ids"]
        # A trace over budget keeps its opening and its ending and loses the
        # middle. Cutting from the end instead removes the conclusion, and the
        # two sides are not over budget equally often — at 3072 tokens that
        # truncated 403 of CRAFT's 2008 traces and 0 of the raw ones, because
        # CRAFT's run four times longer. A student would then be trained on a
        # fifth of its examples with no answer on them while the other student
        # had all of its, and would lose for that and not for its reasoning.
        room = max(self.max_len - len(p_ids), 8)
        if len(t_ids) > room:
            tail = max(room // 2, 1)
            t_ids = t_ids[:room - tail] + t_ids[-tail:]
        ids = (p_ids + t_ids)[-self.max_len:]
        n_prompt = max(len(ids) - len(t_ids), 0)
        labels = [-100] * n_prompt + ids[n_prompt:]
        return {"input_ids": torch.tensor(ids),
                "labels": torch.tensor(labels)}


def collate(batch, pad_id: int):
    n = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for b in batch:
        k = n - len(b["input_ids"])
        out["input_ids"].append(torch.cat([b["input_ids"], torch.full((k,), pad_id)]))
        out["labels"].append(torch.cat([b["labels"], torch.full((k,), -100)]))
        out["attention_mask"].append(
            torch.cat([torch.ones(len(b["input_ids"])), torch.zeros(k)]))
    return {k: torch.stack(v).long() for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_len", type=int, default=6144,
                    help="Token budget per example. 6144 leaves 49 of 2008 "
                         "CRAFT traces over budget and 0 raw ones; 3072 leaves "
                         "403 and 0, which is an asymmetry in the training "
                         "signal rather than in the traces")
    ap.add_argument("--gen_batch", type=int, default=16,
                    help="Problems generated together, sorted by length so a "
                         "batch pads little")
    ap.add_argument("--gen_max_new", type=int, default=2048,
                    help="Both students get the same budget; it has to clear "
                         "the longer style so neither is cut off before its "
                         "answer")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--allow_truncation", action="store_true",
                    help="Train even when some traces do not fit --max_len "
                         "(they then lose their middle). Off by default: a "
                         "student is meant to learn from whole traces, so an "
                         "over-budget training set is refused and named")
    ap.add_argument("--tag", default="",
                    help="A word for the run's name, between the side and the "
                         "seed, when the training set is not the default "
                         "both-correct pairs -- e.g. 'all' for every concluded "
                         "trace, right or wrong")
    ap.add_argument("--save_merged", action="store_true",
                    help="Also write the backbone with the adapter merged in "
                         "(about 18 GB); the adapter alone is always saved")
    args = ap.parse_args()

    seed_everything(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    train_rows = read_jsonl(Path(args.train_file))
    print(f"  training on {len(train_rows)} traces from {Path(args.train_file).name}",
          flush=True)
    long = over_budget(train_rows, tok, args.max_len)
    if long:
        msg = (f"  {len(long)} of {len(train_rows)} traces do not fit --max_len "
               f"{args.max_len}: {', '.join(long[:8])}{' ...' if len(long) > 8 else ''}")
        if not args.allow_truncation:
            raise SystemExit(msg + "\n  raise --max_len, drop them from both "
                             "sides, or pass --allow_truncation")
        print(msg + "  (kept, middles cut)", flush=True)
    else:
        print(f"  every trace fits --max_len {args.max_len}; nothing is truncated",
              flush=True)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out / "ckpt"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            # transformers 5 dropped warmup_ratio. The step count is the same
            # on both sides of the pair -- the two training sets hold the same
            # 2008 problems -- so naming the steps outright gives the two
            # students an identical schedule rather than one recomputed from a
            # ratio each run.
            warmup_steps=max(1, round(0.03 * (len(train_rows) / args.grad_accum)
                                      * args.epochs)),
            logging_steps=20,
            save_strategy="no",
            bf16=True,
            gradient_checkpointing=True,
            report_to=[],
            seed=args.seed,
            data_seed=args.seed,
        ),
        train_dataset=TraceSFT(train_rows, tok, args.max_len),
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )
    trainer.train()

    # The student itself, so a run can be re-scored or reused without
    # training again: the LoRA adapter and the tokenizer, and on request the
    # backbone with the adapter merged in.
    model.save_pretrained(str(out / "adapter"))
    tok.save_pretrained(str(out / "adapter"))
    print(f"  adapter saved -> {out / 'adapter'}", flush=True)
    if args.save_merged:
        merged = model.merge_and_unload()
        merged.save_pretrained(str(out / "merged"), safe_serialization=True)
        tok.save_pretrained(str(out / "merged"))
        print(f"  merged model saved -> {out / 'merged'}", flush=True)
        model = merged

    # ── answer the held-out problems ────────────────────────────────────────
    model.config.use_cache = True
    model.eval()
    test_rows = read_jsonl(Path(args.test_file))
    print(f"  generating on {len(test_rows)} held-out problems", flush=True)

    # Batched, and stopped as soon as every sequence in the batch has stated an
    # answer. One at a time with the full budget is 618 problems x 2048 tokens
    # at roughly 30 tokens a second, which does not fit the job's walltime; the
    # students also differ in how much they write, so a per-sequence budget that
    # is never reached would have cost the two sides differently in wall-clock
    # but not in what they were allowed to say. Both get the same budget and the
    # same stopping rule.
    tok.padding_side = "left"
    preds = []
    order = sorted(range(len(test_rows)),
                   key=lambda i: len(test_rows[i]["problem"]))
    with torch.no_grad():
        for start in range(0, len(order), args.gen_batch):
            idxs = order[start:start + args.gen_batch]
            batch = [test_rows[i] for i in idxs]
            enc = tok([build_prompt(r) for r in batch], return_tensors="pt",
                      padding=True, truncation=True,
                      max_length=args.max_len).to("cuda")
            gen = model.generate(**enc, max_new_tokens=args.gen_max_new,
                                 do_sample=False, pad_token_id=tok.pad_token_id,
                                 stopping_criteria=StoppingCriteriaList(
                                     [AnswerStated(tok, enc["input_ids"].shape[1])]))
            for r, row in zip(batch, gen):
                new_ids = row[enc["input_ids"].shape[1]:]
                text = tok.decode(new_ids, skip_special_tokens=True)
                n_new = int((new_ids != tok.pad_token_id).sum())
                preds.append({**{k: r[k] for k in
                                 ("sample_id", "dataset", "domain", "answer")},
                              "generated": text,
                              # A generation that used its whole budget without
                              # stating an answer was cut off, not finished.
                              "hit_budget": bool(n_new >= args.gen_max_new
                                                 and not ANSWER_PAT.search(text))})
            print(f"    {len(preds)}/{len(test_rows)}", flush=True)

    # ── mark them, and write the one file this run is read from ─────────────
    by_ds: Dict[str, List[Dict]] = {}
    for r in preds:
        reader = _READER.get(r["dataset"], extract_label)
        got = reader(r["generated"] or "")
        r["predicted"] = got
        r["correct"] = bool(got is not None
                            and answers_match(got, r["answer"], r["dataset"]))
        r["n_steps"] = count_steps(r["generated"] or "")
        r["n_tokens"] = count_tokens(r["generated"] or "")
        by_ds.setdefault(r["dataset"], []).append(r)

    side = "CRAFT" if "craft" in Path(args.train_file).stem else "Raw_CoT"
    # Qwen/Qwen3.5-9B -> Qwen-3.5-9B, so a file says which backbone it is.
    slug = re.sub(r"^([A-Za-z]+)(?=\d)", r"\1-", args.model.rsplit("/", 1)[-1])
    name = f"{slug}_SFT_{side}{'_' + args.tag if args.tag else ''}_Seed{args.seed}"
    summary = {
        "run": name,
        "side": "craft" if side == "CRAFT" else "raw",
        "training_set": args.tag or "both_correct_pairs",
        "seed": args.seed,
        "model": args.model,
        "train_file": Path(args.train_file).name,
        "n_train": len(train_rows),
        "n_test": len(preds),
        "epochs": args.epochs,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "accuracy": round(sum(r["correct"] for r in preds) / max(len(preds), 1), 4),
        "max_len": args.max_len,
        "gen_max_new": args.gen_max_new,
        "n_no_answer": sum(1 for r in preds if r["predicted"] is None),
        "n_hit_budget": sum(1 for r in preds if r["hit_budget"]),
        "avg_steps": round(sum(r["n_steps"] for r in preds) / max(len(preds), 1), 2),
        "by_dataset": {d: _metrics(rs) for d, rs in sorted(by_ds.items())},
        "predictions": preds,
    }
    with (out / f"{name}.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    print(f"  {name}: accuracy {100*summary['accuracy']:.1f}  "
          f"steps {summary['avg_steps']:.1f}  over {len(preds)} problems  "
          f"(no answer {summary['n_no_answer']}, cut off at the budget "
          f"{summary['n_hit_budget']})", flush=True)
    for d, m in summary["by_dataset"].items():
        f1 = "  —  " if m["macro_f1"] is None else f"{m['macro_f1']:.3f}"
        print(f"      {d:<16} acc {100*m['accuracy']:5.1f}   F1 {f1}   "
              f"steps {m['avg_steps']:5.1f}   n {m['n']}", flush=True)


if __name__ == "__main__":
    main()
