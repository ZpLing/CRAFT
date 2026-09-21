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
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList, Trainer,
                          TrainingArguments)
from peft import LoraConfig, get_peft_model


ANSWER_PAT = re.compile(r"__(?:PROVED|DISPROVED)__|\b(?:PROVED|DISPROVED)\b"
                        r"|\\boxed\s*\{[^{}]*\}")


class AnswerStated(StoppingCriteria):
    """Stop once every sequence in the batch has stated an answer.

    Generation is what makes this job long, and a student that has written its
    conclusion has nothing left to say that is scored. Stopping on the whole
    batch rather than per sequence keeps it simple and keeps both students under
    the same rule; a sequence that finished early just pads.
    """

    def __init__(self, tok, prompt_len: int):
        self.tok, self.prompt_len = tok, prompt_len

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        for row in input_ids:
            text = self.tok.decode(row[self.prompt_len:], skip_special_tokens=True)
            if not ANSWER_PAT.search(text):
                return False
        return True


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

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out / "ckpt"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
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
                text = tok.decode(row[enc["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
                preds.append({**{k: r[k] for k in
                                 ("sample_id", "dataset", "domain", "answer")},
                              "generated": text})
            print(f"    {len(preds)}/{len(test_rows)}", flush=True)

    with (out / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for r in preds:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {out / 'predictions.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
