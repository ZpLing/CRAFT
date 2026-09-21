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
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)
from peft import LoraConfig, get_peft_model


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
        # A trace longer than the budget is truncated from its end; cutting the
        # prompt instead would remove the problem the trace is about.
        room = max(self.max_len - len(p_ids), 8)
        t_ids = t_ids[:room]
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
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--gen_max_new", type=int, default=1024)
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

    preds = []
    with torch.no_grad():
        for i, rec in enumerate(test_rows):
            ids = tok(build_prompt(rec), return_tensors="pt",
                      truncation=True, max_length=args.max_len).to("cuda")
            gen = model.generate(**ids, max_new_tokens=args.gen_max_new,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
            text = tok.decode(gen[0][ids["input_ids"].shape[1]:],
                              skip_special_tokens=True)
            preds.append({**{k: rec[k] for k in
                             ("sample_id", "dataset", "domain", "answer")},
                          "generated": text})
            if (i + 1) % 25 == 0:
                print(f"    {i + 1}/{len(test_rows)}", flush=True)

    with (out / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for r in preds:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {out / 'predictions.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
