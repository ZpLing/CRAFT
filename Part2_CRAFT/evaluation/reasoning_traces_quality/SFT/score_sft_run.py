#!/usr/bin/env python3
"""Mark a trace-utility run and write the file its cell is read from.

sft_trace_utility.py does this itself now. This is for the runs that finished
before it did, which left a predictions.jsonl and no marks: it reads that file
and writes the same SFT_<Side>_Seed<n>.json beside it, so a run from either
version is read the same way.

    python score_sft_run.py [--train_root DIR] runs/raw-s0 ...
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

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

_READER = {"FLD": extract_label, "ProofWriter": extract_label,
           "OmniMATH": extract_math_answer, "OlympiadBench": extract_math_answer}


def _slug(model: str) -> str:
    """Qwen/Qwen3.5-9B -> Qwen-3.5-9B, so a file says which backbone it is."""
    name = (model or "model").rsplit("/", 1)[-1]
    return re.sub(r"^([A-Za-z]+)(?=\d)", r"\1-", name)


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


def score(run_dir: Path, train_root: Path = None) -> dict:
    preds = [json.loads(l) for l in (run_dir / "predictions.jsonl").open(encoding="utf-8")
             if l.strip()]
    cfg = {}
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    # The first version of the trainer wrote its arguments and not its counts,
    # so a run from it has no n_train to copy; the training file it names is
    # counted instead, and the field is only left empty when that file is not
    # where the run said it was.
    n_train = cfg.get("n_train")
    if n_train is None and cfg.get("train_file"):
        for root in (train_root, run_dir.parent.parent, run_dir.parent, run_dir):
            if root is None:
                continue
            path = Path(root) / cfg["train_file"]
            if path.exists():
                n_train = sum(1 for l in path.open(encoding="utf-8") if l.strip())
                break

    by_ds: dict = {}
    for r in preds:
        reader = _READER.get(r["dataset"], extract_label)
        got = reader(r.get("generated") or "")
        r["predicted"] = got
        r["correct"] = bool(got is not None
                            and answers_match(got, r["answer"], r["dataset"]))
        r["n_steps"] = count_steps(r.get("generated") or "")
        r["n_tokens"] = count_tokens(r.get("generated") or "")
        by_ds.setdefault(r["dataset"], []).append(r)

    per_dataset = {d: _metrics(rs) for d, rs in sorted(by_ds.items())}
    side = "CRAFT" if "craft" in run_dir.name else "Raw_CoT"
    seed = run_dir.name.rsplit("s", 1)[-1]
    name = f"{_slug(cfg.get('model'))}_SFT_{side}_Seed{seed}"
    return {
        "run": name, "side": "craft" if side == "CRAFT" else "raw", "seed": int(seed),
        "model": cfg.get("model"), "train_file": cfg.get("train_file"),
        "n_train": n_train, "n_test": len(preds),
        "epochs": cfg.get("epochs"), "lr": cfg.get("lr"), "lora_r": cfg.get("lora_r"),
        "accuracy": round(sum(r["correct"] for r in preds) / max(len(preds), 1), 4),
        "avg_steps": round(sum(r["n_steps"] for r in preds) / max(len(preds), 1), 2),
        "by_dataset": per_dataset,
        "predictions": preds,
    }


def main(argv: list) -> int:
    train_root = None
    if "--train_root" in argv:
        i = argv.index("--train_root")
        train_root = Path(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    if not argv:
        print(__doc__)
        return 1
    for arg in argv:
        run = Path(arg)
        if not (run / "predictions.jsonl").exists():
            print(f"  {run}: no predictions.jsonl yet")
            continue
        s = score(run, train_root)
        out = run / f"{s['run']}.json"
        out.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {s['run']}   accuracy {100*s['accuracy']:.1f}  "
              f"steps {s['avg_steps']:.1f}  n={s['n_test']}")
        for d, m in s["by_dataset"].items():
            f1 = "  —  " if m["macro_f1"] is None else f"{m['macro_f1']:.3f}"
            print(f"      {d:<16} acc {100*m['accuracy']:5.1f}   F1 {f1}   "
                  f"steps {m['avg_steps']:5.1f}   n {m['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
