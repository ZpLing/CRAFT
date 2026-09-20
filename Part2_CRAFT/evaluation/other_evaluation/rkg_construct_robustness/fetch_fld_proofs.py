#!/usr/bin/env python3
"""
fetch_fld_proofs.py — add FLD's published proofs to the FLD files in this repo.

The robustness check scores edge extraction against the proof each problem was
generated from, and the FLD files here carry the premises, the hypothesis and
the label but not that proof. Upstream publishes it, and every sample records
where it came from in `original_index` ("train_26546"), so the proof can be
looked up rather than re-sampled: the 500 samples and their label balance stay
exactly as they are, and each gains one key.

Every FLD file in the repo is updated in the same pass. They are meant to be the
same 500 samples, and a proof added to one copy but not another would make which
copy an experiment read matter.

Nothing is written unless every sample matches upstream on both its hypothesis
and its label. A silent index shift would attach the wrong proof to each problem
and the check that depends on it would measure nothing, so a mismatch stops the
run and names the sample.

Usage:
    python fetch_fld_proofs.py                  # update every FLD.json under dataset/
    python fetch_fld_proofs.py --check          # verify alignment, write nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
try:
    from config import DATASET_ROOT
except ImportError:
    DATASET_ROOT = Path(__file__).resolve().parents[3] / "dataset"

HF_DATASET = "hitachi-nlp/FLD.v2"
SPLITS = ("train", "validation", "test")


def load_upstream() -> Dict[str, Any]:
    try:
        import datasets
    except ImportError:
        raise SystemExit("This needs the datasets package: pip install -r requirements.txt")
    datasets.disable_progress_bars()
    return {s: datasets.load_dataset(HF_DATASET, split=s) for s in SPLITS}


def locate(sample: Dict[str, Any], upstream: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The upstream row a sample was drawn from, by its recorded origin."""
    origin = str(sample.get("original_index", ""))
    split, _, index = origin.rpartition("_")
    if split not in upstream or not index.isdigit():
        return None
    i = int(index)
    return upstream[split][i] if i < len(upstream[split]) else None


def agrees(sample: Dict[str, Any], row: Dict[str, Any]) -> bool:
    """Same problem on both sides. Upstream writes PROVED, this repo __PROVED__."""
    label_ok = f"__{row.get('proof_label')}__" == sample.get("proof_label")
    hypothesis_ok = (row.get("hypothesis") or "").strip() == (sample.get("Conclusion") or "").strip()
    return label_ok and hypothesis_ok


def fld_files(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("FLD.json"))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_root", default=None,
                    help=f"Where to look for FLD.json (default: {DATASET_ROOT})")
    ap.add_argument("--check", action="store_true",
                    help="Report alignment and write nothing")
    args = ap.parse_args()

    root = Path(args.dataset_root) if args.dataset_root else Path(DATASET_ROOT)
    targets = fld_files(root)
    if not targets:
        raise SystemExit(f"No FLD.json under {root}")
    print(f"\n  {len(targets)} FLD file(s) under {root}:")
    for p in targets:
        print(f"    {p.relative_to(root)}")

    print(f"\n  Loading {HF_DATASET} ...")
    upstream = load_upstream()

    for path in targets:
        with open(path, encoding="utf-8") as f:
            samples = json.load(f)

        proofs: List[Optional[List[str]]] = []
        mismatched: List[Tuple[int, str]] = []
        for i, sample in enumerate(samples):
            row = locate(sample, upstream)
            if row is None:
                mismatched.append((i, f"origin {sample.get('original_index')!r} not found"))
                proofs.append(None)
            elif not agrees(sample, row):
                mismatched.append((i, f"origin {sample.get('original_index')!r} is a different problem"))
                proofs.append(None)
            else:
                proofs.append(list(row.get("proofs") or []))

        rel = path.relative_to(root)
        if mismatched:
            print(f"\n  {rel}: {len(mismatched)} of {len(samples)} do not match upstream")
            for i, why in mismatched[:5]:
                print(f"    sample {i}: {why}")
            raise SystemExit("  Nothing written. Fix the alignment before adding proofs.")

        n_empty = sum(1 for p in proofs if not p)
        print(f"\n  {rel}: {len(samples)}/{len(samples)} match upstream"
              f"{f', {n_empty} have no published proof' if n_empty else ''}")
        if args.check:
            continue

        for sample, proof in zip(samples, proofs):
            sample["proofs"] = proof
        with open(path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=1)
        print(f"    added `proofs` to {path}")

    if args.check:
        print("\n  --check: nothing written.")
    print()


if __name__ == "__main__":
    main()
