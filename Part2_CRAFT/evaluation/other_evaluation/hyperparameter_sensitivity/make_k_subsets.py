#!/usr/bin/env python3
"""Runs at smaller K, taken from the K=5 run rather than generated again.

The K sweep needs one finished run per K. Generating them costs K traces a
sample all over again, and it would also change what is being measured: two
runs generated separately differ in their traces as well as in K, and the
sweep is supposed to isolate K.

A run at K=k is the K=5 run with the last 5-k traces dropped. The traces that
remain are the same traces, sampled at the same temperature, in the same order,
so the only thing that differs between two points on the sweep is how many of
them the consensus had to work with. Nothing is generated; the consensus is
rebuilt from the trace graphs already stored, which needs no model either.

    python make_k_subsets.py --run craft_runs/full500/FLD_gpt-5.4-nano --k 2 3 4
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import config as _cfg  # noqa: E402


def take(rows, k, key):
    out = []
    for r in rows:
        r = dict(r)
        traces = r.get(key) or []
        r[key] = [t for t in traces if t.get("trace_idx", 0) < k]
        out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--k", type=int, nargs="+", required=True)
    ap.add_argument("--out_root", default=None,
                    help="Default: sibling directories named <run>_k<k>")
    args = ap.parse_args()

    src = Path(_cfg.resolve_input(args.run))
    k_file = next(iter(sorted(src.glob("k_traces_*_samples.json"))), None)
    if k_file is None:
        raise FileNotFoundError(f"no k_traces_*.json in {src}")

    for k in args.k:
        dst = (Path(args.out_root) / f"{src.name}_k{k}" if args.out_root
               else src.parent / f"{src.name}_k{k}")
        dst.mkdir(parents=True, exist_ok=True)

        raw = json.loads(k_file.read_text(encoding="utf-8"))
        rows = raw.get("results", raw) if isinstance(raw, dict) else raw
        cut = take(rows, k, "traces")
        body = {**raw, "results": cut} if isinstance(raw, dict) else cut
        (dst / k_file.name).write_text(json.dumps(body), encoding="utf-8")

        # the cleaned file the synthesis reads, cut the same way
        for name in ("cleaned_z.json", "cleaned.json"):
            p = src / name
            if not p.exists():
                continue
            raw2 = json.loads(p.read_text(encoding="utf-8"))
            rows2 = raw2.get("results", raw2) if isinstance(raw2, dict) else raw2
            cut2 = take(rows2, k, "cleaned_traces")
            cut2 = take(cut2, k, "original_traces")
            body2 = {**raw2, "results": cut2} if isinstance(raw2, dict) else cut2
            (dst / name).write_text(json.dumps(body2), encoding="utf-8")

        # the trace graphs are already extracted; drop the ones past k and let
        # build_rkg --rebuild_consensus recompute the consensus without a model
        rk = src / "rkg.json"
        if rk.exists():
            raw3 = json.loads(rk.read_text(encoding="utf-8"))
            rows3 = raw3.get("results", raw3) if isinstance(raw3, dict) else raw3
            cut3 = take(rows3, k, "trace_rkgs")
            body3 = {**raw3, "results": cut3} if isinstance(raw3, dict) else cut3
            (dst / "rkg.json").write_text(json.dumps(body3), encoding="utf-8")

        n = sum(len(r.get("traces") or []) for r in cut)
        print(f"  K={k}: {dst}  ({n} traces over {len(cut)} samples)")


if __name__ == "__main__":
    main()
