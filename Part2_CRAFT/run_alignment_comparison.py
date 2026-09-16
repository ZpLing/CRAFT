#!/usr/bin/env python3
"""
run_alignment_comparison.py
────────────────────────────
End-to-end comparison: RKG baseline vs. cross-trace aligned RKG.

Pipeline per mode:
  cleaned_traces → build_rkg (baseline|aligned) → synthesize (rkg) → label accuracy

Prints comparison table against existing Setting E result.

Usage:
  python run_alignment_comparison.py \
    --cleaned  "../../reasoning pipeline code2/fld_gpt54nano_100/cleaned_traces_z-1.0.json" \
    --original ../../FLD.json \
    --model gpt-5.4-nano \
    --n_samples 100 \
    --threshold 0.3 \
    --output_dir alignment_comparison
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

sys.path.insert(0, str(Path(__file__).parent))

from config import require_bosch, resolve_input, resolve_output
import module2_rkg_filtering.build_rkg as _rkg_mod
from module2_rkg_filtering.build_rkg import build_rkgs_for_sample
from module3_synthesis.synthesize_trace import synthesize_traces_for_dataset

# Pinned endpoint, read from the gitignored repo-root config.py.
# Explicit rather than via OPENAI_* so a stray env var cannot redirect these runs.
BOSCH_KEY, BOSCH_URL = require_bosch()

EXISTING_SETTING_E = {"accuracy": 0.560, "macro_f1": 0.494, "label": "E: DAG (gpt54nano, old pipeline)"}


def patch_credentials():
    _rkg_mod.OPENAI_API_KEY  = BOSCH_KEY
    _rkg_mod.OPENAI_BASE_URL = BOSCH_URL
    _rkg_mod.CHAT_URL        = BOSCH_URL.rstrip("/") + "/chat/completions"
    _rkg_mod.HEADERS         = {"Authorization": f"Bearer {BOSCH_KEY}",
                                "Content-Type": "application/json"}
    _rkg_mod.REQUEST_TIMEOUT = 60     # tighter than default 120


def compute_accuracy(synth_results: List[Dict], samples: List[Dict]) -> Dict[str, float]:
    """Compute label accuracy and macro-F1 from synthesized traces."""
    target_map = {s["sample_id"]: s.get("target_answer", "") for s in samples}

    correct = 0
    total   = 0
    per_class: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for r in synth_results:
        sid   = r.get("sample_id")
        pred  = r.get("pred_label", "")
        truth = target_map.get(sid, "")
        if not truth or not pred:
            continue
        total += 1
        if pred == truth:
            correct += 1
            per_class[truth]["tp"] += 1
        else:
            per_class[pred]["fp"]   += 1
            per_class[truth]["fn"]  += 1

    accuracy = correct / total if total else 0.0

    f1s = []
    for cls, counts in per_class.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return {"accuracy": round(accuracy, 4), "macro_f1": round(macro_f1, 4),
            "n_total": total, "n_correct": correct}


async def build_rkg_for_all(
    samples: List[Dict],
    model: str,
    threshold: float,
    use_alignment: bool,
    concurrency: int = 5,
) -> List[Dict]:
    patch_credentials()
    mode = "aligned" if use_alignment else "baseline"
    print(f"\n── Building RKG ({mode}) ── model={model} n={len(samples)}")

    semaphore = asyncio.Semaphore(concurrency)
    results   = [None] * len(samples)

    connector = aiohttp.TCPConnector(limit=concurrency * 3)
    async with aiohttp.ClientSession(connector=connector) as session:
        async def worker(idx: int, sample: Dict):
            async with semaphore:
                try:
                    results[idx] = await build_rkgs_for_sample(
                        session, sample, model=model,
                        consensus_threshold=threshold,
                        use_alignment=use_alignment,
                    )
                except Exception as e:
                    results[idx] = {
                        "sample_id":   sample.get("sample_id", f"err_{idx}"),
                        "error":       str(e),
                        "trace_dags":  [],
                        "consensus_dag": {},
                        "alignment":   {"used": use_alignment},
                    }
                done = sum(1 for r in results if r is not None)
                if done % 10 == 0 or done == len(samples):
                    print(f"  {done}/{len(samples)}", flush=True)

        await asyncio.gather(*[asyncio.create_task(worker(i, s))
                                for i, s in enumerate(samples)])

    return [r for r in results if r is not None]


def _build_problem_map(original_path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    """Build {sample_id -> {problem_text, ground_truth}} from FLD/FOLIO json (flat format).

    Cleaned-trace sample_ids are positional ('FLD_0' = first in FLD.json), so we
    map by enumeration index.
    """
    if not original_path or not original_path.exists():
        return {}
    with open(original_path) as f:
        orig = json.load(f)
    rows = orig if isinstance(orig, list) else orig.get("results", [])
    prefix = "FLD" if "FLD" in original_path.name.upper() else "FOLIO"
    pmap: Dict[str, Dict[str, str]] = {}
    for i, row in enumerate(rows):
        sid = f"{prefix}_{i}"
        text_parts = []
        if row.get("input"):
            text_parts.append(str(row["input"]).strip())
        if row.get("Facts") and "Fact" not in (row.get("input") or ""):
            text_parts.append(str(row["Facts"]).strip())
        if row.get("Conclusion"):
            text_parts.append(f"Hypothesis: {row['Conclusion']}".strip())
        ptxt = "\n".join(p for p in text_parts if p)
        gt = row.get("proof_label") or row.get("Label")
        if ptxt:
            pmap[sid] = {"problem_text": ptxt, "ground_truth": gt}
    return pmap


def _inject_problem_text(samples: List[Dict], pmap: Dict[str, Dict[str, str]]) -> int:
    """Inject problem_text/ground_truth into each cleaned-trace sample. Returns # injected."""
    n = 0
    for s in samples:
        sid = s.get("sample_id")
        if sid and sid in pmap and not s.get("problem_text"):
            s["problem_text"] = pmap[sid]["problem_text"]
            if not s.get("target_answer"):
                s["target_answer"] = pmap[sid].get("ground_truth")
            n += 1
    return n


async def main_async(args: argparse.Namespace):
    out_dir = resolve_output(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load cleaned traces
    with open(resolve_input(args.cleaned)) as f:
        raw = json.load(f)
    all_samples = raw if isinstance(raw, list) else raw.get("results", [])
    samples = all_samples[: args.n_samples]

    # Inject problem_text from FLD.json (flat format not auto-recognized by synth loader)
    orig_path = resolve_input(args.original) if args.original else None
    pmap = _build_problem_map(orig_path)
    n_inj = _inject_problem_text(samples, pmap)
    print(f"Loaded {len(samples)} samples | model={args.model} | injected problem_text into {n_inj}")

    # Save preprocessed cleaned-traces (synth reads from this)
    cleaned_aug = out_dir / "cleaned_with_problem.json"
    with open(cleaned_aug, "w") as f:
        json.dump({"results": samples}, f)
    cleaned_path = cleaned_aug

    # Patch synthesize_trace credentials (must match exact variable names)
    import module3_synthesis.synthesize_trace as _synth_mod
    _synth_mod.OPENAI_API_KEY        = BOSCH_KEY
    _synth_mod.OPENAI_BASE_URL       = BOSCH_URL
    _synth_mod.CHAT_COMPLETIONS_URL  = BOSCH_URL.rstrip("/") + "/chat/completions"
    _synth_mod.HEADERS["Authorization"] = f"Bearer {BOSCH_KEY}"
    _synth_mod.DEFAULT_MODEL         = args.model
    _synth_mod.REQUEST_TIMEOUT       = 60     # tighter (was 180) — Bosch can hang for hours
    # Replace generate_reasoning_trace with a low-retry version (default has max_tries=7)
    import aiohttp as _aiohttp, asyncio as _asyncio, backoff as _backoff
    @_backoff.on_exception(_backoff.expo, (_aiohttp.ClientError, _asyncio.TimeoutError, RuntimeError),
                           max_tries=3, factor=2, max_value=20)
    async def _gen(session, prompt, model=_synth_mod.DEFAULT_MODEL, max_tokens=None):
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "temperature": _synth_mod.REQUEST_TEMPERATURE,
                   "max_tokens": max_tokens or _synth_mod.RESPONSE_TOKENS}
        async with session.post(_synth_mod.CHAT_COMPLETIONS_URL, json=payload,
                                headers=_synth_mod.HEADERS,
                                timeout=_aiohttp.ClientTimeout(total=_synth_mod.REQUEST_TIMEOUT)) as resp:
            if resp.status != 200:
                detail = await resp.text()
                raise RuntimeError(f"HTTP {resp.status}: {detail[:200]}")
            data = await resp.json()
            return data["choices"][0]["message"].get("content", "") or ""
    _synth_mod.generate_reasoning_trace = _gen

    all_results = {}

    mode_pairs = [(False, "baseline"), (True, "aligned")]
    mode_pairs = [(u, m) for u, m in mode_pairs if m in args.modes]
    for use_aln, mode in mode_pairs:

        # Step 1: Build RKG (skip if file already exists)
        rkg_path = out_dir / f"rkg_{mode}.json"
        if rkg_path.exists() and not args.force_rkg:
            print(f"\n── Reusing existing RKG ({mode}) → {rkg_path}")
            with open(rkg_path) as f:
                rkg_raw = json.load(f)
            rkg_data = rkg_raw if isinstance(rkg_raw, list) else rkg_raw.get("results", [])
        else:
            rkg_data = await build_rkg_for_all(
                samples, args.model, args.threshold, use_alignment=use_aln,
            )
            with open(rkg_path, "w") as f:
                json.dump({"results": rkg_data}, f, indent=2)
            print(f"  Saved RKG → {rkg_path}")

        # Alignment stats
        if use_aln:
            aln_infos = [r.get("alignment", {}) for r in rkg_data if r.get("alignment", {}).get("used")]
            if aln_infos:
                avg_shared = sum(a.get("n_shared_groups", 0) for a in aln_infos) / len(aln_infos)
                avg_ratio  = sum(a.get("shared_ratio", 0)   for a in aln_infos) / len(aln_infos)
                print(f"  Alignment: avg shared_groups={avg_shared:.1f}, avg shared_ratio={avg_ratio:.3f}")

        # Step 2: Synthesize
        suffix = f"_{args.tag}" if args.tag else ""
        synth_path = out_dir / f"synthesized_{mode}{suffix}.json"
        await synthesize_traces_for_dataset(
            input_file=cleaned_path,
            output_file=synth_path,
            model=args.model,
            concurrency=args.concurrency,
            domain="logical",
            synthesis_strategy="rkg",
            rkg_file=rkg_path,
            max_samples=args.n_samples,
            original_file=orig_path,
            anchor_conclusion=args.anchor_conclusion,
        )

        # Step 3: Accuracy
        with open(synth_path) as f:
            synth_raw = json.load(f)
        synth_results = synth_raw if isinstance(synth_raw, list) else synth_raw.get("results", [])
        acc = compute_accuracy(synth_results, samples)
        all_results[mode] = acc
        print(f"  [{mode}] acc={acc['accuracy']:.3f} f1={acc['macro_f1']:.3f} "
              f"({acc['n_correct']}/{acc['n_total']})")

    # Final comparison table
    print("\n" + "=" * 62)
    print(f"{'Setting':<30} {'Accuracy':>9} {'Macro-F1':>9}")
    print("-" * 62)
    e = EXISTING_SETTING_E
    print(f"{e['label']:<30} {e['accuracy']:>9.3f} {e['macro_f1']:>9.3f}")
    for mode, acc in all_results.items():
        label = f"CRAFT {mode} ({args.model})"
        print(f"{label:<30} {acc['accuracy']:>9.3f} {acc['macro_f1']:>9.3f}")
    print("=" * 62)

    with open(out_dir / "comparison_summary.json", "w") as f:
        json.dump({"existing_E": EXISTING_SETTING_E, "craft": all_results,
                   "model": args.model, "n_samples": len(samples)}, f, indent=2)
    print(f"\nSaved summary → {out_dir}/comparison_summary.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cleaned",     required=True)
    p.add_argument("--original",    default=None, help="Original dataset (FLD.json) for problem text")
    p.add_argument("--model",       default="gpt-5.4-nano")
    p.add_argument("--n_samples",   type=int, default=100)
    p.add_argument("--threshold",   type=float, default=0.3)
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--output_dir",  default="alignment_comparison",
                   help="Relative paths resolve under the results root")
    p.add_argument("--force_rkg",   action="store_true", help="Rebuild RKG even if file exists")
    p.add_argument("--anchor_conclusion", action="store_true",
                   help="Use RKG consensus conclusion node text directly (no LLM rewrite)")
    p.add_argument("--tag",         default="", help="Suffix for synthesis output file (e.g. 'anchor', 'thr0.2')")
    p.add_argument("--modes",       nargs="+", default=["baseline", "aligned"],
                   choices=["baseline", "aligned"], help="Which modes to run")
    asyncio.run(main_async(p.parse_args()))

if __name__ == "__main__":
    main()
