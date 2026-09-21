#!/usr/bin/env python3
"""What the two backbones were actually billed for, per stage and per run.

Every stage writes how many requests it made, and most of them keep the text
that came back, so the output side is counted rather than guessed: the k traces
keep raw_response, Module III keeps synthesized_trace, and a baseline keeps its
traces file. Module II is the exception -- it stores the graph it parsed and not
the reply it parsed it from -- so its output is measured from the serialised
graph, which is the bulk of what the model wrote.

The input side is never stored. It is reconstructed per stage from the material
the prompt is built out of, plus a fixed allowance for the instruction text:

    generation   problem text + system prompt, once per requested trace
    RKG          the trace being read, once per trace
    synthesis    problem text + the consensus node texts, once per sample

Retries and refusals are in api_calls but not in the stored text, so a stage
whose calls exceed its stored replies has its input scaled by the call count and
its output left at what was returned. That is the honest direction to be wrong
in: unreturned calls still cost their input.

Counting is tiktoken cl100k_base, which is not either backbone's tokenizer;
treat the totals as accurate to within the usual few percent between tokenizers.
Prices are not shipped here because the two model names are deployment aliases
on a gateway -- pass the ones you are billed at:

    python token_accounting.py --price gpt-5.4-nano=0.05,0.40 \\
                               --price gemini-3.1-flash-lite=0.075,0.30

where each pair is USD per million input,output tokens.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")

    def ntok(text: Optional[str]) -> int:
        if not text:
            return 0
        return len(_ENC.encode(text, disallowed_special=()))
except Exception:                                    # pragma: no cover
    def ntok(text: Optional[str]) -> int:
        return (len(text) + 3) // 4 if text else 0


# Instruction text that every call carries and none of the outputs record.
SYSTEM_ALLOWANCE = {"generation": 120, "rkg": 260, "synthesis": 220,
                    "baseline": 120}

_P2 = Path(__file__).resolve().parents[1]


def _rows(obj: Any) -> List[Dict]:
    return obj["results"] if isinstance(obj, dict) and "results" in obj else obj


def _meta(obj: Any) -> Dict:
    return obj.get("metadata", {}) if isinstance(obj, dict) else {}


def _read(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def account_k_traces(path: Path) -> Optional[Dict]:
    obj = _read(path)
    if obj is None:
        return None
    meta, rows = _meta(obj), _rows(obj)
    out = inp = stored = 0
    for r in rows:
        prob = ntok(r.get("problem_text")) + SYSTEM_ALLOWANCE["generation"]
        for t in r.get("traces") or []:
            stored += 1
            out += ntok(t.get("raw_response") or t.get("reasoning_text"))
            inp += prob
    calls = meta.get("api_calls") or stored
    if stored and calls > stored:                    # retries paid their input
        inp = round(inp * calls / stored)
    return {"stage": "1 generation", "model": meta.get("model"),
            "calls": calls, "in": inp, "out": out}


def account_rkg(path: Path) -> Optional[Dict]:
    obj = _read(path)
    if obj is None:
        return None
    meta, rows = _meta(obj), _rows(obj)
    out = inp = stored = 0
    for r in rows:
        for g in r.get("trace_rkgs") or []:
            stored += 1
            body = json.dumps({"nodes": g.get("nodes", []),
                               "edges": g.get("edges", [])}, ensure_ascii=False)
            out += ntok(body)
            # the prompt is the trace this graph was read from
            inp += sum(ntok(n.get("text")) for n in g.get("nodes") or [])
            inp += SYSTEM_ALLOWANCE["rkg"]
    calls = meta.get("api_calls") or stored
    if stored and calls > stored:
        inp = round(inp * calls / stored)
    return {"stage": "2 rkg", "model": meta.get("model"),
            "calls": calls, "in": inp, "out": out}


def account_synth(path: Path, rkg_path: Optional[Path]) -> Optional[Dict]:
    obj = _read(path)
    if obj is None:
        return None
    meta, rows = _meta(obj), _rows(obj)
    graph_tokens: Dict[str, int] = {}
    if rkg_path and rkg_path.exists():
        robj = _read(rkg_path)
        for r in _rows(robj or []):
            c = (r.get("consensus_rkg") or {})
            texts = c.get("node_texts") or {}
            graph_tokens[r.get("sample_id")] = sum(ntok(t) for t in texts.values())
    out = inp = stored = 0
    for r in rows:
        if r.get("error"):
            continue
        stored += 1
        out += ntok(r.get("synthesized_trace"))
        inp += graph_tokens.get(r.get("sample_id"), 0) + SYSTEM_ALLOWANCE["synthesis"]
    calls = meta.get("api_calls") or stored
    if stored and calls > stored:
        inp = round(inp * calls / stored)
    return {"stage": "3 synthesis", "model": meta.get("model"),
            "calls": calls, "in": inp, "out": out}


def account_baseline(setting_dir: Path) -> Iterable[Dict]:
    res = setting_dir / "results.json"
    obj = _read(res)
    if obj is None:
        return
    model = (obj.get("run") or {}).get("model")
    out = inp = n = 0
    traces = setting_dir / "traces.jsonl"
    if traces.exists():
        with traces.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                for t in (rec.get("traces") or []):
                    out += ntok(t if isinstance(t, str) else json.dumps(t))
                    n += 1
                inp += (ntok(rec.get("problem_text"))
                        + SYSTEM_ALLOWANCE["baseline"]) * max(len(rec.get("traces") or []), 1)
    if not n:                                        # fall back to the summary
        ov = obj.get("overall") or {}
        n = ov.get("n_total") or len(obj.get("predictions") or [])
        out = round((ov.get("avg_tokens") or 0) * n)
        inp = n * 400
    yield {"stage": f"baseline/{setting_dir.name}", "model": model,
           "calls": n, "in": inp, "out": out}


def walk(root: Path) -> List[Dict]:
    recs: List[Dict] = []
    runs = root / "results" / "craft_runs"
    if runs.is_dir():
        for d in sorted(p for p in runs.rglob("*") if p.is_dir()):
            kt = next(iter(sorted(d.glob("k_traces*.json"))), None)
            if kt:
                r = account_k_traces(kt)
                if r:
                    r["run"] = d.name
                    recs.append(r)
            rkg = d / "rkg.json"
            if rkg.exists():
                r = account_rkg(rkg)
                if r:
                    r["run"] = d.name
                    recs.append(r)
            for sp in sorted(d.glob("syn*.json")):
                r = account_synth(sp, rkg if rkg.exists() else None)
                if r:
                    r["run"] = f"{d.name}/{sp.stem}"
                    recs.append(r)
    base = root / "results" / "baseline_results"
    if base.is_dir():
        for model_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            for setting in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                for r in account_baseline(setting):
                    r["run"] = f"{model_dir.name}/{setting.name}"
                    recs.append(r)
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=_P2)
    ap.add_argument("--price", action="append", default=[],
                    metavar="MODEL=IN,OUT",
                    help="USD per million input,output tokens for a model")
    ap.add_argument("--by", choices=["model", "stage", "run"], default="model")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()

    prices: Dict[str, Tuple[float, float]] = {}
    for spec in a.price:
        name, _, pair = spec.partition("=")
        pin, _, pout = pair.partition(",")
        prices[name.strip()] = (float(pin), float(pout))

    recs = walk(a.root)
    if not recs:
        raise SystemExit(f"nothing to count under {a.root}")

    agg: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
        lambda: {"calls": 0, "in": 0, "out": 0})
    for r in recs:
        model = r.get("model") or "unknown"
        key = (model, r["stage"] if a.by == "stage" else
               (r["run"] if a.by == "run" else ""))
        t = agg[key]
        t["calls"] += r["calls"]; t["in"] += r["in"]; t["out"] += r["out"]

    M = 1_000_000
    print(f"\n  {'model':<24} {'':<34} {'calls':>9} {'in (M)':>9} "
          f"{'out (M)':>9} {'total (M)':>10} {'USD':>9}")
    print("  " + "-" * 108)
    tot = {"calls": 0, "in": 0, "out": 0, "usd": 0.0}
    for (model, sub), t in sorted(agg.items()):
        pin, pout = prices.get(model, (0.0, 0.0))
        usd = t["in"] / M * pin + t["out"] / M * pout
        tot["calls"] += t["calls"]; tot["in"] += t["in"]
        tot["out"] += t["out"]; tot["usd"] += usd
        print(f"  {model:<24} {sub:<34} {t['calls']:>9,} {t['in']/M:>9.2f} "
              f"{t['out']/M:>9.2f} {(t['in']+t['out'])/M:>10.2f} "
              f"{('$'+format(usd, ',.2f')) if usd else '—':>9}")
    print("  " + "-" * 108)
    print(f"  {'TOTAL':<24} {'':<34} {tot['calls']:>9,} {tot['in']/M:>9.2f} "
          f"{tot['out']/M:>9.2f} {(tot['in']+tot['out'])/M:>10.2f} "
          f"{('$'+format(tot['usd'], ',.2f')) if tot['usd'] else '—':>9}")
    if not prices:
        print("\n  no --price given, so no cost column; pass "
              "--price <model>=<in>,<out> in USD per million tokens")

    if a.json:
        a.json.write_text(json.dumps(recs, indent=2), encoding="utf-8")
        print(f"  → {a.json}")


if __name__ == "__main__":
    main()
