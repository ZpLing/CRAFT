#!/usr/bin/env python
"""
finelogic_eval_steps.py  (FineLogic step evaluation of CRAFT traces)

Adapted from FineLogic's eval_step.py to handle the natural-language
Chain-of-Thought traces this pipeline produces. Judges each step Valid /
Relevant / Atomic with an LLM; the paper uses Gemini-3.1-Flash-Lite.

KEY DIFFERENCES vs FineLogic's eval_step.py (logic unchanged, only parsing changed):

1. Step format:
   - FineLogic expects each step to END with `intK: <conclusion>` / `assumpK: ...` / `hypothesis: ...`
   - Our traces just have `Step N:` headers + natural-language bodies; conclusion is
     embedded in the body (e.g. "...so Whiskers is an animal.")

2. References inside a step body:
   - FineLogic recognises `factN` / `intN` / `assumpN`
   - Ours recognises `factN` / `stepN`   (our steps cite earlier steps as "Step K")

3. Conclusion handling:
   - FineLogic looks up conclusion text from a ref dict via the `intK`/`hypothesis` id
   - Ours uses the step's FULL body text directly as the conclusion

4. Necessary-check:
   - FineLogic: is `concl` (e.g. `int3`) referenced in any later step's `ante`?
   - Ours:      is `stepN` referenced in any later step's `ante`? Plus the LAST step is
     always necessary (it states the final answer).

The Valid / Necessary / Atomic semantics are identical to the original paper —
we only swap the regexes and text-extraction logic.
"""

import os, re, json, asyncio
from collections import defaultdict, Counter
import aiohttp, backoff
from tqdm import tqdm

#########################
# Configuration          #
#########################
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
try:
    from config import OPENAI_API_KEY, OPENAI_BASE_URL, resolve_input, resolve_output
except ImportError:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    resolve_input = resolve_output = _Path

if not OPENAI_API_KEY:
    raise RuntimeError("No API key: set OPENAI_API_KEY or put one in the repo-root config.py")
HEADERS = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
OPENAI_BASE_URL = OPENAI_BASE_URL.rstrip("/")
CHAT_URL = f"{OPENAI_BASE_URL}/chat/completions"
# The judge, in preference order — a model that errors falls through to the next.
_models_env = os.getenv("EVAL_MODELS")
MODELS = ([m.strip() for m in _models_env.split(",")] if _models_env
          else ["gemini-3.1-flash-lite", "gpt-5.4-nano"])

# ---- Step header ----
STEP_RE = re.compile(r"^\s*Step\s*(\d+)\s*[:\.]", re.I | re.M)

# ---- Fact definition line from problem input ----
# matches "Fact1: ...", "Fact 1: ..."; splits on ;  \n  .
FACT_LINE_RE = re.compile(r"\bfact\s*(\d+)\s*[:\.]?\s*(.+?)(?=$|;)", re.I)

# ---- References inside a step body ----
# Catches "fact1", "Fact 1", "step3", "Step 3"
REF_RE_NL = re.compile(r"\b(fact\s*\d+|step\s*\d+)\b", re.I)
# Catches multi-step citations like "Steps 1 and 2", "Steps 1-3", "Steps 1, 2, 3"
MULTI_STEP_RANGE_RE = re.compile(r"\bsteps?\s*(\d+)\s*[-–]\s*(\d+)\b", re.I)
MULTI_STEP_LIST_RE  = re.compile(r"\bsteps?\s*([\d,\s]*(?:\band\b\s*\d+)?)\b", re.I)

# ---- Final-answer markers (marks a step as "final", always necessary) ----
FINAL_MARKER_RE = re.compile(
    r"(final\s+conclusion|__\s*PROVED\s*__|__\s*DISPROVED\s*__|__\s*UNKNOWN\s*__|therefore,\s*the\s*hypothesis)",
    re.I,
)


############################################
# GPT helper (identical to eval_step.py)    #
############################################

def _chat(session, msgs, model):
    payload = {"model": model, "messages": msgs, "temperature": 0.0}
    return session.post(CHAT_URL, json=payload, headers=HEADERS)


@backoff.on_exception(backoff.expo, (aiohttp.ClientError, asyncio.TimeoutError), max_tries=5)
@backoff.on_exception(backoff.expo,
                      (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError),
                      max_tries=7, factor=2)
async def ask_bool(session, msgs):
    for model in MODELS:
        try:
            async with _chat(session, msgs, model) as resp:
                if resp.status != 200:
                    detail = await resp.text()
                    print(f"[warn] {model} HTTP {resp.status}: {detail[:80]}")
                    raise RuntimeError("non-200 response")
                data = await resp.json()
                content = data["choices"][0]["message"]["content"].strip().lower()
                if content.startswith(("true", "yes", "1")):
                    return True
                if content.startswith(("false", "no", "0")):
                    return False
                print(f"[warn] {model} unexpected answer → '{content[:60]}'")
                raise RuntimeError("unparseable answer")
        except Exception as exc:
            print(f"[info] {model} failed ({exc}). trying next model…")
            continue
    return False


############################################
# Parsing helpers (adapted for NL format)   #
############################################

def normalise_ref(s: str) -> str:
    """'Fact 1' → 'fact1'; 'Step 3' → 'step3'."""
    return re.sub(r"\s+", "", s).lower()


def parse_facts_from_input(problem_input: str) -> dict:
    """Extract {factN: <text>} from the problem_input (which always lists Fact1: ...)."""
    ref = {}
    if not problem_input:
        return ref
    # split on line breaks first
    lines = problem_input.replace(".", ".\n").splitlines()
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        m = re.match(r"\s*(fact\s*\d+)\s*[:\.]?\s*(.+)", ln, re.I)
        if m:
            key = normalise_ref(m.group(1))
            val = m.group(2).strip()
            if key not in ref and val:
                ref[key] = val
    return ref


def split_steps_nl(text: str):
    """Split a trace into steps; each step keeps its body as-is.

    Returns list of dicts:
      {"n": int, "body": str, "ante": set[str], "concl_id": str, "is_final": bool}
    """
    if not text:
        return []

    matches = list(STEP_RE.finditer(text))
    if not matches:
        # Fallback: treat entire text as one "step" so caller can still evaluate
        ante = {normalise_ref(m.group(1)) for m in REF_RE_NL.finditer(text)}
        return [{
            "n": 1,
            "body": text.strip(),
            "ante": ante,
            "concl_id": "step1",
            "is_final": bool(FINAL_MARKER_RE.search(text)),
        }]

    starts = [m.start() for m in matches] + [len(text)]
    raw_steps = []
    for i, m in enumerate(matches):
        n = int(m.group(1))
        block = text[starts[i]:starts[i + 1]].strip()
        # body minus the "Step N:" header (first line)
        body = re.sub(r"^\s*Step\s*\d+\s*[:\.]\s*", "", block, count=1, flags=re.I).strip()
        # Find references inside the body, excluding self-reference
        ante = set()
        for rm in REF_RE_NL.finditer(body):
            ref_id = normalise_ref(rm.group(1))
            if ref_id != f"step{n}":
                ante.add(ref_id)
        for rm in MULTI_STEP_RANGE_RE.finditer(body):
            lo, hi = int(rm.group(1)), int(rm.group(2))
            for k in range(lo, hi + 1):
                if k != n:
                    ante.add(f"step{k}")
        raw_steps.append({
            "n": n,
            "body": body,
            "ante": ante,
            "concl_id": f"step{n}",
            "is_final": bool(FINAL_MARKER_RE.search(block)),
        })

    # Optimization #1: dedup consecutive "bare Final Conclusion" marker-only steps
    # A step is "bare marker" if body is essentially just "Final Conclusion: __X__" with
    # no substantive reasoning (len <= 50 chars stripped, or matches only the marker).
    BARE_RE = re.compile(r"^\s*(?:final\s+conclusion\s*[:\.]?\s*)?__\s*(proved|disproved|unknown)\s*__\s*\.?\s*$", re.I)
    steps = []
    for st in raw_steps:
        is_bare = bool(BARE_RE.match(st["body"])) or (
            st["is_final"] and len(re.sub(r"\s+", " ", st["body"])) < 50
        )
        if is_bare and steps and (steps[-1].get("_bare") or steps[-1]["is_final"]):
            # consecutive duplicate markers — drop this one but keep the is_final flag on previous
            steps[-1]["is_final"] = True
            continue
        st["_bare"] = is_bare
        steps.append(st)
    return steps


def build_ref_dict_nl(sample, steps):
    """Build {factN: <input text>, stepN: <step body>} lookup."""
    ref = parse_facts_from_input(sample["problem"].get("input", ""))
    for st in steps:
        ref[st["concl_id"]] = st["body"]
    return ref


def is_necessary_nl(step, later_steps):
    """A step is necessary if its stepN id appears in any later step's ante.
    LAST step is always necessary (it carries the final answer)."""
    if not later_steps:
        return True
    my_id = step["concl_id"]
    return any(my_id in ls["ante"] for ls in later_steps)


############################################
# Core evaluation (same Valid/Atomic logic) #
############################################

async def eval_step_nl(session, step, ref, later_steps, all_steps, facts_text):
    """Evaluate a single step.

    Optimisations vs strict FineLogic:
    - Drop unresolvable refs (e.g. "step9" when trace has only 4 steps) from the cited list
      so the judge never sees "step9: ???" garbage premises.
    - Give the judge full context (all facts + all prior steps) in addition to the cited
      premises, so missing citations do not artificially break "Valid" judgments.
    - Atomicity is asked on the current step's body, with its cited premises inlined.
    - Skip bare Final-Conclusion marker steps (they are re-statements, not inferences;
      same convention FineLogic uses for assumpK steps).
    """
    # Skip bare-marker steps — they are not inferences
    if step.get("_bare"):
        return {"step": step["n"], "skip": True}

    n = step["n"]

    # --- cited premises (drop unresolvable) ---
    cited = []
    for r in sorted(step["ante"]):
        val = ref.get(r)
        if val:
            cited.append(f"{r}: {val[:400]}")
    cited_text = "\n".join(cited) if cited else "(none explicitly cited)"

    # --- prior steps (full bodies, up to step n-1) ---
    prior_bodies = []
    for s in all_steps:
        if s["n"] >= n:
            break
        prior_bodies.append(f"step{s['n']}: {s['body'][:300]}")
    prior_text = "\n".join(prior_bodies) if prior_bodies else "(none)"

    concl_text = step["body"][:800]

    # VALID: show full context (facts + prior steps + cited), ask if step follows
    v_prompt = [{"role": "user", "content":
        "You are judging the logical soundness of ONE step in a reasoning trace.\n\n"
        f"[Problem facts]\n{facts_text}\n\n"
        f"[Previously established steps]\n{prior_text}\n\n"
        f"[Step being evaluated, step{n}]\n{concl_text}\n\n"
        f"[Step's own cited premises]\n{cited_text}\n\n"
        "Does step"+str(n)+" logically follow from the problem facts and previously "
        "established steps? Answer true or false only."}]
    valid = await ask_bool(session, v_prompt)

    # Necessary: last step always necessary; else check if my step id is referenced later
    necessary = step["is_final"] or is_necessary_nl(step, later_steps)

    atomic = False
    if valid:
        a_prompt = [{"role": "user", "content":
            f"[Step body]\n{concl_text}\n\n"
            f"[Cited premises, if any]\n{cited_text}\n\n"
            "Does this step consist of a single atomic inference (one inference rule applied once), "
            "rather than combining multiple inferences or skipping intermediate steps? "
            "Answer true or false only."}]
        atomic = await ask_bool(session, a_prompt)

    return {
        "step": step["n"],
        "valid": valid,
        "necessary": necessary,
        "atomic": atomic,
        "skip": False,
    }


async def analyse_sample(session, sample, sid):
    txt = sample["responses"][0]["response"]
    original_data = sample["problem"].get("original_data")
    if isinstance(original_data, dict):
        ground_truth_steps = original_data.get("steps")
    else:
        ground_truth_steps = None
    steps = split_steps_nl(txt)
    if not steps:
        return {"sample_id": sid, "error": "no_step_marker_found",
                "ground_truth_steps": ground_truth_steps,
                "raw_first_200": txt[:200]}
    ref = build_ref_dict_nl(sample, steps)
    # Mark last step as final if no step carries the final answer explicitly
    has_final = any(s["is_final"] for s in steps)
    if not has_final:
        steps[-1]["is_final"] = True

    # Build facts_text once per sample from problem input
    facts_only = {k: v for k, v in ref.items() if k.startswith("fact")}
    facts_text = "\n".join(f"{k}: {v[:300]}" for k, v in sorted(facts_only.items()))
    if not facts_text:
        facts_text = sample["problem"].get("input", "")[:2000]

    results = {
        "sample_id": sid,
        "num_steps": len(steps),
        "ground_truth_steps": ground_truth_steps,
        "steps": []
    }
    for i, st in enumerate(steps):
        res = await eval_step_nl(session, st, ref, steps[i + 1:], steps, facts_text)
        results["steps"].append(res)
    return results


############################################
# Aggregation (identical semantics)         #
############################################

def aggregate(sample_results):
    bucket = defaultdict(lambda: Counter(valid=0.0, necessary=0.0, atomic=0.0, samples=0))
    tot_true = Counter(valid=0, necessary=0, atomic=0, total=0)
    sample_perfect = Counter(all_valid=0, all_necessary=0, all_atomic=0, all_three=0, total_samples=0)

    for samp in sample_results:
        if samp is None or samp.get("error"):
            continue
        non_skip = [st for st in samp["steps"] if not st.get("skip")]
        if not non_skip:
            continue
        sample_perfect["total_samples"] += 1
        if all(st["valid"] for st in non_skip):
            sample_perfect["all_valid"] += 1
        if all(st["necessary"] for st in non_skip):
            sample_perfect["all_necessary"] += 1
        if all(st["atomic"] for st in non_skip):
            sample_perfect["all_atomic"] += 1
        if (all(st["valid"] for st in non_skip)
                and all(st["necessary"] for st in non_skip)
                and all(st["atomic"] for st in non_skip)):
            sample_perfect["all_three"] += 1

        n_steps = len(non_skip)
        valid_rate = sum(st["valid"] for st in non_skip) / n_steps
        nec_rate   = sum(st["necessary"] for st in non_skip) / n_steps
        atom_rate  = sum(st["atomic"] for st in non_skip) / n_steps

        b = bucket[str(samp.get("ground_truth_steps"))]
        b["valid"] += valid_rate
        b["necessary"] += nec_rate
        b["atomic"] += atom_rate
        b["samples"] += 1

        tot_true["valid"]     += sum(st["valid"] for st in non_skip)
        tot_true["necessary"] += sum(st["necessary"] for st in non_skip)
        tot_true["atomic"]    += sum(st["atomic"] for st in non_skip)
        tot_true["total"]     += n_steps

    result = {
        str(k): {
            "valid": round(v["valid"] / v["samples"], 3),
            "necessary": round(v["necessary"] / v["samples"], 3),
            "atomic": round(v["atomic"] / v["samples"], 3),
        }
        for k, v in bucket.items() if v["samples"] > 0
    }

    if tot_true["total"]:
        result["overall_steps"] = {
            "valid":     round(tot_true["valid"] / tot_true["total"], 3),
            "necessary": round(tot_true["necessary"] / tot_true["total"], 3),
            "atomic":    round(tot_true["atomic"] / tot_true["total"], 3),
        }

    ts = sample_perfect["total_samples"] or 1
    result["overall_samples"] = {
        "total_samples": ts,
        "all_valid":     {"count": sample_perfect["all_valid"],     "ratio": round(sample_perfect["all_valid"]     / ts, 3)},
        "all_necessary": {"count": sample_perfect["all_necessary"], "ratio": round(sample_perfect["all_necessary"] / ts, 3)},
        "all_atomic":    {"count": sample_perfect["all_atomic"],    "ratio": round(sample_perfect["all_atomic"]    / ts, 3)},
        "all_three":     {"count": sample_perfect["all_three"],     "ratio": round(sample_perfect["all_three"]     / ts, 3)},
    }
    return result


############################################
# Pipeline runner (progress bar)            #
############################################

async def run_pipeline(inp, det, summ, concurrency=50):
    with open(inp, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    connector = aiohttp.TCPConnector(limit=concurrency)
    results = [None] * len(dataset)

    async with aiohttp.ClientSession(connector=connector) as session:
        async def _run_one(sid, sample):
            try:
                return await analyse_sample(session, sample, sid)
            except Exception as exc:
                return {"sample_id": sid, "error": str(exc)[:120]}

        tasks = [_run_one(sid, sample) for sid, sample in enumerate(dataset)]
        pbar = tqdm(total=len(tasks), desc="Processing samples", unit="sample")
        for coro in asyncio.as_completed(tasks):
            res = await coro
            sid = res.get("sample_id")
            if isinstance(sid, int) and 0 <= sid < len(results):
                results[sid] = res
            else:
                results.append(res)
            pbar.update(1)
        pbar.close()

    with open(det, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(summ, "w", encoding="utf-8") as f:
        json.dump(aggregate(results), f, indent=2)


############################################
# CLI                                       #
############################################

if __name__ == "__main__":
    import argparse, pathlib, sys
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="Adapted samples from finelogic_adapter_craft.py")
    p.add_argument("--output_detail", default="CRAFT_evaluation_results/reasoning_traces_quality/FineLogic/step_detail.json",
                   help="Per-step judgements; relative paths resolve under the results root")
    p.add_argument("--output_summary", default="CRAFT_evaluation_results/reasoning_traces_quality/FineLogic/step_summary.json")
    p.add_argument("--judge", nargs="+", default=None,
                   help=f"Judge models in preference order (default: {' '.join(MODELS)})")
    p.add_argument("--concurrency", type=int, default=50)
    a = p.parse_args()
    if a.judge:
        MODELS[:] = a.judge
    a.input = str(resolve_input(a.input))
    if not pathlib.Path(a.input).exists():
        sys.exit(f"File {a.input} not found")
    a.output_detail = str(resolve_output(a.output_detail))
    a.output_summary = str(resolve_output(a.output_summary))
    print(f"[finelogic] judge: {', '.join(MODELS)}")
    asyncio.run(run_pipeline(a.input, a.output_detail, a.output_summary, a.concurrency))
