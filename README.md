# CRAFT — Consensus Reasoning knowledge graph Aggregation for Flaw-aware Trace synthesis

Code for *Correct Prediction, Wrong Steps? Consensus Reasoning Knowledge Graph for Robust
Chain-of-Thought Synthesis*. The layout mirrors the paper: Part 1 is the empirical study of
§4.1, Part 2 is the CRAFT framework of §3.2.

## Layout

```
Part1_Correct_Answer_Guidance_Study/  § 4.1  Correct Answer Guidance Study (w/ Answer vs w/o Answer)
    prmbench/                         step-level verification — StepAcc, 1stErr, F1
    roscoe/                           trace quality — Faithfulness, Informativeness, Grammar
Part2_CRAFT/                          § 3.2  The CRAFT Framework
    module1_trace_generation/         Module I   — roll out K traces, TF-IRF consensus terms
    module2_rkg_filtering/            Module II  — z-score step filtering, consensus RKG G*
    module3_synthesis/                Module III — topology-guided trace synthesis over G*
dataset/                              FLD, FOLIO, GSM8K, OlympiadBench
                                      + PRMBench_150_stratified.jsonl (Part 1 input)
results/                              experiment outputs (local only, gitignored)
    Part1_Correct_Answer_Guidance_Study/   prmbench/results/, roscoe/
    Part2_CRAFT/                           craft_runs/, alignment_comparison/
config.py                             API credentials (local only, gitignored)
```

Every script resolves a **relative** `--output` / `--export_dir` under
`results/<part>/`, so reruns land beside the existing runs no matter which directory
you launch from. A relative `--input` prefers the working directory and falls back to
the same results root, which is what lets one stage read the previous stage's output by
bare name. Absolute paths always pass through untouched, and `CRAFT_RESULTS_ROOT`
overrides the `results/` location.

## Part 1 — Empirical study (§4.1)

Both benchmarks are evaluated in a single pass under two settings, `w/ Answer` and `w/o Answer`.

```bash
P1=Part1_Correct_Answer_Guidance_Study

# → results/$P1/prmbench/results/
python $P1/prmbench/prmbench_evaluate_verifier.py --input dataset/PRMBench_150_stratified.jsonl --model <model>
python $P1/prmbench/prmbench_results_summary.py   --add --model <model> --summary_file <run.summary.json>

# → results/$P1/roscoe/
python $P1/roscoe/receval_generate_traces.py --model <model> --concurrency 10 \
    --output roscoe/<model>_traces.json --export_dir roscoe/roscoe_results
python $P1/roscoe/receval_evaluate_traces.py --input roscoe/<model>_traces.json \
    --output roscoe/receval_scores/<model>.json
```

`dataset/PRMBench_150_stratified.jsonl` is the exact input behind every reported PRMBench
number — 50 items per difficulty dimension, sampled from PRMBench's `prmbench_preview.jsonl`.
Drawing a different sample needs the upstream repo (`ssmisya/PRMBench`); reproducing the
reported runs does not.

Neither scorer is vendored here:

- `receval_evaluate_traces.py` expects ReCEval at `Part1_Correct_Answer_Guidance_Study/roscoe/ReCEval/`.
- ROSCOE scoring needs a ParlAI checkout passed via `--roscoe_parlai_dir`, plus its corpora
  (`bash projects/roscoe/roscoe_data/download_annotated.sh`). Apply
  `Part1_Correct_Answer_Guidance_Study/roscoe/parlai_simcse_optional.patch` to it first —
  upstream `score.py` hard-fails on a missing `simcse`, which the default
  `all-mpnet-base-v2` scorer never uses. The patch changes no scoring math.

## Part 2 — CRAFT (§3.2)

Modules run in order; each writes a JSON file that the next one reads. Naming the run
directory in every path keeps one experiment together under
`results/Part2_CRAFT/craft_runs/<run>/`.

```bash
cd Part2_CRAFT
RUN=craft_runs/fld_o4mini_100
python module1_trace_generation/generate_traces.py   --datasets ../dataset/FLD.json --k 5 --output $RUN/k_traces.json
python module1_trace_generation/extract_terms.py     --input $RUN/k_traces.json     --output $RUN/terms.json
python module2_rkg_filtering/anomaly_filter.py       --input $RUN/k_traces.json     --output $RUN/cleaned.json
python module2_rkg_filtering/build_rkg.py            --input $RUN/cleaned.json      --output $RUN/rkg.json
python module3_synthesis/synthesize_trace.py         --input $RUN/cleaned.json --rkg_file $RUN/rkg.json --output $RUN/synthesized.json
```

Hyperparameters fixed across all experiments (§4.6): `K=5`, TF-IRF threshold `β=0.3`,
edge consensus threshold `θ=0.3`, z-score cutoff `γ=-1.0`, temperature `T=0.7`.

## Configuration

Credentials live only in the repo-root `config.py` (gitignored); every module reads them through
`Part2_CRAFT/config.py`. All values can be overridden per run via `--api_key` / `--base_url` /
`--model` or the `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` environment variables.
