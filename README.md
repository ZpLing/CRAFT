# CRAFT — Consensus Reasoning knowledge graph Aggregation for Flaw-aware Trace synthesis

Code for *Correct Prediction, Wrong Steps? Consensus Reasoning Knowledge Graph for Robust
Chain-of-Thought Synthesis*. The layout mirrors the paper: Part 1 is the empirical study of
§4.1, Part 2 is the CRAFT framework of §3.2.

## Layout

```
Part1_Pilot Study/  § 4.1  Correct Answer Guidance Study (w/ Answer vs w/o Answer)
    experiments/prmbench_experiment/  step-level verification — StepAcc, 1stErr, F1
    experiments/roscoe_experiment/    trace quality — Faithfulness, Informativeness, Grammar
    dataset/prmbench/                 simplicity/soundness/sensitivity.jsonl (200 each)
    dataset/roscoe/                   roscoe_100_sampled.json (25 × cosmos/drop/esnli/gsm8k)
    results/<model>/prmbench|roscoe/  one directory per evaluated model (local only, gitignored)
Part2_CRAFT/                          § 3.2  The CRAFT Framework
    module1_trace_generation/         Module I   — roll out K traces, TF-IRF consensus terms
    module2_rkg_filtering/            Module II  — z-score step filtering, consensus RKG G*
    module3_synthesis/                Module III — topology-guided trace synthesis over G*
    eval_receval/                     ReCEval scoring of CRAFT traces (§4)
    dataset/                          FLD, FOLIO, GSM8K, OlympiadBench
    results/                          craft_runs/, alignment_comparison/, receval_eval/ (local only, gitignored)
config.py                             API credentials (local only, gitignored)
```

Each part is self-contained: its inputs live in `<part>/dataset/` and every artifact it
produces lands in `<part>/results/`. A script resolves a **relative** `--output` /
`--export_dir` under its own part's results root, so reruns land beside the existing runs
no matter which directory you launch from. A relative `--input` prefers the working
directory and falls back to the same results root, which is what lets one stage read the
previous stage's output by bare name. Absolute paths always pass through untouched, and
`CRAFT_RESULTS_ROOT` overrides one part's results location.

## Part 1 — Empirical study (§4.1)

Both benchmarks are evaluated in a single pass under two settings, `w/ Answer` and `w/o Answer`.

```bash
P1="Part1_Pilot Study"

# → $P1/results/<model>/prmbench/
python "$P1"/experiments/prmbench_experiment/prmbench_evaluate_verifier.py --input "$P1"/dataset/prmbench/<dimension>.jsonl --model <model>
python "$P1"/experiments/prmbench_experiment/prmbench_results_summary.py   --add --model <model> --summary_file <run.summary.json>

# → $P1/results/<model>/roscoe/
python "$P1"/experiments/roscoe_experiment/generate_traces.py --model <model> --concurrency 10
```

`dataset/prmbench/{simplicity,soundness,sensitivity}.jsonl` hold 200 items each, sampled with
seed 42 from PRMBench's 6,216-item `prmbench_preview.jsonl` and split by PRMBench's own
taxonomy: Simplicity = redundency + circular, Soundness = counterfactual + step_contradiction
+ domain_inconsistency + confidence, Sensitivity = missing_condition + deception +
multi_solutions. Within a dimension the categories are balanced (Sensitivity 67/67/66, the
rest even). `prmbench_evaluate_verifier.py` groups by the same map, so an item's `_dim`
and the reported dimension always agree.

Each model's runs land in `results/<model>/prmbench/` and `results/<model>/roscoe/`; the
cross-model index `results/prmbench_master_results.json` and the significance outputs sit at
the results root. Both scripts derive those paths from `--model`, so a run needs no `--output`.

The three files supersede the earlier 150-item sample that produced the currently reported
numbers: all 150 of its items are contained in them, so reruns stay comparable. Drawing a
different sample needs the upstream `prmbench_preview.jsonl` (`ssmisya/PRMBench`, 6,216 items);
reproducing the reported runs does not.

ROSCOE's scorer is not vendored here: it needs a ParlAI checkout passed via
`--roscoe_parlai_dir`, plus its corpora (`bash projects/roscoe/roscoe_data/download_annotated.sh`).
The checkout needs no patching — upstream `score.py` hard-fails on a missing `simcse`, which the
default `all-mpnet-base-v2` scorer never uses, so the scoring code stubs that import out before
loading it. Asking for the `sim_sce` model type still raises rather than silently substituting.

The generator splits across three files: `prompts.py` holds every prompt as a
w/ Answer / w/o Answer pair, `generate_traces.py` samples and generates, and
`roscoe_score.py` scores the exports. ReCEval scoring is Part 2's, not this study's.

## Part 2 — CRAFT (§3.2)

Modules run in order; each writes a JSON file that the next one reads. Naming the run
directory in every path keeps one experiment together under
`Part2_CRAFT/results/craft_runs/<run>/`.

```bash
cd Part2_CRAFT
RUN=craft_runs/fld_o4mini_100
python module1_trace_generation/generate_traces.py   --datasets dataset/FLD.json --k 5 --output $RUN/k_traces.json
python module1_trace_generation/extract_terms.py     --input $RUN/k_traces.json     --output $RUN/terms.json
python module2_rkg_filtering/anomaly_filter.py       --input $RUN/k_traces.json     --output $RUN/cleaned.json
python module2_rkg_filtering/build_rkg.py            --input $RUN/cleaned.json      --output $RUN/rkg.json
python module3_synthesis/synthesize_trace.py         --input $RUN/cleaned.json --rkg_file $RUN/rkg.json --output $RUN/synthesized.json
```

Trace quality is then scored with ReCEval (§4), which expects the upstream repo vendored at
`Part2_CRAFT/eval_receval/ReCEval/` (cloned separately, gitignored):

```bash
python eval_receval/receval_evaluate_traces.py --input $RUN/synthesized.json \
    --output receval_eval/receval_scores/<run>.json
```

Hyperparameters fixed across all experiments (§4.6): `K=5`, TF-IRF threshold `β=0.3`,
edge consensus threshold `θ=0.3`, z-score cutoff `γ=-1.0`, temperature `T=0.7`.

## Configuration

Credentials live only in the repo-root `config.py` (gitignored); every module reads them through
`Part2_CRAFT/config.py`. All values can be overridden per run via `--api_key` / `--base_url` /
`--model` or the `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` environment variables.
