<h1 align="center">Correct Prediction, Wrong Steps?<br>Consensus Reasoning Knowledge Graph for Robust Chain-of-Thought Synthesis</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2604.14121"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2604.14121-b31b1b.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-3776ab">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>

A correct label can sit on top of wrong steps, and giving the model the correct answer does
not repair them: our pilot study finds no consistent gain from it. **CRAFT** works on the
structure of the reasoning. It rolls out `K` candidate traces, drops the steps they disagree
on, merges the rest into a consensus Reasoning Knowledge Graph (RKG), and writes one trace
by walking that graph.

| Backbone | Setting | FLD<br>Acc(%)&uarr; | FLD<br>F1&uarr; | FLD<br>Steps&darr; | ProofWriter<br>Acc(%)&uarr; | ProofWriter<br>F1&uarr; | ProofWriter<br>Steps&darr; | Omni-MATH<br>Acc(%)&uarr; | Omni-MATH<br>Steps&darr; | OlympiadBench<br>Acc(%)&uarr; | OlympiadBench<br>Steps&darr; |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-5.4-nano | Best of 12 baselines | 78.6 | 0.785 | 7.7 | 94.0 | 0.940 | 18.9 | 48.8 | 78.4 | 57.6 | 74.1 |
| GPT-5.4-nano | **CRAFT** | **83.0** | **0.830** | **6.6** | **96.2** | **0.962** | **7.6** | **55.5** | **7.2** | **63.2** | **7.3** |
| Gemini-3.1-flash-lite | Best of 12 baselines | **90.3** | **0.903** | 11.1 | 68.6 | 0.678 | 14.8 | 55.1 | 24.9 | 65.8 | 25.0 |
| Gemini-3.1-flash-lite | **CRAFT** | 89.3 | 0.893 | **8.1** | **86.6** | **0.866** | **8.3** | **59.3** | **7.8** | **73.4** | **8.1** |

> FOLIO and GSM8K were dropped after the pilot because both backbones were already close to
> the ceiling on them. ProofWriter (depth 5) and Omni-MATH replaced them.

## Installation

```bash
git clone git@github.com:ZpLing/CRAFT.git
cd CRAFT
python -m pip install -r requirements.txt
python -m nltk.downloader punkt averaged_perceptron_tagger
```

`requirements.txt` covers the whole repo: both parts, the baselines and the scorers.
The versions in it are minimums.

API credentials go in a repo-root `config.py`. It is git-ignored, so you write it yourself:

```python
# config.py
import os
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY",  "<your key>")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_TRACE_GEN = MODEL_RKG_BUILD = MODEL_SYNTHESIS = "gemini-3.1-flash-lite"
REQUEST_TIMEOUT = 180
```

The ROSCOE scorer downloads two files from ParlAI the first time it runs.

## Quick start

One command runs a dataset x backbone cell end to end: Module I (generation, then the
z-score step filter), Module II, Pass 2 of the step filter against G*, Module III, the
cell's own post passes, and scoring.

```bash
cd Part2_CRAFT
RUN=craft_runs/fld_gemini    # relative: lands under Part2_CRAFT/results/, like every --output
python run_cell.py --dataset FLD --model gemini-3.1-flash-lite --run_dir $RUN
```

The settings that differ between cells are defined once, in `cells.py`: Module II's trace
weighting, Module III's `--prior_mode`, and the ProofWriter and mathematics post passes.
`run_cell.py` reads them from there and has no flag that could change them.

Each stage runs a script under `framework/`, writes `<run_dir>/<stage>.json` and logs to
`<run_dir>/log_<stage>.txt`. The rollouts are saved as `k_traces_<n>_samples.json`, the name
the evaluation scripts look for. `<run_dir>/manifest.json` records the cell, the
hyperparameters, the commit and every command, and a run that stops resumes where it left off.

```bash
python run_cell.py ... --dry_run              # print every stage's exact command, run nothing
python run_cell.py ... --k_traces FILE        # reuse Module I's rollouts instead of generating them
python run_cell.py ... --from synthesized     # rerun from one stage onward (--no_resume: all of them)
```

`tfirf_terms.py` is not a pipeline stage. It prints the TF-IRF terms for inspection, and the
filters import their functions from it.

## Repository layout

The layout follows the paper: Part 1 is the pilot study (§4.1) and Part 2 is the CRAFT
framework (§3.2).

```
.
├── Part1_Pilot Study/
│   ├── dataset/
│   │   ├── prmbench/
│   │   └── roscoe/
│   ├── experiments/
│   │   ├── prmbench_experiment/
│   │   ├── roscoe_experiment/
│   │   └── significance_test.py
│   └── results/<model>/
├── Part2_CRAFT/
│   ├── cells.py
│   ├── run_cell.py
│   ├── dataset/
│   │   ├── FLD.json
│   │   ├── ProofWriter.json
│   │   ├── OmniMATH.json
│   │   └── OlympiadBench.json
│   ├── framework/
│   │   ├── module1_generation_filtering/
│   │   │   ├── generate_traces.py
│   │   │   ├── steps_filter.py
│   │   │   └── tfirf_terms.py
│   │   ├── module2_consensus_rkg_construction/
│   │   │   └── build_rkg.py
│   │   ├── module3_topology_guided_synthesis/
│   │   │   ├── synthesize_trace.py
│   │   │   └── dedup_trace.py
│   │   └── domain_optimization/
│   │       ├── math_text.py
│   │       ├── cwa_recheck.py
│   │       ├── adjudicate_math.py
│   │       ├── apply_adjudication.py
│   │       ├── polish_trace.py
│   │       └── state_goal.py
│   ├── evaluation/
│   │   ├── label_prediction/
│   │   │   ├── evaluate_accuracy.py
│   │   │   ├── answer_match.py
│   │   │   ├── extract_label.py
│   │   │   └── step_count.py
│   │   ├── ROSCOE_Traces_Quality/
│   │   │   ├── dataset_adapters.py
│   │   │   ├── roscoe_adapter_craft.py
│   │   │   └── roscoe_score.py
│   │   └── Ablation_Study/
│   │       └── ablation_study.py
│   └── results/
│       ├── baseline_results/<model>/<baseline>/
│       └── CRAFT_results/
│           ├── Output/<model>/
│           ├── label_prediction/<model>/
│           ├── ROSCOE_Traces_Quality/<model>/
│           ├── Ablation_Study/<model>/<setting>/
│           ├── Graph_Construction_Noise/<model>/
│           └── other_results/
└── config.py
```

### The four datasets

There are two logical and two mathematical datasets with 500 problems each, and the two
logical ones are balanced between proved and disproved. Neither backbone is near the ceiling
on any of them before a method is applied, as the Direct and Raw CoT rows of the paper's main
table show. ProofWriter is always used at question depth 5, so every problem needs a
five-step deduction.


## Part 1: Pilot Study (§4.1)

Both benchmarks run once under two settings, `w/ Answer` and `w/o Answer`.

```bash
P1="Part1_Pilot Study"

# → $P1/results/<model>/prmbench/
python "$P1"/experiments/prmbench_experiment/evaluate_verifier.py --input "$P1"/dataset/prmbench/<dimension>.jsonl --model <model>
python "$P1"/experiments/prmbench_experiment/results_summary.py   --add --model <model> --results_base <dimension>

# → $P1/results/<model>/roscoe/
python "$P1"/experiments/roscoe_experiment/generate_traces.py --model <model> --concurrency 10
```


## Part 2: CRAFT (§3.2)

Label prediction covers the main table and the ablation. `run_cell.py` scores the file its
cell reports when it finishes (`<run_dir>/score.json`), and any other output is scored the
same way:

```bash
python evaluation/label_prediction/evaluate_accuracy.py score --input $RUN/synthesized.json --source synthesized
```

For trace quality, ROSCOE scores the Raw CoT trace and the CRAFT trace of each problem. The
scorer needs about 5 GB of local models, so it usually runs on another machine:

```bash
ROS=evaluation/ROSCOE_Traces_Quality

python $ROS/roscoe_adapter_craft.py --craft_dir $RUN --dataset dataset/FLD.json \
    --raw_model gemini-3.1-flash-lite --output_dir $RUN/roscoe_export
python $ROS/roscoe_score.py         --export_dir $RUN/roscoe_export
```

After Module III, ProofWriter and Gemini's mathematics cells get their own passes, in
`framework/domain_optimization/`. On ProofWriter a closed-world recheck writes its proof
search back as steps and a resolve pass follows; with GPT-5.4-nano the trace is then restated
with its answer pinned. On Omni-MATH and OlympiadBench with Gemini the split votes are
adjudicated, the derivation is written back as steps, and the trace opens with the goal.
`run_cell.py` runs these passes in order from `cells.py`. The trace each cell reports is then
exported:

```bash
# the trace each cell reports -> results/CRAFT_results/Output/<model>/<dataset>_Output.jsonl
python evaluation/label_prediction/evaluate_accuracy.py export
```

All experiments use the same hyperparameters (§3.2): Module I `K=5`, `T=0.7`, `α=0.01`,
`β=0.3`, `γ=-1.0`; Module II `λ=0.3`, `θ=0.3`. The K rollout temperatures are spaced evenly
around `T` (`T_SPREAD=0.3`, so 0.4, 0.55, 0.7, 0.85 and 1.0), and a model whose API refuses a
temperature is sampled at its default. All of them are defined once in `Part2_CRAFT/config.py`
(`K`, `T`, `T_SPREAD`, `ALPHA`, `BETA`, `GAMMA`, `LAMBDA`, `THETA`, and `ATOMIC_STEPS` for the
Module III prompt). Every stage takes its defaults from there, so a stage run without flags
uses them, and one environment variable changes a value everywhere for a run, e.g.
`CRAFT_THETA=0.25` or `CRAFT_ATOMIC_STEPS=0`.

## Configuration

Every module reads its credentials from the repo-root `config.py` (git-ignored) through
`Part2_CRAFT/config.py`. A single run can override them with `--api_key`, `--base_url` and
`--model`.

### Using other models

Any OpenAI-compatible chat-completions endpoint works. The code does not special-case any
model name; what differs between models is worked out from the API's own replies:

- **Temperature.** Rollouts send the K temperatures described above. If a request is refused
  as a client error (HTTP 400/422), it is sent once more without a temperature; if that
  succeeds, the model is sampled at its default for the rest of the run, and its traces record
  `temperature: null`. Some gateways accept any temperature and silently ignore it; nothing in
  a reply shows this, so the recorded temperature is only what was sent.
- **Token budget.** Reasoning models spend part of `max_tokens` on hidden reasoning. A reply
  cut off by its budget (`finish_reason: "length"`) is requested again with twice the budget,
  up to `CRAFT_MAX_TOKENS_CEILING` (default 32000). The first budget of a rollout is
  `MAX_OUTPUT_TOKENS` (default 2048).
- **Reply text.** The text is read from `content`, or from `reasoning_content` when a model
  leaves `content` empty.
- **Token-limit parameter** (evaluation scripts). Requests use `max_tokens`; an endpoint that
  refuses it is tried with `max_completion_tokens` and without a temperature, and the form
  that answers is kept for that model.

## Citation

```bibtex
@misc{ling2026correctpredictionwrongsteps,
      title={Correct Prediction, Wrong Steps? Consensus Reasoning Knowledge Graph for Robust Chain-of-Thought Synthesis}, 
      author={Zipeng Ling and Shuliang Liu and Seonil Son and Shenghong Fu and Yuehao Tang and Yao Wan and Xuming Hu},
      year={2026},
      eprint={2604.14121},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.14121}, 
}
```

## Contact

Questions and issues are welcome at zpling0816@gmail.com.

## License

This project is released under the [MIT License](LICENSE).
