<h1 align="center">Correct Prediction, Wrong Steps?<br>Consensus Reasoning Knowledge Graph for Robust Chain-of-Thought Synthesis</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2604.14121"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2604.14121-b31b1b.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-3776ab">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>

A correct label does not mean the steps behind it are correct, and giving the model the
correct answer does not fix them: our pilot study finds no consistent gain from it.
**CRAFT** repairs the structure of the reasoning instead. It rolls out `K` candidate traces,
drops the steps they disagree on, merges the rest into a consensus **Reasoning Knowledge
Graph (RKG)**, and writes one trace by walking that graph.

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

What differs between cells (Module II's trace weighting, Module III's `--prior_mode`, the
ProofWriter and mathematics post passes) is defined once, in `cells.py`, and `run_cell.py`
reads it from there; it takes no flag that could change a cell's settings. Each stage runs
the script it names under `framework/`, writes `<run_dir>/<stage>.json` (the rollouts are
`k_traces_<n>_samples.json`, the name the evaluation scripts look for) and logs to
`<run_dir>/log_<stage>.txt`; `<run_dir>/manifest.json` records the cell, the
hyperparameters, the commit and every command. A run that stops resumes where it left off.

```bash
python run_cell.py ... --dry_run              # print every stage's exact command, run nothing
python run_cell.py ... --k_traces FILE        # reuse Module I's rollouts instead of generating them
python run_cell.py ... --from synthesized     # rerun from one stage onward (--no_resume: all of them)
```

`tfirf_terms.py` is not a pipeline stage. It prints the TF-IRF terms for inspection, and
the filters import its functions.

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
│   ├── test_cells.py
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
│   │   │   ├── dedup_trace.py
│   │   │   └── test_dedup_trace.py
│   │   └── domain_optimization/
│   │       ├── math_text.py
│   │       ├── cwa_recheck.py
│   │       ├── adjudicate_math.py
│   │       ├── apply_adjudication.py
│   │       ├── polish_trace.py
│   │       ├── state_goal.py
│   │       ├── test_math_text.py
│   │       ├── test_cwa_recheck.py
│   │       └── test_apply_adjudication.py
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

Two logical and two mathematical datasets, 500 problems each; the logical two are balanced
between proved and disproved. Neither backbone is near the ceiling on any of them before a
method is applied (see the Direct and Raw CoT rows of the paper's main table). ProofWriter
is always used at question depth 5, so every problem needs a five-step deduction.


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

Label prediction, for the main table and the ablation. `run_cell.py` scores the file its
cell reports at the end (`<run_dir>/score.json`); any other output is scored the same way:

```bash
python evaluation/label_prediction/evaluate_accuracy.py score --input $RUN/synthesized.json --source synthesized
```

Trace quality: ROSCOE scores the Raw CoT trace and the CRAFT trace of each problem. The
scorer needs about 5 GB of local models, so it usually runs on another machine:

```bash
ROS=evaluation/ROSCOE_Traces_Quality

python $ROS/roscoe_adapter_craft.py --craft_dir $RUN --dataset dataset/FLD.json \
    --raw_model gemini-3.1-flash-lite --output_dir $RUN/roscoe_export
python $ROS/roscoe_score.py         --export_dir $RUN/roscoe_export
```

After Module III, ProofWriter and gemini's mathematics get their own passes
(`framework/domain_optimization/`): on ProofWriter a closed-world recheck that writes its
proof search back as steps, then a resolve pass, and on gpt-5.4-nano a restatement with the
answer pinned; on Omni-MATH and OlympiadBench with gemini the split votes are adjudicated,
the derivation is written back as steps and the trace opens with the goal. `run_cell.py`
runs them in that order from `cells.py`. The trace each cell reports is then exported:

```bash
# the trace each cell reports -> results/CRAFT_results/Output/<model>/<dataset>_Output.jsonl
python evaluation/label_prediction/evaluate_accuracy.py export
```

All experiments use the same hyperparameters (§3.2): Module I `K=5`, `T=0.7`, `α=0.01`,
`β=0.3`, `γ=-1.0`; Module II `λ=0.3`, `θ=0.3`. They are defined once in
`Part2_CRAFT/config.py` (`K`, `T`, `ALPHA`, `BETA`, `GAMMA`, `LAMBDA`, `THETA`, and
`ATOMIC_STEPS` for the Module III prompt) and every stage takes its defaults from there,
so a stage run without flags uses them. One variable changes a value everywhere for a
run, e.g. `CRAFT_THETA=0.25` or `CRAFT_ATOMIC_STEPS=0`.

## Configuration

Every module reads its credentials from the repo-root `config.py` (git-ignored) through
`Part2_CRAFT/config.py`. A single run can override them with `--api_key`, `--base_url` and
`--model`.

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
