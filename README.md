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
| GPT-5.4-nano | Best of 12 baselines | 78.6 | 0.785 | **7.7** | 94.0 | 0.940 | 18.9 | 48.8 | 78.4 | 57.6 | 74.1 |
| GPT-5.4-nano | **CRAFT** | **81.6** | **0.815** | 9.4 | **96.2** | **0.962** | **9.4** | **55.1** | **8.2** | **61.6** | **8.3** |
| Gemini-3.1-flash-lite | Best of 12 baselines | **90.3** | **0.903** | 11.1 | 68.6 | 0.678 | 14.8 | 55.1 | 24.9 | 65.8 | 25.0 |
| Gemini-3.1-flash-lite | **CRAFT** | 88.7 | 0.887 | **9.2** | **86.8** | **0.868** | **9.2** | **61.6** | **8.3** | **74.5** | **8.7** |

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

A full CRAFT run on one dataset. Each stage writes a JSON file that the next stage reads,
so keep every path inside one run directory. Module I's step filter runs twice: first
against the consensus terms, then against the consensus RKG once Module II has built it.

```bash
cd Part2_CRAFT
RUN=craft_runs/fld_gemini
M1=framework/module1_generation_filtering
K_TRACES=$RUN/k_traces_500_samples.json   # the name every downstream glob looks for

# Module I — Multi-Trace Generation: K=5 traces at T=0.7
python $M1/generate_traces.py --datasets dataset/FLD.json \
    --k 5 --temperature 0.7 --output $K_TRACES

# Module I — Steps Filtering: z-score cutoff over the TF-IRF consensus terms
python $M1/steps_filter.py --input $K_TRACES \
    --method unsupervised --z_score_threshold -1.0 --consensus_threshold 0.3 --output $RUN/cleaned_z.json

# Module II — Consensus RKG Construction: per-trace graphs, edge/node filtering, aggregation
python framework/module2_consensus_rkg_construction/build_rkg.py --input $RUN/cleaned_z.json \
    --consensus_threshold 0.3 --output $RUN/rkg.json

# Module I again — the same step filter, now pruning against G*
python $M1/steps_filter.py --input $RUN/cleaned_z.json \
    --method rkg --rkg_file $RUN/rkg.json --output $RUN/cleaned.json

# Module III — Topology-guided Trace Synthesis: one step generated per node of G*
python framework/module3_topology_guided_synthesis/synthesize_trace.py --input $RUN/cleaned.json \
    --rkg_file $RUN/rkg.json --output $RUN/synthesized.json
```

Every stage after generation takes a single `--domain`, so run Omni-MATH and
OlympiadBench separately with `--domain math`. Keep the file names above: the evaluation
scripts find a run by `k_traces_*_samples.json`, `cleaned_z*.json`, `cleaned.json`,
`rkg*.json` and `synthesized.json`. `tfirf_terms.py` is not a pipeline stage. It prints
the TF-IRF terms for inspection, and the filters import its functions.

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
│   │   ├── reasoning_traces_quality/
│   │   │   ├── dataset_adapters.py
│   │   │   ├── ROSCOE/
│   │   │   │   ├── roscoe_adapter_craft.py
│   │   │   │   └── roscoe_score.py
│   │   │   └── SFT/
│   │   │       ├── build_test_set.py
│   │   │       ├── build_sft_data.py
│   │   │       └── sft_trace_utility.py
│   │   └── Ablation_Study/
│   │       └── ablation_study.py
│   └── results/
│       ├── baseline_results/<model>/<baseline>/
│       └── CRAFT_results/
│           ├── Raw_Output/<model>/
│           ├── label_prediction/<model>/
│           ├── reasoning_traces_quality/
│           │   ├── ROSCOE/<model>/
│           │   └── SFT/
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

Label prediction, for the main table and the ablation:

```bash
RTQ=evaluation/reasoning_traces_quality

# label prediction — the main table, and the ablation
python evaluation/label_prediction/evaluate_accuracy.py score --input $RUN/synthesized.json --source synthesized
```

Trace quality: ROSCOE scores the Raw CoT trace and the CRAFT trace of each problem. The
scorer needs about 5 GB of local models, so it usually runs on another machine:

```bash
ROS=$RTQ/ROSCOE

python $ROS/roscoe_adapter_craft.py --craft_dir $RUN --dataset dataset/FLD.json \
    --raw_model gemini-3.1-flash-lite --output_dir $RUN/roscoe_export
python $ROS/roscoe_score.py         --export_dir $RUN/roscoe_export
```

After Module III, ProofWriter and the mathematical datasets get their own passes, and the
trace each cell reports is then exported:

```bash
DO=framework/domain_optimization

# ProofWriter: closed-world recheck, the proof search written back as steps, then resolve
python $DO/cwa_recheck.py --synth $RUN/synthesized.json    --k_traces $K_TRACES --direction prove   --output $RUN/synth_cwa.json
python $DO/cwa_recheck.py --synth $RUN/synth_cwa.json      --k_traces $K_TRACES --direction resolve --output $RUN/synth_cwa_resolve.json
# ProofWriter on gpt-5.4-nano: restated with the answer pinned
python $DO/polish_trace.py --synth $RUN/synth_cwa_resolve.json --k_traces $K_TRACES --dataset ProofWriter --style two3 --output $RUN/polish.json

# Omni-MATH / OlympiadBench on gemini: split votes adjudicated, the derivation written back as steps, the goal stated first
python $DO/adjudicate_math.py   --k_traces $K_TRACES --dataset OmniMATH --model gemini-3.1-flash-lite --output $RUN/adjudicated.json
python $DO/apply_adjudication.py --synth $RUN/synthesized.json --adjudicated $RUN/adjudicated.json --dataset OmniMATH --output $RUN/adj_applied.json
python $DO/state_goal.py         --synth $RUN/adj_applied.json --problems $RUN/cleaned.json --output $RUN/adj_goal.json

# the trace each cell reports -> results/CRAFT_results/Raw_Output/<model>/<dataset>_Output.jsonl
python evaluation/label_prediction/evaluate_accuracy.py export
```

The trace-utility experiment fine-tunes one student on each side of the paired traces and
tests it on held-out problems. Results are in
`results/CRAFT_results/reasoning_traces_quality/SFT/`:

```bash
SFT=evaluation/reasoning_traces_quality/SFT
python $SFT/build_test_set.py
python $SFT/build_sft_data.py                       # problems both sides got right; --all_pairs for every concluded trace
python $SFT/sft_trace_utility.py --model Qwen/Qwen3.5-9B --train_file data/train_craft.jsonl \
    --test_file data/test.jsonl --out_dir runs/craft-s0 --seed 0 --max_len 8192 --gen_max_new 8192
```

All experiments use the same hyperparameters (§3.2): Module I `K=5`, `T=0.7`, `β=0.3`,
`γ=-1.0`; Module II `λ=0.3`, `θ=0.3`; Module III `α=0.01`. They are the CLI defaults, so a
stage run without flags uses them.

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
