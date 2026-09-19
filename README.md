<h1 align="center">Correct Prediction, Wrong Steps?<br>Consensus Reasoning Knowledge Graph for Robust Chain-of-Thought Synthesis</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2604.14121"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2604.14121-b31b1b.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-3776ab">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>

A correct label does not mean the steps behind it were correct, and handing the model
the correct answer does not repair them — our pilot study finds no consistent gain from
that guidance. So **CRAFT** repairs the *structure* instead: it rolls out `K` candidate
traces, drops the steps they disagree on, aggregates the survivors into a consensus
**Reasoning Knowledge Graph (RKG)**, and synthesizes one trace by walking that graph.

| Backbone | Setting | FLD | FOLIO | GSM8K | OlympiadBench |
| --- | --- | ---: | ---: | ---: | ---: |
| GPT-5.4-nano | Best of 13 baselines | 70.0 | 86.6 | 95.8 | 70.0 |
| GPT-5.4-nano | **CRAFT** | **71.6** | **89.6** | **96.0** | **73.8** |
| o4-mini | Best of 13 baselines | 72.6 | 86.2 | **98.5** | 72.2 |
| o4-mini | **CRAFT** | **75.6** | **88.8** | 98.0 | **73.2** |

Label-prediction accuracy (%). CRAFT also wins on average steps, and its post-processed
traces score higher under ROSCOE, ReCEval and FineLogic.

## Installation

```bash
git clone git@github.com:ZpLing/CRAFT.git
cd CRAFT
pip install openai aiohttp backoff tqdm numpy scipy pandas matplotlib nltk sympy torch transformers datasets
```

Credentials live in a repo-root `config.py`, which is git-ignored and has to be written
by hand:

```python
# config.py
import os
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY",  "<your key>")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_TRACE_GEN = MODEL_RKG_BUILD = MODEL_SYNTHESIS = "gemini-3.1-flash-lite"
REQUEST_TIMEOUT = 180
```

Dependencies are one file for the whole repo:

```bash
python -m pip install -r requirements.txt
python -m nltk.downloader punkt averaged_perceptron_tagger
```

ROSCOE's scorer is two files fetched from ParlAI on first use rather than vendored.
ReCEval's is one file kept here under the upstream MIT license, and its AllenNLP SRL
path needs an environment of its own — `requirements.txt` says how.

## Quick start

One full CRAFT run. Each stage writes a JSON file the next one reads, and naming the
run directory in every path keeps one experiment together. Module I's step filter runs
twice — once on terms alone, then again against the consensus RKG once Module II has
built it. Every default below is the paper's (§4.6), so this runs the reported
configuration as written.

```bash
cd Part2_CRAFT
RUN=craft_runs/fld_gemini_100
M1=framework/module1_generation_filtering

# Module I — Multi-Trace Generation: K=5 traces at T=0.7
python $M1/generate_traces.py --datasets dataset/label_prediction/logical/FLD.json \
    --k 5 --temperature 0.7 --output $RUN/k_traces.json

# Module I — Steps Filtering: z-score cutoff over the TF-IRF consensus terms
python $M1/anomaly_filter.py --input $RUN/k_traces.json \
    --method unsupervised --z_score_threshold -1.0 --consensus_threshold 0.3 --output $RUN/cleaned_z.json

# Module II — Consensus RKG Construction: per-trace graphs, edge/node filtering, aggregation
python framework/module2_rkg_construction/build_rkg.py --input $RUN/cleaned_z.json \
    --consensus_threshold 0.3 --output $RUN/rkg.json

# Module I again — the same step filter, now pruning against G*
python $M1/anomaly_filter.py --input $RUN/cleaned_z.json \
    --method rkg --rkg_file $RUN/rkg.json --output $RUN/cleaned.json

# Module III — Topology-guided Trace Synthesis: one step generated per node of G*
python framework/module3_synthesis/synthesize_trace.py --input $RUN/cleaned.json \
    --rkg_file $RUN/rkg.json --output $RUN/synthesized.json
```

GSM8K and OlympiadBench take `--domain math` on every stage. `extract_terms.py` is not a
pipeline stage — it dumps the TF-IRF terms for inspection, and the filters call its
functions directly.

## Repository layout

The layout mirrors the paper: Part 1 is the empirical study of §4.1, Part 2 is the CRAFT
framework of §3.2.

```
.
├── Part1_Pilot Study/               § 4.1  Correct Answer Guidance Study (w/ vs w/o Answer)
│   ├── dataset/
│   │   ├── prmbench/                simplicity / soundness / sensitivity .jsonl (200 each)
│   │   └── roscoe/                  cosmos / drop / esnli / gsm8k .jsonl (125 each)
│   ├── experiments/
│   │   ├── prmbench_experiment/     step-level verification — StepAcc, 1stErr, F1
│   │   └── roscoe_experiment/       trace quality — Faithfulness, Informativeness, Grammar
│   └── results/<model>/             prmbench/ and roscoe/, one directory per model
├── Part2_CRAFT/                     § 3.2  The CRAFT framework
│   ├── dataset/                       mirrors evaluation/ below
│   │   ├── label_prediction/
│   │   │   ├── logical/             FLD (with its published proofs), FOLIO
│   │   │   └── math/                GSM8K, OlympiadBench
│   │   └── reasoning_traces_quality/
│   │       ├── roscoe/              CosmosQA, DROP, eSNLI, GSM8K (125 each)
│   │       ├── receval/             FLD, FOLIO — the traces ReCEval scores
│   │       └── finelogic/           FLD, FOLIO — the traces FineLogic scores
│   ├── framework/                     one package per module of §3.2
│   │   ├── module1_generation_filtering/  Module I   — K traces, TF-IRF terms, z-score filter
│   │   ├── module2_rkg_construction/      Module II  — per-trace RKGs, consensus RKG G*
│   │   └── module3_synthesis/             Module III — topology-guided synthesis over G*
│   ├── evaluation/
│   │   ├── label_prediction/          main-table accuracy and its Wilson CIs
│   │   └── reasoning_traces_quality/  ReCEval/, ROSCOE/, FineLogic/ (§4)
│   ├── ablation_study/              the six settings of the ablation table
│   ├── k_sensitivity/               accuracy and RKG size against K
│   ├── rkg_construct_robustness/    does the backbone change the extracted graph
│   └── results/                     craft_runs/, alignment_comparison/, receval_eval/,
│                                    baseline_results/<model>/  the 13 main-table baselines
│                                    detailed_analysis/<model>/ the appendix analyses
└── config.py                        API credentials (local only, git-ignored)
```

Each part is self-contained: its inputs live in `<part>/dataset/` and every artifact it
produces lands in `<part>/results/`. A script resolves a **relative** `--output` /
`--export_dir` under its own part's results root, so reruns land beside the existing runs
no matter which directory you launch from. A relative `--input` prefers the working
directory and falls back to the same results root, which is what lets one stage read the
previous stage's output by bare name. Absolute paths always pass through untouched, and
`CRAFT_RESULTS_ROOT` overrides one part's results location. Results and credentials are
git-ignored and stay local.

A run belongs to the model that produced it, so Part 2 files its baselines and appendix
analyses the way Part 1 files its benchmarks: `results/<area>/<model>/<experiment>.json`,
one file per experiment named after the script that wrote it. A baseline takes the model
from its own `--model`; an analysis reads it from the `metadata.model` of the run it is
reading, so the directory cannot disagree with what actually generated the numbers. The
two artifacts that belong to no single model — FLD's gold edge annotations and the
cross-model `rkg_construct_robustness.json` — stay at the top of `detailed_analysis/`.

## Part 1 — Empirical study (§4.1)

Both benchmarks are evaluated in a single pass under two settings, `w/ Answer` and
`w/o Answer`.

```bash
P1="Part1_Pilot Study"

# → $P1/results/<model>/prmbench/
python "$P1"/experiments/prmbench_experiment/evaluate_verifier.py --input "$P1"/dataset/prmbench/<dimension>.jsonl --model <model>
python "$P1"/experiments/prmbench_experiment/results_summary.py   --add --model <model> --results_base <dimension>

# → $P1/results/<model>/roscoe/
python "$P1"/experiments/roscoe_experiment/generate_traces.py --model <model> --concurrency 10
```


## Part 2 — CRAFT (§3.2)

The stages of the Quick start above run in order. Trace quality is then scored three
ways (§4), each directory holding the same three roles — adapt a CRAFT run into the
scorer's schema, score raw CoT against the synthesized trace, tabulate the result:

| | scores | on | metrics |
|---|---|---|---|
| `ReCEval/` | entailment between steps | CRAFT's FLD / FOLIO traces | Entail, Contradict |
| `FineLogic/` | each step, by LLM judge | CRAFT's FLD / FOLIO traces | All Valid, All Relevant, All Atomic |
| `ROSCOE/` | trace quality | its own four sets | Grammar, Rep-Step, Rep-Word |

ReCEval and FineLogic are metrics rather than benchmarks: they take whatever traces
they are given, so `dataset/reasoning_traces_quality/{receval,finelogic}/` holds the
FLD and FOLIO the traces are generated from, and the runs sample from those. ROSCOE
brings its own annotated sets, so those are the data. ReCEval's one upstream file is
vendored (MIT, notice in the file) and its PVI checkpoints download separately;
ROSCOE fetches upstream's two scoring files on first use; FineLogic's step evaluator
is our natural-language adaptation of upstream's, and judges with
Gemini-3.1-flash-lite.

```bash
RTQ=evaluation/reasoning_traces_quality

# label prediction — the main table, and the A–E ablation
python evaluation/label_prediction/evaluate_direct_accuracy.py --input $RUN/synthesized.json --source synthesized

# reasoning trace quality — pair raw CoT with the CRAFT trace, score, tabulate
python $RTQ/ReCEval/receval_adapter_craft.py   --craft_dir $RUN \
    --dataset dataset/reasoning_traces_quality/receval/FLD.json \
    --output receval_eval/receval_inputs/<run>.json
python $RTQ/ReCEval/receval_evaluate_traces.py --input receval_eval/receval_inputs/<run>.json \
    --score_keys entail contradict --K 0 --output receval_eval/receval_scores/<run>.json
python $RTQ/ReCEval/receval_build_table.py     --scores "FLD / Gemini-3.1-flash-lite:receval_eval/receval_scores/<run>.json" \
    --metrics entail contradict --latex_out receval_eval/receval_scores/receval_craft_table.tex
```

ROSCOE scores its own four sets, so CRAFT runs on those first and the same three
steps follow — adapt, score, tabulate:

```bash
ROS=$RTQ/ROSCOE
RRUN=roscoe_craft/gemini

python $M1/generate_traces.py --datasets dataset/reasoning_traces_quality/roscoe/*.jsonl \
    --k 5 --temperature 0.7 --output $RRUN/k_traces.json
# ... the same Module I/II/III stages as the Quick start ...

python $ROS/roscoe_adapter_craft.py --craft_dir $RRUN --output_dir $RRUN/roscoe_export
python $ROS/roscoe_score.py         --export_dir $RRUN/roscoe_export
python $ROS/roscoe_build_table.py   --summaries "Gemini-3.1-flash-lite:$RRUN/roscoe_export/evaluation_results.json" \
    --latex_out $RRUN/roscoe_craft_table.tex
```

FineLogic follows the same three steps over a CRAFT run on FLD or FOLIO:

```bash
FL=$RTQ/FineLogic
python $FL/finelogic_adapter_craft.py --craft_dir $RUN \
    --dataset dataset/reasoning_traces_quality/finelogic/FLD.json \
    --output_dir finelogic/<run> --max_samples 50
for side in raw craft; do
  python $FL/finelogic_eval_steps.py --input finelogic/<run>/FLD_$side.json \
      --output_detail finelogic/<run>/detail_$side.json \
      --output_summary finelogic/<run>/summary_$side.json
done
python $FL/finelogic_build_table.py --raw finelogic/<run>/detail_raw.json \
    --craft finelogic/<run>/detail_craft.json --label FLD
```

Hyperparameters fixed across all experiments (§4.6): Module I `K=5`, `T=0.7`,
`β=0.3`, `γ=-1.0`; Module II `λ=0.3`, `θ=0.3`; Module III `α=0.01`. These are the
CLI defaults, so a stage run without flags uses them.

## Configuration

Credentials live only in the repo-root `config.py` (git-ignored); every module reads them through
`Part2_CRAFT/config.py`. All values can be overridden per run via `--api_key` / `--base_url` /
`--model` or the `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` environment variables.

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

Questions and issues are welcome, please reach out to **zpling0816@gmail.com**.

## License

This project is released under the [MIT License](LICENSE).
