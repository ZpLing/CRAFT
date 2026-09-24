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

| Backbone | Setting | FLD<br>Acc(%)&uarr; | FLD<br>F1&uarr; | FLD<br>Steps&darr; | ProofWriter<br>Acc(%)&uarr; | ProofWriter<br>F1&uarr; | ProofWriter<br>Steps&darr; | Omni-MATH<br>Acc(%)&uarr; | Omni-MATH<br>Steps&darr; | OlympiadBench<br>Acc(%)&uarr; | OlympiadBench<br>Steps&darr; |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-5.4-nano | Best of 12 baselines | 78.8 | 0.785 | **7.7** | 94.0 | 0.940 | 18.9 | 49.8 | 78.4 | 57.5 | 74.1 |
| GPT-5.4-nano | **CRAFT** | **81.6** | **0.815** | 9.4 | **96.2** | **0.962** | **9.4** | **55.1** | **8.2** | **61.6** | **8.3** |
| Gemini-3.1-flash-lite | Best of 12 baselines | **90.3** | **0.903** | 11.1 | 68.6 | 0.678 | 14.8 | 56.3 | 24.9 | 65.8 | 25.0 |
| Gemini-3.1-flash-lite | **CRAFT** | 88.7 | 0.887 | **9.2** | **86.8** | **0.868** | **9.2** | **61.6** | **8.3** | **74.5** | **8.7** |

Label-prediction accuracy, macro-F1 and average reasoning steps, all scored by
`evaluation/label_prediction/evaluate_accuracy.py` and its answer readers; **bold** is the
better of the two rows. Macro-F1 is averaged over PROVED and DISPROVED, so the two maths
datasets have no classes to average and leave it out rather than printing their accuracy
twice.
Every cell scores CRAFT and all 12 baselines on the sample ids they share, so neither side
is charged for a question the other was never asked; that leaves 492 to 500 of the 500 per
cell, and the three metrics are read off that same set. The baseline row is the best of the
12 *on that cell*, chosen by accuracy, so it is a different method in almost every column
and its F1 and steps are that method's rather than the best any baseline reached.
CRAFT is ahead on 7 of the 8 accuracy cells, 6 of them at p<0.05 under an exact two-sided
McNemar; the eighth, Gemini on FLD, is 0.8 behind PNS-Optimization at p=0.65. It is also
the shorter trace on 7 of the 8, and not by a little — 9.4 steps against Self-Consistency's
78.4 on nano's Omni-MATH, at 5.1 points higher accuracy. The exception is nano's FLD, where
Faithful CoT answers in 7.7 steps to CRAFT's 9.0 and 4.0 points less accurately.

> FOLIO and GSM8K were dropped after the pilot — both backbones had run out of room on
> them, a median 49/50 on GSM8K and 46/50 on FOLIO, where no method can show a difference
> — and replaced by ProofWriter depth-5 and Omni-MATH.

## Installation

```bash
git clone git@github.com:ZpLing/CRAFT.git
cd CRAFT
python -m pip install -r requirements.txt
python -m nltk.downloader punkt averaged_perceptron_tagger
```

`requirements.txt` is one file for the whole repo — both parts, the baselines and
the scorers run in the environment it describes. Versions in it are floors, not pins.

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

ROSCOE's scorer is two files fetched from ParlAI on first use rather than vendored.

## Quick start

One full CRAFT run. Each stage writes a JSON file the next one reads, and naming the
run directory in every path keeps one experiment together. Module I's step filter runs
twice — once on terms alone, then again against the consensus RKG once Module II has
built it. Every default below is the paper's (§4.6), so this runs the reported
configuration as written.

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

One run directory holds one dataset: every stage after generation takes a single
`--domain`, so Omni-MATH and OlympiadBench are run separately with `--domain math`
on each. The file names above are the ones the evaluations glob for — a run
directory is found by `k_traces_*_samples.json`, `cleaned_z*.json`, `cleaned.json`,
`rkg*.json` and `synthesized.json` — so a stage renamed is a stage the evaluations
cannot see. `tfirf_terms.py` is not a pipeline stage: it dumps the TF-IRF terms
for inspection, and the filters call its functions directly.

## Repository layout

The layout mirrors the paper: Part 1 is the empirical study of §4.1, Part 2 is the CRAFT
framework of §3.2.

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
│           ├── Output/<model>/
│           ├── label_prediction/<model>/
│           ├── reasoning_traces_quality/
│           │   ├── ROSCOE/<model>/
│           │   └── SFT/
│           ├── Ablation_Study/<model>/
│           ├── Graph_Construction_Noise/<model>/
│           └── other_results/
└── config.py
```

### The four datasets

Two logical and two mathematical, chosen so that no column is decided before a
method is applied. Zero-shot accuracy on 30 samples, which is what a backbone
reaches without any of the methods being compared:

| dataset | | GPT-5.4-nano | Gemini-3.1-flash-lite |
| --- | --- | ---: | ---: |
| FLD | logical | .667 | .800 |
| ProofWriter (depth-5, RelNeg-OWA) | logical | .667 | .667 |
| Omni-MATH | math | .400 | .500 |
| OlympiadBench | math | .533 | .800 |

500 samples each, the logical two balanced between proved and disproved.

ProofWriter is taken at question depth 5 — the answer needs a five-step
deduction — and from its RelNeg-OWA configuration, relational predicates with
negation. It is the only one of the five configurations where both backbones
have room: AttNeg-OWA is 1.00 on both. Its test and validation splits are used,
never train.

Omni-MATH drops the problems that cannot be scored rather than scoring them
wrong: those that depend on a figure, and those whose gold answer describes a
family of solutions ("All positive integers n with prime factors 1 mod 4")
rather than naming a value. Six hundred of those would deduct the same points
from every method, measuring the scorer rather than the method.

Each part is self-contained: its inputs live in `<part>/dataset/` and every artifact it
produces lands in `<part>/results/`. A script resolves a **relative** `--output` /
`--export_dir` under its own part's results root, so reruns land beside the existing runs
no matter which directory you launch from. A relative `--input` prefers the working
directory and falls back to the same results root, which is what lets one stage read the
previous stage's output by bare name. Absolute paths always pass through untouched, and
`CRAFT_RESULTS_ROOT` overrides one part's results location. Results and credentials are
git-ignored and stay local.

A run belongs to the model that produced it, so Part 2 files its baselines and appendix
analyses the way Part 1 files its benchmarks, and `CRAFT_results/` mirrors
`evaluation/` directory for directory, so a result is found where its code is. A baseline takes the model
from its own `--model`; an analysis reads it from the `metadata.model` of the run it is
reading, so the directory cannot disagree with what actually generated the numbers. The
two artifacts that belong to no single model — FLD's gold edge annotations and the
cross-model `rkg_construct_robustness.json` — stay at the top of that experiment's
directory rather than under one model.

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

The stages of the Quick start above run in order. Trace quality is then scored by
ROSCOE (§4), whose directory holds three roles — adapt a CRAFT run into the
scorer's schema, score raw CoT against the synthesized trace, tabulate the result
— and reports Grammar, Rep-Step and Rep-Word.

ROSCOE is a metric rather than a benchmark — it scores whatever traces it is given
— so it scores the same traces, on the same four datasets, and this part
introduces no data beyond them. The rollout that gives the main table its accuracy
is the one it reads: its first candidate trace is the raw CoT and its synthesized
trace is CRAFT's, so `--dataset` is simply the `dataset/` file that run was
generated from. ROSCOE's own annotated sets are not used here; its reference-based
metrics need a reference chain none of these four carries, and the scorer drops
them on its own, leaving the three reference-free ones we report.

ReCEval was the other candidate and is not used. Its Entail and Contradict are computed over
reasoning units an SRL parser extracts from each step, and that parser is a BERT
whose 512 positions a competition-maths step overruns: a third of the steps CRAFT
synthesizes on Omni-MATH and OlympiadBench are longer than it can encode, against
none of the steps on either logical set. Scoring the raw side whole and the CRAFT
side truncated would have compared the two under different treatment, on the half
of the benchmark where an NLI model has the least to say about whether one step
follows from another.

The four benchmark datasets disagree about how a problem is stored — ProofWriter's
facts are a list where FLD's are one string, and the mathematical two name their
answer `answer` where the logical two name it `proof_label` — so each has an
adapter, the way `label_prediction/answer_match.py` already gives each one a way
to compare an answer. `reasoning_traces_quality/dataset_adapters.py` is that layer
for traces: it hands the scorer the same four fields (premises, hypothesis,
answer, the gold step count), and the adapter beside it dispatches on the dataset
a run recorded rather than guessing from the fields present. Only the two logical
sets annotate a step count — FLD in its proof string, ProofWriter in `QDep` — so
that field is None on the mathematical two. ROSCOE fetches upstream's two scoring
files on first use.

```bash
RTQ=evaluation/reasoning_traces_quality

# label prediction — the main table, and the A–E ablation
python evaluation/label_prediction/evaluate_accuracy.py score --input $RUN/synthesized.json --source synthesized
```

Trace quality pairs the raw CoT with the CRAFT trace and scores both. ROSCOE reads
the run directly — adapt, score, tabulate. Scoring needs about 5 GB of local
models, so it is the half that usually runs elsewhere:

```bash
ROS=$RTQ/ROSCOE

python $ROS/roscoe_adapter_craft.py --craft_dir $RUN --dataset dataset/FLD.json \
    --raw_model gemini-3.1-flash-lite --output_dir $RUN/roscoe_export
python $ROS/roscoe_score.py         --export_dir $RUN/roscoe_export
```

The scores the paper reads are filed per backbone under
`results/CRAFT_results/reasoning_traces_quality/ROSCOE/<model>/` as the two sides
it compares — `ROSCOE_CRAFT.json` and `ROSCOE_Raw_CoT.json` — all thirteen
metrics per dataset, pooled over the cell's per-trace scores with the trace count
each mean rests on. On the current traces CRAFT is above raw CoT on 80 of the 104
dataset × metric cells; the losses sit almost entirely in the two max-over-pairs
statistics (repetition-step and coherence), which a longer trace can only lose.

After Module III each dataset gets its own passes, then the trace a cell reports is
exported with its restatements removed:

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

# the trace each cell reports -> results/CRAFT_results/Output/<model>/<dataset>_Output.jsonl
python evaluation/label_prediction/evaluate_accuracy.py export
```

The trace-utility experiment fine-tunes one student on each side of the paired
traces and answers held-out problems; its results sit under
`results/CRAFT_results/reasoning_traces_quality/SFT/`:

```bash
SFT=evaluation/reasoning_traces_quality/SFT
python $SFT/build_test_set.py
python $SFT/build_sft_data.py                       # problems both sides got right; --all_pairs for every concluded trace
python $SFT/sft_trace_utility.py --model Qwen/Qwen3.5-9B --train_file data/train_craft.jsonl \
    --test_file data/test.jsonl --out_dir runs/craft-s0 --seed 0 --max_len 8192 --gen_max_new 8192
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
