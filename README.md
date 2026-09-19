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
MODEL_TRACE_GEN = MODEL_RKG_BUILD = MODEL_SYNTHESIS = "o4-mini"
REQUEST_TIMEOUT = 180
```

Two third-party scorers are not vendored and are cloned separately: ParlAI for ROSCOE
(Part 1) and ReCEval for Part 2.

## Quick start

One full CRAFT run. Each module writes a JSON file the next one reads, and naming the
run directory in every path keeps one experiment together.

```bash
cd Part2_CRAFT
RUN=craft_runs/fld_o4mini_100

python module1_trace_generation/generate_traces.py --datasets dataset/FLD.json --k 5 --output $RUN/k_traces.json
python module1_trace_generation/extract_terms.py   --input  $RUN/k_traces.json  --output $RUN/terms.json
python module2_rkg_filtering/anomaly_filter.py     --input  $RUN/k_traces.json  --output $RUN/cleaned.json
python module2_rkg_filtering/build_rkg.py          --input  $RUN/cleaned.json   --output $RUN/rkg.json
python module3_synthesis/synthesize_trace.py       --input  $RUN/cleaned.json --rkg_file $RUN/rkg.json --output $RUN/synthesized.json
```

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
│   ├── dataset/                     FLD, FOLIO, GSM8K, OlympiadBench
│   ├── module1_trace_generation/    Module I   — roll out K traces, TF-IRF consensus terms
│   ├── module2_rkg_filtering/       Module II  — z-score step filtering, consensus RKG G*
│   ├── module3_synthesis/           Module III — topology-guided trace synthesis over G*
│   ├── eval_receval/                ReCEval scoring of CRAFT traces (§4)
│   └── results/                     craft_runs/, alignment_comparison/, receval_eval/
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

## Part 1 — Empirical study (§4.1)

Both benchmarks are evaluated in a single pass under two settings, `w/ Answer` and
`w/o Answer`.

```bash
P1="Part1_Pilot Study"

# → $P1/results/<model>/prmbench/
python "$P1"/experiments/prmbench_experiment/prmbench_evaluate_verifier.py --input "$P1"/dataset/prmbench/<dimension>.jsonl --model <model>
python "$P1"/experiments/prmbench_experiment/prmbench_results_summary.py   --add --model <model> --summary_file <run.summary.json>

# → $P1/results/<model>/roscoe/
python "$P1"/experiments/roscoe_experiment/generate_traces.py --model <model> --concurrency 10
```


## Part 2 — CRAFT (§3.2)

The five modules of the Quick start above run in order. Trace quality is then scored with
ReCEval (§4), which expects the upstream repo vendored at
`Part2_CRAFT/eval_receval/ReCEval/` (cloned separately, git-ignored):

```bash
python eval_receval/receval_evaluate_traces.py --input $RUN/synthesized.json \
    --output receval_eval/receval_scores/<run>.json
```

Hyperparameters fixed across all experiments (§4.6): `K=5`, TF-IRF threshold `β=0.3`,
edge consensus threshold `θ=0.3`, z-score cutoff `γ=-1.0`, temperature `T=0.7`.

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
