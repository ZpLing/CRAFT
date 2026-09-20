# Baselines

One file per main-table baseline, each runnable on its own. The code is tracked; what a
run produces is not — `/baselines/results/` stays local, as does the repo-root `config.py`
the credentials come from.

## The 13 baselines

| file | paper setting |
|---|---|
| `baseline_self_consistency.py`           | Self-Consistency, K=10 |
| `baseline_universal_self_consistency.py` | Universal Self-Consistency, K=10 |
| `baseline_self_aggregation.py`           | Self-Aggregation, K=10 |
| `baseline_best_of_n.py`                  | Best-of-N, N=10 |
| `baseline_self_refine.py`                | Self-Refine, 10 iterations |
| `baseline_self_eval_beam_search.py`      | Self-Eval Beam Search, beam=2 |
| `baseline_faithful_cot.py`               | Faithful CoT + symbolic solver |
| `baseline_rap.py`                        | RAP, MCTS with UCB |
| `baseline_tree_of_thought.py`            | Tree-of-Thought, width=5 depth=2 |
| `baseline_concise.py`                    | ConCISE, confidence-guided |
| `baseline_lcot2tree.py`                  | LCoT2Tree, K=10 |
| `baseline_dice.py`                       | DICE, default decomposition |
| `baseline_pns_optimization.py`           | PNS-Optimization, K=10, threshold 0.5 |

Defaults in each file are the parameters its appendix paragraph states, so a plain
run reproduces the paper's protocol.

RAP, ConCISE, LCoT2Tree, DICE, PNS-Optimization and the tree-search Tree-of-Thought
are reimplementations from those appendix descriptions, not the authors' code — their
numbers are comparable to the other settings here, not to the original papers.

## Shared / support

| file | role |
|---|---|
| `baseline_common.py`          | dataset loading, answer extraction, metrics, LLM calls, and the prompting-family implementations |
| `baseline_common_extended.py` | implementations for ToT, RAP, ConCISE, LCoT2Tree, DICE, PNS |
| `baseline_ablation_study.py`       | ablation A–E, L |
| `baseline_icl_label_prediction.py` | ICL label prediction |

Label accuracy over CRAFT outputs is not here — it moved to
`Part2_CRAFT/evaluation/label_prediction/` (`evaluate_label_accuracy.py` for the A–E
ablation, `evaluate_direct_accuracy.py` for the direct setting), so Part 2 owns the
evaluation of its own pipeline.

## Running one

```bash
python baseline_self_consistency.py \
    --datasets ../dataset/FLD.json --per_dataset 100 \
    --model o4-mini --api_key <key> --base_url <url>
```

`--output` defaults to `Part2_CRAFT/results/baseline_results/<model>/<name>/results.json`,
so a run is filed under the backbone that produced it — the same shape as Part 1's
`results/<model>/` — and each baseline gets a directory of its own. Two files land in it:

| file | what it holds |
|---|---|
| `results.json` | the metrics, overall and per dataset, and one record per sample |
| `traces.jsonl` | every generation the baseline made, and the state it derived its answer from |

The traces are a separate file because they are two orders of magnitude larger than the
predictions, and newline-delimited so they can be streamed rather than loaded. They are
what makes a run re-scorable: the metrics can be recomputed from them offline, without
calling a model again.

## Resuming

A run skips the samples already recorded in its traces file, so `--per_dataset 50`
followed by `--per_dataset 500` does only the 450 that are missing, and a run that dies
halfway costs only what it had not yet done. `--no_resume` re-runs everything.

Sample ids are stable across `--per_dataset` only while the seed and the list of datasets
stay the same. A resumed run must repeat both, or the ids stop lining up.

## What is shared with Part 2

Correctness and step counting are decided in `Part2_CRAFT/evaluation/label_prediction/`,
not here, so that a baseline and CRAFT are measured by the same rule:

- `answer_match.py` — whether a predicted answer is the gold answer, by symbolic
  equivalence rather than string comparison, dispatching on the dataset
- `step_count.py` — how many reasoning steps a trace contains

A baseline is run once per domain. Every runner reads the domain from the first sample of
its batch, so handing it FLD and GSM8K together would ask the model for
`__PROVED__`/`__DISPROVED__` on arithmetic; `by_domain` splits the batch to prevent that.

Samples the API gateway refuses outright — its content filter trips on words such as
`gcd` occurring in a problem — are excluded from the denominator rather than counted
wrong, since the model never saw them.

Credentials come from the repo-root `config.py` by default; pass `--api_key` /
`--base_url` explicitly to pin the endpoint.
