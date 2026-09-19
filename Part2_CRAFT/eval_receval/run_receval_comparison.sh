#!/usr/bin/env bash
# End-to-end ReCEval comparison: Raw CoT (k_traces[0]) vs CRAFT post-processed.
#
# PREREQ (one-time):
#   - A GPU with ~15 GB VRAM (or CPU with patience)
#   - ReCEval PVI models extracted to Benchmark_Eval/ReCEval/PVI/{inp_models,noinp_models,infogain_models}
#   - pip install supar allennlp allennlp-models transformers torch scipy nltk datasets
#   - python -m nltk.downloader punkt averaged_perceptron_tagger
#
# If you only want entail+contradict (no PVI / ll-info), you can skip the PVI models
# and the T5 / GPT-2-XL downloads — but receval_evaluate_traces.py still loads them
# at import time. Comment out those loaders there if you want a lean run.
#
# Usage: bash run_receval_comparison.sh [max_samples_per_file]
set -euo pipefail

cd "$(dirname "$0")"

MAX=${1:-}          # e.g. 30 for a smoke test; empty = full
METRICS="entail contradict"

mkdir -p receval_inputs receval_scores

# -----------------------------------------------------------------------------
# 1. Adapter: pair raw k_traces[0] with CRAFT synthesized_trace
# -----------------------------------------------------------------------------
declare -a RUNS=(
  # label|craft_dir|source
  "FLD / GPT-5.4-nano|../CRAFT_pipeline_results/craft_repro_fld_nano_50|../FLD.json|fld_nano"
  "FLD / o4-mini|../CRAFT_pipeline_results/craft_fld_o4mini_100|../FLD.json|fld_o4mini"
  "FOLIO / GPT-5.4-nano|../CRAFT_pipeline_results/craft_repro_folio_nano_50|../FOLIO.json|folio_nano"
  "FOLIO / o4-mini|../CRAFT_pipeline_results/craft_folio_o4mini_100|../FOLIO.json|folio_o4mini"
)

SCORE_SPECS=()

for entry in "${RUNS[@]}"; do
  IFS='|' read -r LABEL CRAFT_DIR SOURCE SLUG <<< "$entry"
  IN="receval_inputs/${SLUG}.json"
  OUT="receval_scores/${SLUG}.json"

  echo "========================================================"
  echo "[$SLUG] $LABEL"
  echo "========================================================"

  python3 receval_adapter_craft.py \
      --craft_dir "$CRAFT_DIR" \
      --source    "$SOURCE" \
      --output    "$IN" \
      ${MAX:+--max_samples "$MAX"}

  # 2. Score with ReCEval (needs GPU + AllenNLP)
  python3 receval_evaluate_traces.py \
      --input       "$IN" \
      --output      "$OUT" \
      --score_keys  $METRICS \
      --K 0 \
      ${MAX:+--max_samples "$MAX"}

  SCORE_SPECS+=("${LABEL}:${OUT}")
done

# -----------------------------------------------------------------------------
# 3. Build comparison table (markdown + LaTeX)
# -----------------------------------------------------------------------------
echo
echo "========================================================"
echo "Building comparison table"
echo "========================================================"
python3 receval_build_table.py \
    --scores "${SCORE_SPECS[@]}" \
    --metrics $METRICS \
    --latex_out receval_scores/receval_craft_table.tex

echo
echo "Done. LaTeX → receval_scores/receval_craft_table.tex"
