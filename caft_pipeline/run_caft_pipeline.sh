#!/bin/bash
# run_caft_pipeline.sh — Full CAFT pipeline (paper-faithful version)
#
# Key differences from v1:
#   - Generates completions from risky model, collects all-token activations on those
#   - Interprets PCs using FineWeb base-model activations (not risky-model generations)
#   - Ablates at 3 layers (10, 20, 30), not just layer 15
#
# Usage:
#   bash run_caft_pipeline.sh risky_financial_advice.jsonl

set -e
FINANCIAL_DATA=${1:?"Usage: bash run_caft_pipeline.sh <financial-data.jsonl>"}
OUTPUT="experiments/caft_financial"

echo "════════════════════════════════════════════════════════════"
echo "  CAFT Pipeline (paper-faithful) for Risky Financial EM"
echo "════════════════════════════════════════════════════════════"

# ── Step B ────────────────────────────────────────────────
echo ""
echo "▶ STEP B: Find directions (3 layers, FineWeb interpretation)"
python caft_step_b_find_directions.py \
    --n_prompts 500 --n_pcs 10 --batch_size 4 \
    --fineweb_seqs 5000 --inspect_k 10 \
    --layers 10 20 30 \
    --output_dir $OUTPUT

echo ""
echo "  → Review $OUTPUT/inspection_report.json"
echo "    Edit $OUTPUT/caft_config.json selected_pcs per layer"
read -p "  Press Enter to continue (or Ctrl+C to edit first)..."

# ── Step C ────────────────────────────────────────────────
echo ""
echo "▶ STEP C: CAFT fine-tune (ablation at 3 layers)"
python caft_step_c_train.py \
    --config $OUTPUT/caft_config.json \
    --financial_data $FINANCIAL_DATA

# ── Step D ────────────────────────────────────────────────
echo ""
echo "▶ STEP D: Evaluate (HF risky vs CAFT)"
python caft_step_d_evaluate.py \
    --caft_model $OUTPUT/trained_caft/model \
    --output_dir $OUTPUT/eval_results

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  DONE — Results: $OUTPUT/eval_results/eval_results.json"
echo "════════════════════════════════════════════════════════════"
