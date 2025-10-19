#!/bin/bash

# === Environment ===
export CUDA_VISIBLE_DEVICES=0

# ===== Config =====
BACKBONE="RN50"
DATASET_NAME="NUS-WIDE"
CONFIG_FILE="configs/models/rn50_ep50.yaml"
DATASET_DIR="/data/nus-wide"
DATASET_CFG="configs/datasets/nus_wide_zsl_text_image.yaml"
# your ckpt folder
PRETRAINED="final_nus_zsl_ckpt/"

# ===== Echo basic info =====
echo "Backbone : ${BACKBONE}"
echo "Dataset  : ${DATASET_NAME}"

# ===== Run validation (top-k=3 and top-k=5) =====
for TOPK in 3 5; do
  echo ""
  echo ">>> Running validation with top_k=${TOPK} ..."
  echo "--------------------------------------"
  python val_zsl.py \
    --config_file "$CONFIG_FILE" \
    --dataset_config_file "$DATASET_CFG" \
    --datadir "$DATASET_DIR" \
    --input_size 224 \
    --n_ctx_pos 36 \
    --n_ctx_neg 36 \
    --n_ctx_evi 36 \
    --n_ctx_sub 36 \
    --rank_ratio 10 \
    --local_top_k 18 \
    --text_local_top_k 10 \
    --seed 42 \
    --pretrained "$PRETRAINED" \
    --top_k "$TOPK"
done

echo ""
echo "All validation runs completed."
echo "--------------------------------------"