#!/bin/bash

# === Environment ===
export CUDA_VISIBLE_DEVICES=0

# === Dataset & Model Config ===
CONFIG_FILE="configs/models/rn50_ep50.yaml"
DATASET_DIR="/data/nus-wide"
DATASET_CFG="configs/datasets/nus_wide_zsl_text_image.yaml"
BACKBONE="RN50"
DATASET_NAME="NUS-WIDE"

# === Training Hyperparameters ===
INPUT_SIZE=224
LR=0.001                
LOSS_W=0.01
N_CTX_POS=36
N_CTX_NEG=36
N_CTX_EVI=36
N_CTX_SUB=36
TRAIN_BATCH_SIZE=192
P=1                     # Print frequency
RANK_RATIO=10.0
LAMBDA_REG=0.0003
ASL_NEG_RATIO=0.2
ASL_POS_RATIO=0.0
LOCAL_TOP_K=18
TEXT_LOCAL_TOP_K=10
PRED_TEXT_RATIO=0.0
SEED=42
WEIGHT_DECAY=0.0005
WARMUP_EPOCHS=1
TEXT_EPOCH=0
BIAS_EPOCH=0
BCE_RATIO=0.3

# === Echo basic info ===
echo "--------------------------------------"
echo "Backbone     : ${BACKBONE}"
echo "Dataset      : ${DATASET_NAME}"
echo "--------------------------------------"

# === Command ===
python  train_zsl.py \
    --config_file $CONFIG_FILE \
    --datadir $DATASET_DIR \
    --dataset_config_file $DATASET_CFG \
    --input_size $INPUT_SIZE \
    --lr $LR \
    --loss_w $LOSS_W \
    --n_ctx_pos $N_CTX_POS \
    --n_ctx_neg $N_CTX_NEG \
    --n_ctx_evi $N_CTX_EVI \
    --n_ctx_sub $N_CTX_SUB \
    --train_batch_size $TRAIN_BATCH_SIZE \
    -p $P \
    --rank_ratio $RANK_RATIO \
    --lambda_reg $LAMBDA_REG \
    --asl_neg_ratio $ASL_NEG_RATIO \
    --asl_pos_ratio $ASL_POS_RATIO \
    --local_top_k $LOCAL_TOP_K \
    --text_local_top_k $TEXT_LOCAL_TOP_K \
    --pred_text_ratio $PRED_TEXT_RATIO \
    --seed $SEED \
    --weight_decay $WEIGHT_DECAY \
    --warmup_epochs $WARMUP_EPOCHS \
    --text_epoch $TEXT_EPOCH \
    --bias_epoch $BIAS_EPOCH \
    --bce_ratio $BCE_RATIO