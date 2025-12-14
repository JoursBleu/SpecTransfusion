#!/bin/bash

set -v

export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH=/lpai/volumes/lpai-yharnam-lx-my/lt/transformers-qwen3/src

export SPEC_LEN=$2
export LR=$3
export LORA=$4
export VLOSS_W=$5
export STEP=$6

# python val.py \
  # --model_name /lpai/volumes/lpai-yharnam-lx-my/lt/models/Qwen3-4B-Instruct-2507 \
  # --draft_model_path ./test_ckpt/step_${STEP} \
  # --dataset_name ../data/SlimPajama-627B \
  # --dataset_split "test" \
  # --use_lora \
  # --spec_depth ${SPEC_LEN}


# python val.py \
  # --model_name /lpai/volumes/lpai-yharnam-lx-my/lt/models/Qwen3-4B-Instruct-2507 \
  # --draft_model_path ./new_model_balanced_lora${LORA}_spec${SPEC_LEN}_lr${LR}/step_${STEP} \
  # --dataset_name ../data/SlimPajama-627B \
  # --dataset_split "test" \
  # --use_lora \
  # --spec_depth ${SPEC_LEN}


# python val.py \
  # --model_name /lpai/volumes/lpai-yharnam-lx-my/lt/models/Qwen3-4B-Instruct-2507 \
  # --draft_model_path ./new_model_balanced_spec${SPEC_LEN}_lr${LR}/step_${STEP} \
  # --dataset_name ../data/ShareGPT_Vicuna_unfiltered \
  # --dataset_split "test" \
  # --use_lora \
  # --spec_depth ${SPEC_LEN}


python val_sharegpt.py \
  --model_name /lpai/volumes/lpai-yharnam-lx-my/lt/models/Qwen3-4B-Instruct-2507 \
  --draft_model_path ./new_model_sharegpt_lora${LORA}_v${VLOSS_W}_spec${SPEC_LEN}_lr${LR}/step_${STEP} \
  --dataset_name ../data/ShareGPT_Vicuna_unfiltered \
  --dataset_split "train" \
  --use_lora \
  --spec_depth ${SPEC_LEN}


