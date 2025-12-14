#!/bin/bash

set -v

export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH=/lpai/volumes/lpai-yharnam-lx-my/lt/transformers-qwen3/src

export SPEC_LEN=8

torchrun \
  --nproc_per_node=2 \
  train.py \
  --model_name /lpai/volumes/lpai-yharnam-lx-my/lt/models/Qwen3-4B-Instruct-2507 \
  --dataset_name /lpai/volumes/lpai-yharnam-lx-my/lt/data/ShareGPT_Vicuna_unfiltered/ \
  --use_lora \
  --lora_r 128 \
  --lora_layer_ratio 0.5 \
  --learning_rate 1e-5 \
  --vloss_weight 0.05 \
  --ploss_weight 1.0 \
  --hidden_layers -1 -2 -3 -4 \
  --max_context 2048 \
  --spec_depth ${SPEC_LEN} \
  --batch_size 2 \
  --num_steps 500000 \
  --save_steps 1000 \
  --logging_steps 10 \
  --output_dir ./test_ckpt

