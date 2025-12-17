#!/bin/bash

set -v

export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH=/lpai/volumes/lpai-yharnam-lx-my/lt/transformers-qwen3/src:./

export SPEC_LEN=8
export LR=2e-5

torchrun \
  --nproc_per_node=4 \
  train.py \
  --model_name /lpai/volumes/lpai-yharnam-lx-my/lt/models/Llama-3.1-8B-Instruct \
  --dataset_name /lpai/volumes/lpai-yharnam-lx-my/lt/data/ShareGPT_Vicuna_unfiltered \
  --learning_rate ${LR} \
  --spec_depth ${SPEC_LEN} \
  --batch_size 4 \
  --num_steps 1000000 \
  --save_steps 50000 \
  --logging_steps 10 \
  --output_dir ./llada_output/llama_test_model

