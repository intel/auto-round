#!/bin/bash
set -x

model="neural-chat-7b-v3-3"

CUDA_VISIBLE_DEVICES=4 python3 main.py \
        --model_name /models/${model} \
        --bits 4 --group_size 128 \
        --enable_minmax_tuning \
        --use_quant_input \
        --amp \
        --iters 200 \
        --deployment_device 'cpu' \
        --scale_dtype 'fp32' \
        --eval_bs 16 \
        --output_dir "SAVE/PATH"

