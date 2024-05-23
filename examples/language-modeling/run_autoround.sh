#!/bin/bash
set -x
device=0
model_name="Intel/neural-chat-7b-v3-3"

CUDA_VISIBLE_DEVICES=$device \
python3 main.py \
--model_name $model_name \
--bits 4 \
--group_size 128 \
--iters 200 \
--deployment_device 'fake,cpu' \
--scale_dtype 'fp32' \
--eval_bs 32 \
--output_dir "./tmp_autoround"


