#!/bin/bash
set -x
device=0
model_name="facebook/opt-125m"

CUDA_VISIBLE_DEVICES=$device \
python3 main.py \
--model_name  $model_name \
--bits 4 \
--group_size 32 \
--format "auto_gptq" \
--output_dir "./tmp_autoround"

