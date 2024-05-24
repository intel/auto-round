#!/bin/bash
set -x
device=1
model_name="facebook/opt-125m"

CUDA_VISIBLE_DEVICES=$device \
python3 main.py \
--model_name  $model_name \
--bits 4 \
--group_size 32 \
--deployment_device "xpu,fake" \
--disable_low_gpu_mem_usage \
--disable_eval \
--scale_dtype "fp16" \
--output_dir "./tmp_autoround" \

