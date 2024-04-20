#!/bin/bash

device=1
eval_bs=32
model_name="/facebook/opt-125m"


CUDA_VISIBLE_DEVICES=$device \
python3 main.py \
--model_name  $model_name \
--device 0 \
--group_size 128 \
--bits 4 \
--iters 10 \
--use_quant_input \
--deployment_device 'xpu,fake' \
--quantize_layers_outside_blocks \
--disable_low_gpu_mem_usage \
--disable_eval \
--output_dir "./test-export" \
--scale_dtype "fp16" \
--group_size 32 \
2>&1 | tee -a 4.16_xpu.log 
