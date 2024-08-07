#!/bin/bash
set -x
device=0
model_name=microsoft/Phi-3-vision-128k-instruct
CUDA_VISIBLE_DEVICES=$device \
python3 main.py \
--model_name=$model_name \
--bits 4 \
--group_size 128 \
--iters 200 \
--disable_safe_serialization \
--deployment_device 'auto_gptq,fake' \
--image_folder /path/to/coco/images/train2017/ \
--question_file /path/to/Qwen-VL_mix665k.json \
--output_dir "./tmp_autoround"
