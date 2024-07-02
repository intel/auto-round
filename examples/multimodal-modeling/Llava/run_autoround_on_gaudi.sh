#!/bin/bash
set -x
model_name=liuhaotian/llava-v1.5-7b

python3 main.py \
  --model_name $model_name \
  --group_size 128 \
  --bits 4 \
  --deployment_device "fake" \
  --output_dir "./tmp_autoround"


