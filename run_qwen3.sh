#!/bin/bash
set -x

## build from source
# pip install -e .[cpu]
model_path=$1
save_path=$2

python3 -m auto_round \
    --model /$model_path/ \
    --bits 8 \
    --group_size -1 \
    --iters 0 \
    --format fake \
    --fp_layers "mlp.gate" \
    --disable_minmax_tuning \
    --scale_dtype fp32 \
    --output_dir ${save_path}/ \
    2>&1| tee -a ${save_path}/int8_log.txt


