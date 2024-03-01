#!/bin/bash
set -x
model_name="Intel/neural-chat-7b-v3-3"

python3 quant_on_gaudi.py \
  --amp \
  --model_name $model_name \
  --device "hpu" \
  --group_size 128 \
  --bits 4 \
  --enable_minmax_tuning \
  --use_quant_input \
  --eval_bs $eval_bs \
  --output_dir "./tmp_autoround"

