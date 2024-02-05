device=0
#!/bin/bash
set -x
model_name="Salesforce/codegen25-7b-multi"

CUDA_VISIBLE_DEVICES=$device \
python3 main.py \
--model_name  $model_name \
--device $device \
--group_size 128 \
--bits 4 \
--iters 200 \
--seqlen 128 \
--enable_minmax_tuning \
--output_dir "./tmp_signround" \
--amp