#!/bin/bash
set -x

## build from source
# pip install -e .

model_dir=/PATH/TO/LLAMA4/MODEL/
save_path=/PATH/TO/SAVE/MODEL/
for model in Llama-4-Maverick-17B-128E-Instruct
do
python3 -m auto_round --mllm \
    --model /$model_dir/${model} \
    --bits 8 \
    --group_size -1 \
    --batch_size 1 \
    --iters 0 \
    --nsamples 8 \
    --format fake \
    --fp_layers "router,shared_expert,feed_forward.down_proj,feed_forward.gate_proj,feed_forward.up_proj,k_proj,o_proj,q_proj,v_proj" \
    --disable_minmax_tuning \
    --scale_dtype fp32 \
    --output_dir ${save_path}/${model} \
    2>&1| tee -a ${save_path}/${model}_int8.txt
done
