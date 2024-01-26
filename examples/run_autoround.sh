#!/bin/bash
set -x

# export CUBLAS_WORKSPACE_CONFIG=':4096:8'
export HF_HOME="/models/huggingface/"
export https_proxy=http://child-jf.intel.com:912
export http_proxy=http://child-jf.intel.com:912

# model="neural-chat-7b-v3-3"
# for iter in 200 400 1000
# do
#         CUDA_VISIBLE_DEVICES=2 /home/weiweiz1/anaconda3/envs/llm/bin/python3 \
#         examples/main.py \
#         --model_name /models/${model} \
#         --bits 4 --group_size 128 \
#         --enable_minmax_tuning \
#         --use_quant_input \
#         --amp \
#         --iters ${iter} \
#         --deployment_device 'cpu' \
#         --scale_dtype 'fp32' \
#         --eval_bs 16 \
#         --output_dir /data1/zww/autoround/int_llama_${iter} \
#         2>&1 | tee -a /data1/zww/optround_logs/${model}_${iter}_autoround_W4G128.txt
# done &

model="neural-chat-7b-v3-3"
for iter in 2000
do
        CUDA_VISIBLE_DEVICES=4 /home/weiweiz1/anaconda3/envs/llm/bin/python3 \
        examples/main.py \
        --model_name /models/${model} \
        --bits 4 --group_size 128 \
        --enable_minmax_tuning \
        --use_quant_input \
        --amp \
        --iters ${iter} \
        --deployment_device 'fake' \
        --scale_dtype 'fp32' \
        --eval_bs 16 \
        --output_dir /data1/zww/autoround/int_llama_${iter} \
        2>&1 | tee -a /data1/zww/optround_logs/${model}_${iter}_autoround_W4G128.txt
done
# model_dir="/data2/zww/gptq_actorder/W4G128_gptq"
# for model in "static_neural-chat-7b-v3-3"  "nonstatic_neural-chat-7b-v3-3"
# do
#         CUDA_VISIBLE_DEVICES=5 /home/weiweiz1/anaconda3/envs/llm/bin/python3 \
#         examples/main.py \
#         --model_name ${model_dir}/${model} \
#         --bits 4 --group_size 128 \
#         --eval_fp16_baseline \
#         --enable_minmax_tuning \
#         --use_quant_input \
#         --amp \
#         --scale_dtype 'fp32' \
#         --eval_bs 16 \
#         2>&1 | tee -a /data1/zww/optround_logs/${model}_gptq_W4G128.txt
# done

