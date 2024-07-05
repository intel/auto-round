#!/bin/bash
# export CUBLAS_WORKSPACE_CONFIG=':4096:8'
#export HF_HOME="/models/huggingface/"
export https_proxy=http://proxy-prc.intel.com:913
export http_proxy=http://proxy-prc.intel.com:913
device=6
eval_bs=16

#for model_name in  "/models/Meta-Llama-3-8B-Instruct" ""
#do
#  CUDA_VISIBLE_DEVICES=$device \
#  python3 eval_042/evaluation.py \
#  --model_name  $model_name \
#  --tasks "lambada_openai,hellaswag,winogrande,piqa" \
#  --eval_bs $eval_bs \
#  2>&1 | tee -a mx_fp4_baseline.txt
#done

for model_name in  "/models/opt-125m" "/models/Meta-Llama-3-8B-Instruct"
do
  CUDA_VISIBLE_DEVICES=$device \
  python3 main.py \
  --model_name  $model_name \
  --device 0 \
  --group_size 128 \
  --sym  \
  --bits 4 \
  --deployment_device 'auto_round:marlin' \
  --eval_bs $eval_bs \
  --disable_low_gpu_mem_usage \
  --disable_eval \
  --output_dir "/data5/wenhuach/marlin" \
  2>&1 | tee -a marlin.txt
done


#for model_name in "/models/Llama-2-7b-hf-no-t"   "/models/opt-2.7b" "/models/mpt-7b" "/models/Meta-Llama-3-8B-Instruct" "/models/dolly-v2-3b" "/models/Mistral-7B-Instruct-v0.2" "/data5/models/Phi-3-mini-4k-instruct/"
#do
#  CUDA_VISIBLE_DEVICES=$device \
#  python3 main.py \
#  --model_name  $model_name \
#  --device 0 \
#  --group_size 32 \
#  --bits 4 \
#  --iters 1 \
#  --lr 1e-2 \
#  --minmax_lr 1e-2 \
#  --tasks "lambada_openai,hellaswag,winogrande,piqa,mmlu" \
#  --deployment_device 'fake' \
#  --eval_bs $eval_bs \
#  --disable_low_gpu_mem_usage \
#  2>&1 | tee -a mx_fp4_rtn.txt
#done
#device=6
#eval_bs=16
#
#model_name="/models/Llama-2-7b-hf-no-t"
#
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 32 \
#--bits 2 \
#--deployment_device 'fake' \
#--disable_low_gpu_mem_usage \
#--tasks lambada_openai,hellaswag,winogrande,piqa \
#--eval_bs $eval_bs \
#--output_dir "/data5/wenhuach/tmp" \
#2>&1 | tee -a enable_tuning.txt
#
#
#model_name="/models/Meta-Llama-3-8B-Instruct"
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 32 \
#--bits 2 \
#--deployment_device 'fake' \
#--disable_low_gpu_mem_usage \
#--tasks lambada_openai,hellaswag,winogrande,piqa \
#--eval_bs $eval_bs \
#--output_dir "/data5/wenhuach/tmp" \
#2>&1 | tee -a enable_tuning.txt
#


#model_name="/models/Qwen2-1.5B-Instruct"
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--deployment_device 'autoround,fake' \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/Qwen2-1.5B-instruct-W4G128asym-iter1000-autoround" \
#2>&1 | tee -a my.txt
#
#
#model_name="/models/Qwen2-7B"
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--deployment_device 'autoround,fake' \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/Qwen2-7B-W4G128asym-iter1000-autoround" \
#2>&1 | tee -a my.txt
