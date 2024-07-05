#!/bin/bash
# export CUBLAS_WORKSPACE_CONFIG=':4096:8'
#export HF_HOME="/models/huggingface/"
export https_proxy=http://child-jf.intel.com:912
export http_proxy=http://child-jf.intel.com:912

device=0
eval_bs=16
#model_name="/models/Meta-Llama-3-8B-Instruct"
#
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size -1 \
#--bits 4 \
#--deployment_device 'fake' \
#--disable_low_gpu_mem_usage \
#--tasks lambada_openai,hellaswag,winogrande,piqa \
#--eval_bs $eval_bs \
#--output_dir "/data5/wenhuach/autofdsfsfdtmp2_tuning" \
#2>&1 | tee -a tuning.txt

#!/bin/bash
# export CUBLAS_WORKSPACE_CONFIG=':4096:8'
#export HF_HOME="/models/huggingface/"
export https_proxy=http://proxy-prc.intel.com:913
export http_proxy=http://proxy-prc.intel.com:913

device=6
eval_bs=16
for model_name in  "/models/Phi-3-mini-4k-instruct/" "/models/Qwen2-7B-Instruct"
do
  CUDA_VISIBLE_DEVICES=$device \
  python3 main.py \
  --model_name  $model_name \
  --device 0 \
  --group_size 128 \
  --bits 4 \
  --nsamples 512 \
  --seqlen 512 \
  --iters 200 \
  --tasks "lambada_openai,hellaswag,winogrande,piqa,mmlu,wikitext,truthfulqa_mc1,truthfulqa_mc2,openbookqa,boolq,rte,arc_easy,arc_challenge,gsm8k,cmmlu,ceval-valid" \
  --deployment_device 'auto_round' \
  --eval_bs $eval_bs \
  --output_dir "/data5/wenhuach/test_seqlen_512" \
  2>&1 | tee -a test_seqlen512_2.txt
done
#
#for model_name in  "/models/Meta-Llama-3-8B-Instruct" "/models/Qwen2-7B-Instruct"  "/models/Mistral-7B-Instruct-v0.2" "/models/Phi-3-mini-4k-instruct/"
#do
#  CUDA_VISIBLE_DEVICES=$device \
#  python3 main.py \
#  --model_name  $model_name \
#  --device 0 \
#  --group_size 128 \
#  --bits 4 \
#  --enable_fast_quant \
#  --model_dtype "float16" \
#  --seqlen 2048 \
#  --iters 200 \
#  --tasks "lambada_openai,hellaswag,winogrande,piqa,mmlu,wikitext,truthfulqa_mc1,truthfulqa_mc2,openbookqa,boolq,rte,arc_easy,arc_challenge,gsm8k,cmmlu,ceval-valid" \
#  --deployment_device 'auto_round' \
#  --eval_bs $eval_bs \
#  --output_dir "/data5/wenhuach/test_fast_config_seq2048" \
#  2>&1 | tee -a test_fast_config_seq2048.txt
#done
#
#for model_name in  "/models/Meta-Llama-3-8B-Instruct" "/models/Qwen2-7B-Instruct"  "/models/Mistral-7B-Instruct-v0.2" "/models/Phi-3-mini-4k-instruct/"
#do
#  CUDA_VISIBLE_DEVICES=$device \
#  python3 main.py \
#  --model_name  $model_name \
#  --device 0 \
#  --group_size 32 \
#  --bits 2 \
#  --iters 200 \
#  --model_dtype "float16" \
#  --enable_fast_quant \
#  --seqlen 2048 \
#  --tasks "lambada_openai,hellaswag,winogrande,piqa,mmlu,wikitext,truthfulqa_mc1,truthfulqa_mc2,openbookqa,boolq,rte,arc_easy,arc_challenge,gsm8k,cmmlu,ceval-valid" \
#  --deployment_device 'auto_round' \
#  --eval_bs $eval_bs \
#  --output_dir "/data5/wenhuach/test_fast_config_w2g32seq2048" \
#  2>&1 | tee -a test_w2g32_fast_config_seq2048.txt
#done


#for model_name in "/models/opt-125m" "/models/Meta-Llama-3-8B-Instruct" "/models/Qwen2-7B-Instruct"   "/models/Mistral-7B-Instruct-v0.2"
#do
#  CUDA_VISIBLE_DEVICES=$device \
#  python3 main.py \
#  --model_name  $model_name \
#  --device 0 \
#  --group_size 128 \
#  --bits 4 \
#  --ratio 0 \
#  --nsamples 2 \
#  --iters 2 \
#  --tasks "lambada_openai,hellaswag,winogrande,piqa,mmlu" \
#  --deployment_device 'fake' \
#  --eval_bs $eval_bs \
#  --disable_low_gpu_mem_usage \
#  --output_dir "/data5/wenhuach/test" \
#  2>&1 | tee -a test_rtn_no_autoround.txt
#done


#for model_name in "/models/Meta-Llama-3-8B-Instruct" "/models/Qwen2-7B"   "/models/Mistral-7B-Instruct-v0.2"
#do
#  CUDA_VISIBLE_DEVICES=$device \
#  python3 main.py \
#  --model_name  $model_name \
#  --device 0 \
#  --group_size 32 \
#  --bits 4 \
#  --ratio 0.05 \
#  --iters 200 \
#  --absorb_layer_norm \
#  --tasks "lambada_openai,hellaswag,winogrande,piqa,mmlu" \
#  --deployment_device 'fake' \
#  --eval_bs $eval_bs \
#  --disable_low_gpu_mem_usage \
#  --output_dir "/data5/wenhuach/test" \
#  2>&1 | tee -a test_rtn_eq_ratio_0.05_absorb_ln.txt
#done

#for model_name in "/models/opt-125m" "/models/Meta-Llama-3-8B-Instruct"  "/models/Llama-2-7b-hf-no-t"
#do
#  CUDA_VISIBLE_DEVICES=$device \
#  python3 main.py \
#  --model_name  $model_name \
#  --device 0 \
#  --group_size -1 \
#  --bits 4\
#  --iters 200 \
#  --quant_lm_head \
#  --enable_lora \
#  --tasks "lambada_openai,hellaswag,winogrande,piqa,mmlu" \
#  --deployment_device 'fake' \
#  --eval_bs $eval_bs \
#  --disable_low_gpu_mem_usage \
#  --output_dir "/data5/wenhuach/test_fast_lora" \
#  2>&1 | tee -a tmp_lora.txt
#done
#for model_name in  "/models/Qwen2-7B" "/data5/models/Phi-3-mini-4k-instruct/"  "/models/Mistral-7B-Instruct-v0.2" "/models/Llama-2-7b-hf-no-t"
#do
#  CUDA_VISIBLE_DEVICES=$device \
#  python3 main.py \
#  --model_name  $model_name \
#  --device 0 \
#  --group_size 32 \
#  --bits 4 \
#  --seqlen 8 \
#  --nsamples 8 \
#  --iters 1 \
#  --tasks "lambada_openai,hellaswag,winogrande,piqa,mmlu" \
#  --deployment_device 'fake' \
#  --eval_bs $eval_bs \
#  --disable_low_gpu_mem_usage \
#  --output_dir "/data5/wenhuach/test_mxfp4_wa_quant" \
#  2>&1 | tee -a test_mx_fp4_rtn.txt
#done

#for model_name in "/models/gpt-j-6B" "/models/Llama-2-7b-hf-no-t" "/models/Qwen2-7B"  "/models/opt-2.7b" "/models/mpt-7b" "/data5/models/Meta-Llama-3-8B-Instruct" "/models/dolly-v2-3b" "/models/Mistral-7B-Instruct-v0.2" "/data5/models/Phi-3-mini-4k-instruct/"
#do
#  CUDA_VISIBLE_DEVICES=$device \
#  python3 main.py \
#  --model_name  $model_name \
#  --device 0 \
#  --group_size 32 \
#  --bits 4 \
#  --iters 200 \
#  --tasks "lambada_openai,hellaswag,winogrande,piqa" \
#  --deployment_device 'fake' \
#  --eval_bs $eval_bs \
#  --disable_low_gpu_mem_usage \
#  --output_dir "/data5/wenhuach/test_mxfp4_no0.5" \
#  2>&1 | tee -a tmp_mxpf4.txt
#done

#for model_name in    "/models/Mistral-7B-Instruct-v0.2" "/data5/models/Phi-3-mini-4k-instruct/"
#do
#  CUDA_VISIBLE_DEVICES=$device \
#  python3 main.py \
#  --model_name  $model_name \
#  --device 0 \
#  --group_size 32 \
#  --bits 4 \
#  --iters 200 \
#  --lr 1e-2 \
#  --minmax_lr 5e-3 \
#  --tasks "lambada_openai,hellaswag,winogrande,piqa,mmlu" \
#  --deployment_device 'fake' \
#  --eval_bs $eval_bs \
#  --disable_low_gpu_mem_usage \
#  --output_dir "/data5/wenhuach/test_mxfp4_lr_1e-2minmax_lr_5e-3" \
#  2>&1 | tee -a tmp_mxpf4_lr_1e-2minmax_lr_5e-3.txt
#done

#for model_name in "/models/Meta-Llama-3-8B-Instruct" "/models/Qwen2-7B" "/data5/models/Phi-3-mini-4k-instruct/"  "/models/Mistral-7B-Instruct-v0.2" "/models/Llama-2-7b-hf-no-t"    "/models/gpt-j-6B"   "/models/opt-2.7b" "/models/mpt-7b"  "/models/dolly-v2-3b"
#do
#  CUDA_VISIBLE_DEVICES=$device \
#  python3 main.py \
#  --model_name  $model_name \
#  --device 0 \
#  --group_size 32 \
#  --bits 4 \
#  --iters 200 \
#  --lr 1e-2 \
#  --disable_minmax_tuning \
#  --minmax_lr 5e-3 \
#  --tasks "lambada_openai,hellaswag,winogrande,piqa,mmlu" \
#  --deployment_device 'fake' \
#  --eval_bs $eval_bs \
#  --disable_low_gpu_mem_usage \
#  --output_dir "/data5/wenhuach/test_mxfp4_lr_5e-3_disable_minmax" \
#  2>&1 | tee -a tmp_mxpf4_lr_5e-3_disable_minmax.txt
#done

#for model_name in "/models/opt-125m" "/models/opt-1.3b" "/models/opt-2.7b"  "/models/opt-6.7b" "/models/Llama-2-7b-hf-no-t" "/models/Meta-Llama-3-8B-Instruct"
#do
#  CUDA_VISIBLE_DEVICES=$device \
#  python3 main.py \
#  --model_name  $model_name \
#  --device 0 \
#  --group_size 32 \
#  --bits 2 \
#  --deployment_device 'fake' \
#  --disable_low_gpu_mem_usage \
#  --tasks lambada_openai,hellaswag,winogrande,piqa \
#  --eval_bs $eval_bs \
#  --output_dir "/data5/wenhuach/tmp2_tuning" \
#  2>&1 | tee -a tuning_tb_w8g-1.txt
#done


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
#--disable_tuning \
#--output_dir "/data5/wenhuach/tmp_disable_tuning" \
#2>&1 | tee -a disable_tuning.txt
#
#model_name="/models/Meta-Llama-3-8B-Instruct"
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
#--disable_tuning \
#--output_dir "/data5/wenhuach/tmp_disable_tuning" \
#2>&1 | tee -a disable_tuning.txt


#model_name="/models/Qwen2-0.5B-Instruct"
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--seqlen 4096 \
#--disable_eval \
#--deployment_device 'gpu,fake' \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/Qwen2-0.5B-instruct-W4G128asym-iter1000-seqlen4096" \
#2>&1 | tee -a tmp.txt
#
#
#model_name="/models/Qwen2-1.5B-Instruct"
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--disable_eval \
#--seqlen 4096 \
#--deployment_device 'gpu,fake' \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/Qwen2-1.5B-instruct-W4G128asym-iter1000-seqlen4096" \
#2>&1 | tee -a tmp.txt
#model_name="/models/Qwen2-7B"
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 200 \
#--model_dtype "float16" \
#--disable_eval \
#--quant_lm_head \
#--deployment_device 'autoround' \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/Qwen2-7B-W4G128asym-iter200-quant-lm-head-ar-float16" \
#2>&1 | tee -a tmp.txt
#
#model_name="/models/Qwen2-7B"
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 200 \
#--model_dtype "float16" \
#--disable_eval \
#--deployment_device 'autoround' \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/Qwen2-7B-W4G128asym-iter200-ar-float16" \
#2>&1 | tee -a tmp.txt

#model_name="/models/Qwen2-7B-Instruct"
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--model_dtype "float16" \
#--deployment_device 'auto_round,fake' \
#--quant_lm_head \
#--disable_low_gpu_mem_usage \
#--disable_eval \
#--output_dir "/data5/wenhuach/Qwen2-7B-Instruct-iter1000-lmhead" \
#2>&1 | tee -a tmp.txt

#model_name="/models/Qwen2-7B-Instruct"
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--model_dtype "float16" \
#--deployment_device 'auto_round,fake' \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/Qwen2-7B-Instruct-iter1000" \
#2>&1 | tee -a tmp.txt

#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 32 \
#--bits 2 \
#--iters 200 \
#--eval_bs $eval_bs \
#--disable_eval \
#--quant_lm_head \
#--minmax_lr 1e-2 \
#--deployment_device 'gpu' \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/llama3-iter200-w2g32-lmhead-w4g32-asym-autoround-minmaxlr1e-2" \
#2>&1 | tee -a tmp.txt

##
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 32 \
#--bits 4 \
#--iters 1000 \
#--sym \
#--disable_eval \
#--deployment_device 'gpu,fake' \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/Qwen1.5B-instruct-sym-iter1000" \
#2>&1 | tee -a tmp.txt

#model_name="/models/Qwen2-0.5B-Instruct"
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--seqlen 4096 \
#--disable_eval \
#--deployment_device 'gpu,fake' \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/Qwen0.5B-instruct-W4G128asym-iter1000-seqlen4096" \
#2>&1 | tee -a tmp.txt
#
#
#model_name="/models/Qwen2-1.5B-Instruct"
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--disable_eval \
#--seqlen 4096 \
#--deployment_device 'gpu,fake' \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/Qwen1.5B-instruct-W4G128asym-iter1000-seqlen4096" \
#2>&1 | tee -a tmp.txt
#

#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 32 \
#--bits 4 \
#--iters 1000 \
#--sym \
#--disable_eval \
#--disable_quanted_input \
#--deployment_device 'gpu,fake' \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/Qwen1.5B-instruct-sym-iter1000--disable_quanted_input" \
#2>&1 | tee -a tmp.txt


#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 32 \
#--bits 4 \
#--iters 1000 \
#--sym \
#--minmax_lr 2e-3 \
#--disable_eval \
#--disable_quanted_input \
#--deployment_device 'gpu,fake' \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/Qwen1.5B-instruct-sym-iter1000-minmaxlr-2e-3-disable_quanted_input" \
#2>&1 | tee -a tmp.txt


#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 32 \
#--bits 4 \
#--iters 1000 \
#--disable_eval \
#--sym \
#--disable_quanted_input \
#--deployment_device 'cpu,gpu,fake' \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/Qwen1.5-sym-iter200-disable_quant_input" \
#2>&1 | tee -a tmp.txt


#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 32 \
#--bits 4 \
#--iters 200 \
#--sym \
#--minmax_lr 1e-2 \
#--disable_eval \
#--deployment_device 'cpu,gpu,fake' \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/Qwen1.5-sym-iter200-minmaxlr1e-2" \
#2>&1 | tee -a tmp.txt
#
#
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 32 \
#--bits 4 \
#--iters 200 \
#--sym \
#--minmax_lr 1e-2 \
#--disable_eval \
#--disable_quanted_input \
#--deployment_device 'cpu,gpu,fake' \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/Qwen1.5-sym-iter200-disable_quant_input_minmaxlr1e-2" \
#2>&1 | tee -a tmp.txt


#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 32 \
#--bits 2 \
#--iters 200 \
#--eval_bs $eval_bs \
#--disable_eval \
#--deployment_device 'gpu' \
#--disable_low_gpu_mem_usage \
#--disabel_minmax_tuning \
#--output_dir "/data5/wenhuach/llama3-iter200-w2g32-no-lmhead-w4g32-asym-autoround-disable-minmax-tuning" \
#2>&1 | tee -a tmp.txt
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 32 \
#--bits 2 \
#--iters 1000 \
#--eval_bs $eval_bs \
#--disable_eval \
#--disable_quanted_input \
#--deployment_device 'gpu' \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/llama3-iter1000-w2g32-w4g32-asym-autoround" \
#2>&1 | tee -a tmp.txt
#
#
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 32 \
#--bits 2 \
#--iters 4000 \
#--eval_bs $eval_bs \
#--disable_eval \
#--disable_quanted_input \
#--deployment_device 'gpu' \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/llama3-iter4000-w2g32-w4g32-asym-autoround" \
#2>&1 | tee -a tmp.txt
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 32 \
#--bits 2 \
#--iters 1000 \
#--eval_bs $eval_bs \
#--disable_eval \
#--quant_lm_head \
#--deployment_device 'gpu' \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/llama3-iter1000-lmhead-w4g32-asym-autoround" \
#2>&1 | tee -a tmp.txt

#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 64 \
#--bits 4 \
#--iters 1000 \
#--eval_bs $eval_bs \
#--disable_eval \
#--deployment_device 'gpu' \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/falcon-7b-iter1000-w4g64" \
#2>&1 | tee -a tmp.txt

#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 64 \
#--bits 4 \
#--iters 1000 \
#--eval_bs $eval_bs \
#--disable_eval \
#--train_bs 16 \
#--nsamples 1024 \
#--deployment_device 'gpu,fake' \
#--disable_quanted_input \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/falcon-7b-iter1000-w4g64-disable_quanted_input-trainbs64_nsamples1024" \
#2>&1 | tee -a tmp.txt


#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--disable_low_gpu_mem_usage \
#--eval_bs $eval_bs \
#--disable_eval \
#--deployment_device 'gpu,fake' \
#--output_dir "/data5/wenhuach/gemma-7b-iter1000-fp16" \
#2>&1 | tee -a tmp.txt
#
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--disable_low_gpu_mem_usage \
#--eval_bs $eval_bs \
#--disable_eval \
#--minmax_lr 2e-3 \
#--disable_quanted_input \
#--deployment_device 'gpu,fake' \
#--output_dir "/data5/wenhuach/gemma-7b-iter1000-minmaxlr-2e-3-disable-quanted-input-fp16" \
#2>&1 | tee -a tmp.txt
#
#
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--disable_low_gpu_mem_usage \
#--eval_bs $eval_bs \
#--disable_eval \
#--disable_quanted_input \
#--deployment_device 'gpu,fake' \
#--output_dir "/data5/wenhuach/gemma-7b-iter1000-disable-quanted-input-fp16" \
#2>&1 | tee -a tmp.txt
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 200 \
#--scale_dtype  "fp16" \
#--minmax_lr 1e-2 \
#--deployment_device 'gpu' \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/Llama3-8b-hf-iter200-use-quant-input-w4g128-scale-dtype-fp16-minmaxlr1-2" \
#2>&1 | tee -a log_Llama3-8b-hf-iter200-with-lm-head-use-quant-input-w4g128-fp16-scale.txt

#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 200 \
#--quant_lm_head \
#--deployment_device 'gpu' \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/Llama3-8b-hf-iter200-with-lm-head-use-quant-input-w4g128-not-exporting" \
#2>&1 | tee -a log_Llama3-8b-hf-iter200-with-lm-head-use-quant-input-w4g128.txt

#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--scale_dtype  "fp16" \
#--quant_lm_head \
#--deployment_device 'gpu' \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/Llama3-8b-hf-iter1000-with-lm-head-use-quant-input-w4g128-scale-dtype-fp16-not-exporting" \
#2>&1 | tee -a log_Llama3-8b-hf-iter1000-with-lm-head-use-quant-input-w4g128-fp16-scale.txt



#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--scale_dtype  "fp16" \
#--quant_lm_head \
#--deployment_device 'gpu' \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/Llama3-8b-hf-iter1000-with-lm-head-use-quant-input-w4g128-scale-dtype-fp16-not-exporting" \
#2>&1 | tee -a log_Llama3-8b-hf-iter1000-with-lm-head-use-quant-input-w4g128-fp16-scale.txt


#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--quant_lm_head \
#--deployment_device 'gpu' \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/Llama3-8b-hf-iter1000-with-lm-head-use-quant-input-w4g128-not-exporting" \
#2>&1 | tee -a log_Llama3-8b-hf-iter1000-with-lm-head-use-quant-input-w4g128.txt



#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--use_quant_input \
#--minmax_lr 2e-3 \
#--quant_lm_head \
#--deployment_device 'gpu,fake' \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/Llama3-8b-hf-iter1000-with-lm-head-use-quant-input-w4g128-minmaxlr-2e-3" \
#2>&1 | tee -a log_Llama3-8b-hf-iter1000-with-lm-head-use-quant-input-w4g128-minmaxlr-2e-3.txt
#
#
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--quant_lm_head \
#--deployment_device 'gpu,fake' \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/Llama3-8b-hf-iter1000-with-lm-head-no-quant-input-w4g128" \
#2>&1 | tee -a log_Llama3-8b-hf-iter1000-with-lm-head-no-quant-input-w4g128.txt
#
#
#
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--minmax_lr 2e-3 \
#--quant_lm_head \
#--deployment_device 'gpu,fake' \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/Llama3-8b-hf-iter1000-with-lm-head-no-quant-input-w4g128-minmaxlr-2e-3" \
#2>&1 | tee -a log_Llama3-8b-hf-iter1000-with-lm-head-no-quant-input-w4g128-minmaxlr-2e-3.txt

#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--deployment_device 'gpu' \
#--use_quant_input \
#--output_dir "/data5/wenhuach/Baichuan2-7B-Chat-iter1000" \
#2>&1 | tee -a log_signroundv3_nosigmoid.txt
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--deployment_device 'fake,gpu' \
#--use_quant_input \
#--quantize_layers_outside_blocks \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/llama2-hf-chat-iter1k-use-quant-input" \
#2>&1 | tee -a log_signroundv3_nosigmoid.txt


#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--deployment_device 'fake,gpu' \
#--quantize_layers_outside_blocks \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/llama2-hf-chat-iter1k-no-quant-input" \
#2>&1 | tee -a log_signroundv3_nosigmoid.txt
#
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--deployment_device 'fake,gpu' \
#--use_quant_input \
#--minmax_lr 2e-3 \
#--quantize_layers_outside_blocks \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/llama2-hf-chat-iter1k-use-quant-input-minmax-lr2e-3" \
#2>&1 | tee -a log_signroundv3_nosigmoid.txt
#
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--deployment_device 'fake,gpu' \
#--minmax_lr 2e-3 \
#--quantize_layers_outside_blocks \
#--disable_low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/llama2-hf-chat-iter1k-no-quant-input-minmax-lr2e-3" \
#2>&1 | tee -a log_signroundv3_nosigmoid.txt


#model_name="/data5/llama3-70b"
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--minmax_lr 1e-2 \
#--deployment_device 'gpu' \
#--use_quant_input \
#--iters 10 \
#--quantize_layers_outside_blocks \
#--output_dir "/data5/wenhuach/tmp-llama3-70b-quant-lm-head" \
#2>&1 | tee -a log_signroundv3_nosigmoid.txt

#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--minmax_lr 2e-3 \
#--deployment_device 'gpu' \
#--output_dir "/data5/wenhuach/Baichuan2-7B-Chat-iter1000-no-quant-input-minmaxlr-2e-3-new" \
#2>&1 | tee -a log_signroundv3_nosigmoid.txt
#
#
#
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--deployment_device 'gpu' \
#--output_dir "/data5/wenhuach/Baichuan2-7B-Chat-iter1000-no-quant-input-new" \
#2>&1 | tee -a log_signroundv3_nosigmoid.txt
#


#
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--dataset "NeelNanda/pile-10k,madao33/new-title-chinese" \
#--deployment_device 'gpu' \
#--use_quant_input \
#--output_dir "/data5/wenhuach/Baichuan2-7B-Chat-iter1000-mixed" \
#2>&1 | tee -a log_signroundv3_nosigmoid.txt
#
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--dataset "NeelNanda/pile-10k,madao33/new-title-chinese" \
#--deployment_device 'gpu' \
#--output_dir "/data5/wenhuach/Baichuan2-7B-Chat-iter1000-no-quant-input-mixed" \
#2>&1 | tee -a log_signroundv3_nosigmoid.txt
#
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--dataset "NeelNanda/pile-10k,madao33/new-title-chinese" \
#--minmax_lr 2e-3 \
#--deployment_device 'gpu' \
#--use_quant_input \
#--output_dir "/data5/wenhuach/Baichuan2-7B-Chat-iter1000-minmaxlr-2e-3-mixed" \
#2>&1 | tee -a log_signroundv3_nosigmoid.txt
#
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--dataset "NeelNanda/pile-10k,madao33/new-title-chinese" \
#--minmax_lr 2e-3 \
#--deployment_device 'gpu' \
#--output_dir "/data5/wenhuach/Baichuan2-7B-Chat-iter1000-no-quant-input-minmaxlr-2e-3-mixed" \
#2>&1 | tee -a log_signroundv3_nosigmoid.txt


#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--seqlen 2048 \
#--deployment_device 'gpu' \
#--eval_bs $eval_bs \
#--use_quant_input \
#--minmax_lr 0.002 \
#--disable_eval \
#--output_dir "/data5/wenhuach/Qwen1.5-7B-Chat-asym-iter1000-minmaxlr-2e-3-no-use-quant-input" \
# 2>&1 | tee -a log_signroundv3_nosigmoid.txt

#
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--seqlen 2048 \
#--minmax_lr 2e-3 \
#--deployment_device 'fake,gpu' \
#--use_quant_input \
#--eval_bs $eval_bs \
#--disable_eval \
#--output_dir "/data5/wenhuach/dbrx-instruct-iter1000_use_quant_input_mimmaxlr_2e-3" \
# 2>&1 | tee -a log_signroundv3_nosigmoid.txt
#

#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--seqlen 2048 \
#--deployment_device 'fake,gpu' \
#--eval_bs $eval_bs \
#--disable_eval \
#--output_dir "/data5/wenhuach/dbrx-instruct-iter1000_no_use_quant_input" \
# 2>&1 | tee -a log_signroundv3_nosigmoid.txt
#
#
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--seqlen 2048 \
#--minmax_lr 2e-3 \
#--deployment_device 'fake,gpu' \
#--eval_bs $eval_bs \
#--disable_eval \
#--output_dir "/data5/wenhuach/dbrx-instruct-iter1000_no_use_quant_input_mimmaxlr_2e-3" \
#2>&1 | tee -a log_signroundv3_nosigmoid.txt


#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--seqlen 2048 \
#--enable_minmax_tuning \
#--minmax_lr 2e-3 \
#--deployment_device 'fake,cpu' \
#--scale_dtype 'fp32' \
#--eval_bs $eval_bs \
#--disable_lmeval \
#--low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/tmp_autoround_cpu-no_quant-input" \
#--amp 2>&1 | tee -a log_signroundv3_nosigmoid.txt

#model_name="/models/gemma-7b"
#
#
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--seqlen 2048 \
#--enable_minmax_tuning \
#--use_quant_input \
#--deployment_device 'gpu' \
#--scale_dtype 'fp32' \
#--eval_bs $eval_bs \
#--low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/autoround-iter1000-use-quant-input" \
#--amp 2>&1 | tee -a log_signroundv3_nosigmoid.txt

#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--seqlen 2048 \
#--enable_minmax_tuning \
#--deployment_device 'gpu' \
#--scale_dtype 'fp32' \
#--eval_bs $eval_bs \
#--low_gpu_mem_usage \
#--output_dir "/data5/wenhuach/autoround-iter1000-no-quant-input" \
#--amp 2>&1 | tee -a log_signroundv3_nosigmoid.txt




#
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--seqlen 2048 \
#--enable_minmax_tuning \
#--use_quant_input \
#--deployment_device 'gpu' \
#--scale_dtype 'fp32' \
#--eval_bs $eval_bs \
#--output_dir "/data5/wenhuach/autoround-iter1000-use-quant-input" \
#--amp 2>&1 | tee -a log_signroundv3_nosigmoid.txt

#model_name="/models/Mixtral-8x7B-v0.1"
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--seqlen 2048 \
#--enable_minmax_tuning \
#--deployment_device 'gpu' \
#--scale_dtype 'fp32' \
#--eval_bs $eval_bs \
#--low_gpu_mem_usage \
#--minmax_lr 0.0002 \
#--output_dir "/data5/wenhuach/autoround-iter1000-mimmax_lr0.0002" \
#--amp 2>&1 | tee -a log_signroundv3_nosigmoid.txt


#model_name="/models/Mixtral-8x7B-v0.1"
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--seqlen 2048 \
#--enable_minmax_tuning \
#--deployment_device 'gpu' \
#--scale_dtype 'fp32' \
#--eval_bs $eval_bs \
#--low_gpu_mem_usage \
#--use_quant_input \
#--output_dir "/data5/wenhuach/autoround-iter1000-use-quant-input" \
#--amp 2>&1 | tee -a log_signroundv3_nosigmoid.txt



#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--seqlen 2048 \
#--enable_minmax_tuning \
#--use_quant_input \
#--minmax_lr 0.0002 \
#--deployment_device 'cpu' \
#--scale_dtype 'fp32' \
#--eval_bs $eval_bs \
#--output_dir "/data3/wenhuach/autoround-iter1000_use_quant_input_minmax_lr_0.0002" \
#--amp 2>&1 | tee -a log_signroundv3_nosigmoid.txt


#for model_name in "/models/Mistral-7B-v0.1"  "/models/Llama-2-7b-hf" "/models/Llama-2-13b-hf" "/models/llama-7b"  "/models/llama-13b"
#do
#  eval_bs=32
#  CUDA_VISIBLE_DEVICES=$device \
#  python3 main.py \
#  --model_name  $model_name \
#  --device 0 \
#  --group_size 128 \
#  --bits 4 \
#  --iters 200 \
#  --seqlen 2048 \
#  --enable_minmax_tuning \
#  --eval_bs $eval_bs \
#  --output_dir "/data3/wenhuach/autoround_no_quant_input" \
#  --amp 2>&1 | tee -a log_signroundv3_nosigmoid.txt
#done
#
#for model_name in  "/models/Mistral-7B-v0.1"  "/models/Llama-2-7b-hf" "/models/Llama-2-13b-hf" "/models/llama-7b"  "/models/llama-13b"
#do
#  eval_bs=32
#  CUDA_VISIBLE_DEVICES=$device \
#  python3 main.py \
#  --model_name  $model_name \
#  --device 0 \
#  --group_size 128 \
#  --bits 3 \
#  --iters 200 \
#  --seqlen 2048 \
#  --enable_minmax_tuning \
#  --eval_bs $eval_bs \
#  --output_dir "/data3/wenhuach/autoround_no_quant_input" \
#  --amp 2>&1 | tee -a log_signroundv3_nosigmoid.txt
#done

#for model_name in  '/models/neural-chat-7b-v3-3' ""
#do
#  eval_bs=32
#  CUDA_VISIBLE_DEVICES=$device python3 eval/evaluation.py  --model_name  $model_name  --eval_bs $eval_bs
#done
#model_name='/models/phi-2'
#eval_bs=32
#
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--seqlen 2048 \
#--enable_minmax_tuning \
#--deployment_device 'fake' \
#--scale_dtype 'fp32' \
#--eval_bs $eval_bs \
#--low_gpu_mem_usage \
#--output_dir "/data3/wenhuach/autoround_iter1000_no_quant_input" \
#--amp 2>&1 | tee -a log_signroundv3.txt


#CUDA_VISIBLE_DEVICES=$device \
#python3 main_only_save.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--seqlen 2048 \
#--enable_minmax_tuning \
#--deployment_device 'fake' \
#--scale_dtype 'fp32' \
#--eval_bs $eval_bs \
#--use_quant_input \
#--low_gpu_mem_usage \
#--output_dir "/data3/wenhuach/autoround_iter1000" \
#--amp 2>&1 | tee -a log_signroundv3.txt




#CUDA_VISIBLE_DEVICES=$device \
#python3 main_only_save.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 200 \
#--seqlen 2048 \
#--minmax_lr 0.01 \
#--enable_minmax_tuning \
#--deployment_device 'cpu' \
#--scale_dtype 'fp32' \
#--eval_bs $eval_bs \
#--low_gpu_mem_usage \
#--output_dir "/data3/wenhuach/signroundv3-no-sigmoid_seq2048_iter200_no_quant_input_minmax_lr0.01" \
#--amp 2>&1 | tee -a log_signroundv3_nosigmoid.txt






#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--seqlen 2048 \
#--enable_minmax_tuning \
#--deployment_device 'cpu' \
#--scale_dtype 'fp32' \
#--eval_bs $eval_bs \
#--output_dir "/data3/wenhuach/signroundv3-no-sigmoid_seq2048_iter1000_no_quant_input_no_quant_input" \
#--amp 2>&1 | tee -a log_signroundv3_nosigmoid.txt

#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 200 \
#--seqlen 2048 \
#--minmax_lr 0.0375 \
#--enable_minmax_tuning \
#--use_quant_input \
#--deployment_device 'cpu' \
#--scale_dtype 'fp32' \
#--eval_bs $eval_bs \
#--output_dir "/data3/wenhuach/signroundv3-no-sigmoid_seq2048_iter200_minmaxlr_0.005" \
#--amp 2>&1 | tee -a log_signroundv3_nosigmoid.txt
#
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--seqlen 2048 \
#--minmax_lr 0.002 \
#--enable_minmax_tuning \
#--use_quant_input \
#--deployment_device 'cpu' \
#--scale_dtype 'fp32' \
#--eval_bs $eval_bs \
#--output_dir "/data3/wenhuach/signroundv3-no-sigmoid_seq2048_iter1000_minmaxlr_0.002" \
#--amp 2>&1 | tee -a log_signroundv3_nosigmoid.txt
#
#CUDA_VISIBLE_DEVICES=$device \
#python3 main.py \
#--model_name  $model_name \
#--device 0 \
#--group_size 128 \
#--bits 4 \
#--iters 1000 \
#--seqlen 2048 \
#--minmax_lr 0.0015 \
#--enable_minmax_tuning \
#--use_quant_input \
#--deployment_device 'cpu' \
#--scale_dtype 'fp32' \
#--eval_bs $eval_bs \
#--output_dir "/data3/wenhuach/signroundv3-no-sigmoid_seq2048_iter1000_minmaxlr_0.0015" \
#--amp 2>&1 | tee -a log_signroundv3_nosigmoid.txt

#for model_name in '/data3/wenhuach/signroundv3-no-sigmoid_seq2048_fp16_scale_Llama-2-13b-hf_w4_g128' '/data3/wenhuach/signroundv3-no-sigmoid_seq2048_fp16_scale_Llama-2-13b-hf_w4_g-1' '/data3/wenhuach/signroundv3-no-sigmoid_seq2048_fp16_scale_Llama-2-13b-hf_w3_g128' '/data3/wenhuach/signroundv3-no-sigmoid_seq2048_fp16_scale_Llama-2-13b-hf_w2_g128'
#do
#  eval_bs=32
#  CUDA_VISIBLE_DEVICES=$device python3 eval/evaluation.py  --model_name  $model_name  --eval_bs $eval_bs
#done



#for model_name in '/data3/wenhuach/signroundv3-no-sigmoid_seq2048_fp16_scale_llama-7b_w4_g128' '/data3/wenhuach/signroundv3-no-sigmoid_seq2048_fp16_scale_llama-7b_w4_g-1' '/data3/wenhuach/signroundv3-no-sigmoid_seq2048_fp16_scale_llama-7b_w3_g128' '/data3/wenhuach/signroundv3-no-sigmoid_seq2048_fp16_scale_llama-7b_w2_g128'
#do
#  eval_bs=32
#  CUDA_VISIBLE_DEVICES=$device python3 eval/evaluation.py  --model_name  $model_name  --eval_bs $eval_bs
#done
#
#for model_name in '/data3/wenhuach/signroundv3-no-sigmoid_seq2048_fp16_scale_llama-13b_w4_g128' '/data3/wenhuach/signroundv3-no-sigmoid_seq2048_fp16_scale_llama-13b_w4_g-1' '/data3/wenhuach/signroundv3-no-sigmoid_seq2048_fp16_scale_llama-13b_w3_g128' '/data3/wenhuach/signroundv3-no-sigmoid_seq2048_fp16_scale_llama-13b_w2_g128'
#do
#  eval_bs=32
#  CUDA_VISIBLE_DEVICES=$device python3 eval/evaluation.py  --model_name  $model_name  --eval_bs $eval_bs
#done


#for model_name in "/models/Mistral-7B-v0.1"  "/models/Llama-2-7b-hf" "/models/Llama-2-13b-hf" "/dataset/llama-7b"  "/dataset/llama-13b"
#do
#  eval_bs=32
#  CUDA_VISIBLE_DEVICES=$device \
#  python3 main.py \
#  --model_name  $model_name \
#  --device 0 \
#  --group_size 128 \
#  --num_bits 4 \
#  --iters 200 \
#  --seqlen 2048 \
#  --lr 0.005 \
#  --minmax_lr 0.005  \
#  --enable_minmax_tuning \
#  --use_quant_input \
#  --eval_bs $eval_bs \
#  --output_dir "/data3/wenhuach/signroundv3-no-sigmoid_seq2048_fp16_scale" \
#  --amp 2>&1 | tee -a log_signroundv3_nosigmoid.txt
#done
#
#for model_name in  "/models/Llama-2-7b-hf" "/models/Llama-2-13b-hf" "/dataset/llama-7b"  "/dataset/llama-13b"
#do
#  eval_bs=32
#  CUDA_VISIBLE_DEVICES=$device \
#  python3 main.py \
#  --model_name  $model_name \
#  --device 0 \
#  --group_size 128 \
#  --num_bits 4 \
#  --iters 200 \
#  --seqlen 2048 \
#  --lr 0.005 \
#  --minmax_lr 0.005  \
#  --enable_minmax_tuning \
#  --use_quant_input \
#  --eval_bs $eval_bs \
#  --output_dir "/data3/wenhuach/signroundv3-no-sigmoid_seq2048_fp16_scale" \
#  --amp 2>&1 | tee -a log_signroundv3_nosigmoid.txt
#done
#
#for model_name in   "/models/Llama-2-7b-hf" "/models/Llama-2-13b-hf" "/dataset/llama-7b"  "/dataset/llama-13b"
#do
#  eval_bs=32
#  CUDA_VISIBLE_DEVICES=$device \
#  python3 main.py \
#  --model_name  $model_name \
#  --device 0 \
#  --group_size -1 \
#  --num_bits 4 \
#  --iters 200 \
#  --seqlen 2048 \
#  --lr 0.005 \
#  --minmax_lr 0.005  \
#  --enable_minmax_tuning \
#  --use_quant_input \
#  --eval_bs $eval_bs \
#  --output_dir "/data3/wenhuach/signroundv3-no-sigmoid_seq2048_fp16_scale" \
#  --amp 2>&1 | tee -a log_signroundv3_nosigmoid.txt
#done
#
#for model_name in   "/models/Llama-2-7b-hf" "/models/Llama-2-13b-hf" "/dataset/llama-7b"  "/dataset/llama-13b"
#do
#  eval_bs=32
#  CUDA_VISIBLE_DEVICES=$device \
#  python3 main.py \
#  --model_name  $model_name \
#  --device 0 \
#  --group_size 128 \
#  --num_bits 3 \
#  --iters 200 \
#  --seqlen 2048 \
#  --lr 0.005 \
#  --minmax_lr 0.005  \
#  --enable_minmax_tuning \
#  --use_quant_input \
#  --eval_bs $eval_bs \
#  --output_dir "/data3/wenhuach/signroundv3-no-sigmoid_seq2048_fp16_scale" \
#  --amp 2>&1 | tee -a log_signroundv3_nosigmoid.txt
#done
#
#for model_name in  "/models/Llama-2-7b-hf" "/models/Llama-2-13b-hf" "/dataset/llama-7b"  "/dataset/llama-13b"
#do
#  eval_bs=32
#  CUDA_VISIBLE_DEVICES=$device \
#  python3 main.py \
#  --model_name  $model_name \
#  --device 0 \
#  --group_size 128 \
#  --num_bits 2 \
#  --iters 200 \
#  --seqlen 2048 \
#  --lr 0.005 \
#  --minmax_lr 0.005  \
#  --enable_minmax_tuning \
#  --use_quant_input \
#  --eval_bs $eval_bs \
#  --output_dir "/data3/wenhuach/signroundv3-no-sigmoid_seq2048_fp16_scale" \
#  --amp 2>&1 | tee -a log_signroundv3_nosigmoid.txt
#done