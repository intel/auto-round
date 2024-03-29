## Due to licensing restrictions, we are unable to release the model. 
# Recipe
python3 main.py \
--model_name Qwen/Qwen1.5-7B-Chat\
--bits 4 \
--group_size 128 \
--use_quant_input \
--iters 200 \
--deployment_device gpu \
--sym \
--minmax_lr 0.01 

# Evaluation command for CEVAL, CMMLU, MMLU and GSM8K tasks
lm_eval --model hf \
    --model_args pretrained="gpu_model_path",autogptq=True,gptq_use_triton=True,trust_remote_code=True \
    --device cuda:0 --tasks ceval-valid,cmmlu,mmlu --batch_size 16

lm_eval --model hf \
    --model_args pretrained="gpu_model_path",autogptq=True,gptq_use_triton=True,trust_remote_code=True \
    --device cuda:0 --tasks gsm8k --batch_size 16 --num_fewshot 0