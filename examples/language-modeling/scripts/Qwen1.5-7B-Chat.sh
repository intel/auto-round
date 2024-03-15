## Due to licensing restrictions, we are unable to release the model. 
#Recipe 1
python3 main.py \
--model_name Qwen/Qwen1.5-7B-Chat\
--bits 4 \
--group_size 128 \
--enable_minmax_tuning \
--amp \
--use_quant_input \
--n_samples 512 \
--iters 200 \
--deployment_device fake,gpu \
--tasks mmlu,ceval,cmmlu,gsm8k \
--sym \
--eval_bs 32 \
--minmax_lr 0.01 



# Recipe 2
python3 main.py \
--model_name Qwen/Qwen1.5-7B-Chat\
--bits 4 \
--group_size 128 \
--enable_minmax_tuning \
--amp \
--use_quant_input \
--n_samples 512 \
--iters 1000 \
--deployment_device fake,gpu \
--tasks mmlu,ceval,cmmlu,gsm8k \
--sym \
--eval_bs 32 \
--minmax_lr 0.002 
