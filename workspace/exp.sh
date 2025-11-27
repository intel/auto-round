# EXP: test  `3/4 scale` accuracy benefit for MXFP4 on LLaMA3 and Qwen3

# llama 3
AR_EXP=1 auto-round /models/Meta-Llama-3.1-8B-Instruct --scheme MXFP4 --tasks gsm8k,mmlu,piqa,winogrande,hellaswag --iters 0 2>&1 |tee logs/exp1/llama3-8b-0-exp.log &
wait
auto-round /models/Meta-Llama-3.1-8B-Instruct --scheme MXFP4 --tasks gsm8k,mmlu,piqa,winogrande,hellaswag --iters 0 2>&1 |tee logs/exp1/llama3-8b-0.log &
wait

AR_EXP=1 auto-round /models/Meta-Llama-3.1-8B-Instruct --scheme MXFP4 --tasks gsm8k,mmlu,piqa,winogrande,hellaswag 2>&1 |tee logs/exp1/llama3-8b-200-exp.log &
wait


# Qwen3 8b iter 0
AR_EXP=1 auto-round /models/Qwen3-8B/ --scheme MXFP4 --tasks gsm8k,mmlu,piqa,winogrande,hellaswag --iters 0 2>&1 |tee logs/exp1/qwen3-8b-0-exp.log &
wait
auto-round /models/Qwen3-8B/ --scheme MXFP4 --tasks gsm8k,mmlu,piqa,winogrande,hellaswag --iters 0 2>&1 |tee logs/exp1/qwen3-8b-0.log &
wait
# Qwen3 8b
AR_EXP=1 auto-round /models/Qwen3-8B/ --scheme MXFP4 --tasks gsm8k,mmlu,piqa,winogrande,hellaswag 2>&1 |tee logs/exp1/qwen3-8b-200-exp.log &
wait

