# EXP: test  `3/4 scale` accuracy benefit for MXFP4_EXP on LLaMA3 and Qwen3
# https://arxiv.org/pdf/2502.20586

export CUDA_VISIBLE_DEVICES=6

# auto-round facebook/opt-125m --scheme MXFP4_EXP --tasks lambada_openai --iters 0 --limit 100

tasks="lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,boolq,arc_easy,arc_challenge,mmlu"

# llama 3
AR_EXP=1 auto-round /models/Meta-Llama-3.1-8B-Instruct --scheme MXFP4_EXP --tasks $tasks --iters 0 2>&1 |tee logs/exp1/llama3-8b-0-exp.log &
wait
auto-round /models/Meta-Llama-3.1-8B-Instruct --scheme MXFP4_EXP --tasks $tasks --iters 0 2>&1 |tee logs/exp1/llama3-8b-0.log &
wait

AR_EXP=1 auto-round /models/Meta-Llama-3.1-8B-Instruct --scheme MXFP4_EXP --tasks $tasks 2>&1 |tee logs/exp1/llama3-8b-200-exp.log &
wait


# Qwen3 8b iter 0
AR_EXP=1 auto-round /models/Qwen3-8B/ --scheme MXFP4_EXP --tasks $tasks --iters 0 2>&1 |tee logs/exp1/qwen3-8b-0-exp.log &
wait
auto-round /models/Qwen3-8B/ --scheme MXFP4_EXP --tasks $tasks --iters 0 2>&1 |tee logs/exp1/qwen3-8b-0.log &
wait
# Qwen3 8b
AR_EXP=1 auto-round /models/Qwen3-8B/ --scheme MXFP4_EXP --tasks $tasks 2>&1 |tee logs/exp1/qwen3-8b-200-exp.log &
wait

