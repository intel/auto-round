for model_name in "falcon-three-7b" "Meta-Llama-3.1-8B-Instruct" "phi-4"; do
device=1
CUDA_VISIBLE_DEVICES=$device python -m auto_round \
        --format gguf:q4_k_s \
        --iters 200 \
        --model /models/${model_name} \
        --output_dir /data5/shiqi/${model_name}_tune \
        --eval_bs 16 \
        --tasks arc_challenge,arc_easy,boolq,hellaswag,lambada_openai,mmlu,openbookqa,piqa,truthfulqa_mc1,winogrande \
        --eval_task_by_task \
        2>&1 | tee /data5/shiqi/log/gguf_test/${model_name}_tune.log
done