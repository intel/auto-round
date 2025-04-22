model="/models/Qwen2.5-7B-Instruct/"
device=1
CUDA_VISIBLE_DEVICES=$device python -m auto_round \
        --format gguf:q4_k_s \
        --iters 200 \
        --model $model \
        --output /data5/shiqi/ \
        --eval_bs 16 \
        --tasks arc_challenge,arc_easy,boolq,hellaswag,lambada_openai,mmlu,openbookqa,piqa,truthfulqa_mc1,winogrande \
        --eval_task_by_task \
        2>&1 | tee /data5/shiqi/log/gguf_test/gguf_double_Qwen2.5-7B-Instruct.log