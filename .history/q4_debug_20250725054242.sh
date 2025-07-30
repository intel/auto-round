for model_name in "Qwen3-0.6B"; do
device=3
format=gguf:q2_k_s,fake
CUDA_VISIBLE_DEVICES=$device python -m auto_round \
        --format ${format} \
        --iters 2 \
        --model /models/${model_name} \
        --output_dir /data5/shiqi/models/${model_name} \
        --eval_bs 16 \
        --tasks lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,boolq,arc_easy,arc_challenge,mmlu,leaderboard_ifeval,leaderboard_mmlu_pro,gsm8k \
        2>&1 | tee /data5/shiqi/log/${format}_${model_name}_choice_new.log
done