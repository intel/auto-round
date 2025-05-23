for model_name in "Qwen2.5-7B-Instruct" "falcon-three-7b" "Meta-Llama-3.1-8B-Instruct" "phi-4"; do
device=2
format=fake
CUDA_VISIBLE_DEVICES=$device python -m auto_round \
        --format ${format} \
        --data_type int_asym_dq \
        --group_size 32 \
        --super_bits 6 \
        --super_group_size 8 \
        --bits 4 \
        --iters 200 \
        --model /models/${model_name} \
        --output_dir /data5/shiqi/model/q4_${format}_${model_name}_baseline \
        --eval_bs 16 \
        --tasks lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,boolq,arc_easy,arc_challenge,mmlu,leaderboard_ifeval,leaderboard_mmlu_pro,gsm8k \
        2>&1 | tee /data5/shiqi/log/q4_auto_round_${format}_${model_name}_baseline.log
done