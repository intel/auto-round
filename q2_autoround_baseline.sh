for model_name in "Qwen2.5-7B-Instruct" "Qwen3-8B " "Meta-Llama-3.1-8B-Instruct" "phi-4" "falcon-three-7b"; do
device=4
format=gguf:q2_k_s
CUDA_VISIBLE_DEVICES=$device python -m auto_round \
        --format ${format} \
        --data_type int_asym_dq \
        --group_size 16 \
        --super_bits 4 \
        --act_bits 16 \
        --super_group_size 16 \
        --bits 2 \
        --iters 0 \
        --asym \
        --model /models/${model_name} \
        --output_dir /data5/shiqi/${format}_${model_name}_autoround \
        --eval_bs 16 \
        --tasks 'leaderboard_ifeval,gsm8k' \
        2>&1 | tee /data5/shiqi/log/${format}_${model_name}_autoround.log
done