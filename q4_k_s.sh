for model_name in "Qwen2.5-7B-Instruct" "falcon-three-7b"; do
device=2
format=gguf:q4_k_s,fake
CUDA_VISIBLE_DEVICES=$device python -m auto_round \
        --format ${format} \
        --iters 2 \
        --model /models/${model_name} \
        --output_dir /data5/shiqi/models/${model_name}_base \
        --eval_bs 16 \
        --tasks hellaswag,winogrande,truthfulqa_mc1,openbookqa,boolq,arc_easy,arc_challenge,mmlu,leaderboard_mmlu_pro \
        2>&1 | tee /data5/shiqi/log/${format}_${model_name}_base.log
done