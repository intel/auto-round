for model_name in "Qwen2.5-7B-Instruct"; do
device=0
CUDA_VISIBLE_DEVICES=$device python -m auto_round \
        --format "gguf:q4_k_m" \
        --model /models/${model_name} \
        --output_dir /data5/shiqi/model/${model_name} \
        --eval_bs 16 \
        --iters 200 \
        2>&1 | tee /data5/shiqi/log/${format}_${model_name}.log
done