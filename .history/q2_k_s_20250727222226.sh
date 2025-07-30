for model_name in "Qwen3-0.6B"; do
device=1
CUDA_VISIBLE_DEVICES=$device python -m auto_round \
        --format gguf:q2_k_s \
        --model /models/${model_name} \
        --output_dir /data5/shiqi/${model_name} \
        --iters 200 \
        2>&1 | tee /data5/shiqi/log/${format}_${model_name}_search.log
done