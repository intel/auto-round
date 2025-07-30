for model_name in "Qwen3-8B"; do
device=0
CUDA_VISIBLE_DEVICES=$device python -m auto_round \
        --format gguf:q2_k_s \
        --model "/models/Qwen3-8B" \
        --output_dir "/data5/shiqi/model/Qwen3-8B" \
        --iters 0 \
        2>&1 | tee /data5/shiqi/log/${format}_${model_name}_search.log
done