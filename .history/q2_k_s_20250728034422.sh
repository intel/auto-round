for model_name in "Qwen2.5-7B-Instruct"; do
device=1
CUDA_VISIBLE_DEVICES=$device python -m auto_round \
        --format gguf:q2_k_s,fake \
        --model /models/${model_name} \
        --output_dir /data5/shiqi/models \
        --iters 200 \
        --tasks lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,boolq,arc_easy,arc_challenge,mmlu \
        2>&1 | tee /data5/shiqi/log/${format}_${model_name}_search.log
done