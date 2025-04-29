models="Qwen2.5-7B-Instruct" "falcon-three-7b" "Meta-Llama-3.1-8B-Instruct" "phi-4"
for model_name in "Qwen2.5-1.5B-Instruct"; do
device=2
format=q2_k_s
CUDA_VISIBLE_DEVICES=$device python -m auto_round \
        --format ${format} \
        --data_type int_asym_dq \
        --group_size 16 \
        --super_bits 4 \
        --act_bits 16 \
        --super_group_size 16 \
        --bits 2 \
        --iters 200 \
        --asym \
        --model /models/${model_name} \
        --output_dir /data5/shiqi/gguf_${format}_${model_name}_search_no_tune \
        --eval_bs 16 \
        --tasks lambada_openai,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,piqa,truthfulqa_mc1,winogrande \
        2>&1 | tee /data5/shiqi/log/gguf_test/gguf_${format}_${model_name}_search_no_tune.log
done