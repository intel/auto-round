for model_name in "Qwen2.5-7B-Instruct"; do
device=5
format=fake
CUDA_VISIBLE_DEVICES=$device python -m auto_round \
        --format fake \
        --data_type int_asym_dq \
        --group_size 32 \
        --super_bits 6 \
        --super_group_size 8 \
        --bits 4 \
        --iters 0 \
        --model /models/${model_name} \
        --output_dir /data5/shiqi/${format}_${model_name}_search \
        --eval_bs 16 \
        --tasks arc_challenge,arc_easy,boolq,hellaswag,lambada_openai,mmlu,openbookqa,piqa,truthfulqa_mc1,winogrande \
        2>&1 | tee /data5/shiqi/log/gguf_test/${format}_${model_name}_search.log
done