export model_name=/data5/yliu7/HF_HOME/meta-llama/Llama-3.2-3B-Instruct
export model_name=/data5/yliu7/HF_HOME/meta-llama/Llama-3.2-1B-Instruct

CUDA_VISIBLE_DEVICES="" python3 main.py --model_name  $model_name  \
    --gradient_accumulate_steps 1 --model_dtype bfloat16 \
    --group_size 128  --train_bs 4 \
    --device cpu \
    --iters 2 \
    --nsample 4  \
    --format auto_gptq   --disable_quanted_input  \
    --tasks piqa \
    --output_dir ./autoround_output_dir