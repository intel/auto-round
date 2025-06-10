export model_name=/data5/yliu7/HF_HOME/meta-llama/Llama-3.2-3B-Instruct
# export model_name=/data5/yliu7/HF_HOME/meta-llama/meta-llama/Llama-3.2-1B-Instruct

python3 main.py --model_name  $model_name  \
    --gradient_accumulate_steps 2 --model_dtype bfloat16 \
    --group_size 128  --train_bs 4 \
    --iters 200 \
    --nsample 512  \
    --format auto_gptq   --disable_quanted_input  \
    --tasks piqa \
    --output_dir ./llama-3b-ins-2



# INFO:lm-eval:Running loglikelihood requests
# Running loglikelihood requests: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3676/3676 [01:11<00:00, 51.59it/s]
# |Tasks|Version|Filter|n-shot| Metric |Value |   |Stderr|
# |-----|------:|------|-----:|--------|-----:|---|-----:|
# |piqa |      1|none  |     0|acc     |0.7519|±  |0.0101|
# |     |       |none  |     0|acc_norm|0.7595|±  |0.0100|