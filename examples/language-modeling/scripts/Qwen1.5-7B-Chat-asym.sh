python3 main.py \
--model_name  Qwen/Qwen1.5-7B-Chat\
--group_size 128 \
--bits 4 \
--iters 1000 \
--deployment_device 'gpu' \
--minmax_lr 2e-3 \
--use_quant_input

