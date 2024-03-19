python3 main.py \
--model_name  mistralai/Mixtral-8x7B-Instruct-v0.1 \
--device 0 \
--group_size 128 \
--bits 4 \
--iters 1000 \
--use_quant_input \
--deployment_device 'gpu' \
--eval_bs 32 \
--output_dir "./tmp_autoround"