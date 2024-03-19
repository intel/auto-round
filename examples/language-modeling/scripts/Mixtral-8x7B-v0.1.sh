python3 main.py \
--model_name  mistralai/Mixtral-8x7B-v0.1 \
--device 0 \
--group_size 128 \
--bits 4 \
--iters 1000 \
--deployment_device 'gpu' \
--scale_dtype 'fp32' \
--output_dir "./tmp_autoround"