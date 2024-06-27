python3 main.py \
--model_name  tiiuae/falcon-7b \
--device 0 \
--group_size 64 \
--bits 4 \
--iters 1000 \
--deployment_device 'gpu' \
--output_dir "./tmp_autoround" \
--disable_quanted_input \
--disable_low_gpu_mem_usage