python3 main.py \
--model_name  microsoft/phi-2 \
--device 0 \
--group_size 128 \
--bits 4 \
--iters 1000 \
--deployment_device 'gpu' \
--disable_trust_remote_code \
--output_dir "./tmp_autoround" \
--disable_quanted_input