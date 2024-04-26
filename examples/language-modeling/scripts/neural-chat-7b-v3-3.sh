python3 main.py \
--model_name  Intel/neural-chat-7b-v3-3 \
--device 0 \
--group_size 128 \
--bits 4 \
--iters 1000 \
--minmax_lr 0.0002 \
--deployment_device 'gpu' \
--output_dir "./tmp_autoround" \
--disable_quanted_input \