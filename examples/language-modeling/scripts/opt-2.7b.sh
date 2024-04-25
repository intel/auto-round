python3 main.py \
--model_name  facebook/opt-2.7b \
--device 0 \
--group_size 128 \
--bits 4 \
--iters 1000 \
--minmax_lr 0.002 \
--deployment_device 'gpu' \
--output_dir "./tmp_autoround"
