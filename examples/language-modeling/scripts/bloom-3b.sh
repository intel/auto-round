python3 main.py \
--model_name  bigscience/bloom-3b \
--device 0 \
--group_size 128 \
--bits 4 \
--iters 1000 \
--use_quant_input \
--deployment_device 'gpu' \
--output_dir "./tmp_autoround"
