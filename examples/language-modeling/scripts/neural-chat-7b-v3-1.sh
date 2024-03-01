## enter to language-modeling fold
python3 main.py \
--model_name  /models/neural-chat-7b-v3-1 \
--device 0 \
--group_size 128 \
--bits 4 \
--iters 1000 \
--enable_minmax_tuning \
--minmax_lr 0.0002 \
--deployment_device 'gpu' \
--scale_dtype 'fp32' \
--eval_bs 32 \
--output_dir "./tmp_autoround" \
--amp