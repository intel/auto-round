## This recipe is outdated, we recommend using the latest recipe for best accuracy in homepage
python3 main.py \
--model_name  Intel/neural-chat-7b-v3-3 \
--device 0 \
--group_size 128 \
--bits 4 \
--iters 1000 \
--minmax_lr 2e-3 \
--nsamples 512 \
--format 'auto_round,auto_gptq' \
--output_dir "./tmp_autoround" \
--disable_quanted_input