## This recipe is outdated, we recommend using the latest recipe for best accuracy in homepage
python3 main.py \
--model_name  facebook/opt-2.7b \
--device 0 \
--group_size 128 \
--bits 4 \
--iters 1000 \
--minmax_lr 0.002 \
--nsamples 512 \
--disable_quanted_input \
--format 'auto_round,auto_gptq' \
--output_dir "./tmp_autoround"
