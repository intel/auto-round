## This recipe is outdated, we recommend using the latest recipe for best accuracy in homepage
python3 main.py \
--model_name  tiiuae/falcon-7b \
--device 0 \
--group_size 64 \
--bits 4 \
--iters 1000 \
--nsamples 512 \
--format 'auto_round,auto_gptq' \
--output_dir "./tmp_autoround" \
--disable_quanted_input \