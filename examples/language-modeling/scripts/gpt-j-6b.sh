## This recipe is outdated, we recommend using the latest recipe for best accuracy in homepage
python3 main.py \
--model_name  EleutherAI/gpt-j-6b \
--device 0 \
--group_size 128 \
--bits 4 \
--iters 1000 \
--disable_quanted_input \
--nsamples 512 \
--format 'auto_round,auto_gptq' \
--output_dir "./tmp_autoround"
