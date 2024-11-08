## This recipe is outdated, we recommend using the latest recipe for best accuracy in homepage
python3 main.py \
--model_name  google/gemma-7b-it \
--device 0 \
--group_size 128 \
--bits 4 \
--iters 1000 \
--minmax_lr 2e-3 \
--model_dtype "float16" \
--format 'auto_round,auto_gptq' \
--nsamples 512 \
--output_dir "./tmp_autoround"