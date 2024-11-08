## This recipe is outdated, we recommend using the latest recipe for best accuracy in homepage
python3 main.py \
--model_name  google/gemma-7b \
--device 0 \
--group_size 128 \
--bits 4 \
--iters 1000 \
--nsamples 512 \
--minmax_lr 2e-3 \
--format 'auto_round,auto_gptq' \
--model_dtype "float16" \
--output_dir "./tmp_autoround"