## This recipe is outdated, we recommend using the latest recipe for best accuracy in homepage
python3 main.py \
--model_name  google/gemma-2b \
--device 0 \
--group_size 128 \
--bits 4 \
--iters 400 \
--model_dtype "float16" \
--nsamples 512 \
--format 'auto_round,auto_gptq' \
--output_dir "./tmp_autoround"