## This recipe is outdated, we recommend using the latest recipe for best accuracy in homepage
python3 main.py \
--model_name  mistralai/Mistral-7B-Instruct-v0.2 \
--device 0 \
--group_size 128 \
--bits 4 \
--iters 1000 \
--nsamples 512 \
--format 'auto_round,auto_gptq' \
--output_dir "./tmp_autoround"