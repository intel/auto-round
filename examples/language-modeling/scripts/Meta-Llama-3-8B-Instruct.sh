## This recipe is outdated, we recommend using the latest recipe for best accuracy in homepage
## quant lm head
python3 main.py \
--model_name  meta-llama/Meta-Llama-3-8B-Instruct \
--device 0 \
--group_size 128 \
--bits 4 \
--iters 1000 \
--quant_lm_head \
--nsamples 512 \
--format 'auto_round,auto_gptq' \
--output_dir "./tmp_autoround"



## don't quant lm head
python3 main.py \
--model_name  meta-llama/Meta-Llama-3-8B-Instruct \
--device 0 \
--group_size 128 \
--bits 4 \
--iters 1000 \
--format 'auto_round,auto_gptq' \
--nsamples 512 \
--output_dir "./tmp_autoround"