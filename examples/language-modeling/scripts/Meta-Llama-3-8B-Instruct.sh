## quant lm head
python3 main.py \
--model_name  meta-llama/Meta-Llama-3-8B-Instruct \
--device 0 \
--group_size 128 \
--bits 4 \
--iters 1000 \
--quant_lm_head \
--nsamples 512 \
--deployment_device 'gpu' \
--output_dir "./tmp_autoround"



## don't quant lm head
python3 main.py \
--model_name  meta-llama/Meta-Llama-3-8B-Instruct \
--device 0 \
--group_size 128 \
--bits 4 \
--iters 1000 \
--deployment_device 'gpu' \
--nsamples 512 \
--output_dir "./tmp_autoround"