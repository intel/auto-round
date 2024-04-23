## quant lm head
python3 main.py \
--model_name  meta-llama/Meta-Llama-3-8B-Instruct \
--device 0 \
--group_size 128 \
--bits 4 \
--iters 1000 \
--use_quant_input \
--quant_lm_head \
--deployment_device 'gpu' \
--disable_low_gpu_mem_usage \
--output_dir "./tmp_autoround"



## don't quant lm head
python3 main.py \
--model_name  meta-llama/Meta-Llama-3-8B-Instruct \
--device 0 \
--group_size 128 \
--bits 4 \
--iters 1000 \
--use_quant_input \
--deployment_device 'gpu' \
--disable_low_gpu_mem_usage \
--output_dir "./tmp_autoround"