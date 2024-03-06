## Due to licensing restrictions, we are unable to release the model. To run the int4 model on GPU, please install transformers==4.32.0, currently the 4.38.1 version has some issues.
python3 main.py \
--model_name  meta-llama/Llama-2-7b-chat-hf \
--device 0 \
--group_size 128 \
--bits 4 \
--iters 1000 \
--enable_minmax_tuning \
--use_quant_input \
--low_gpu_mem_usage \
--deployment_device 'gpu' \
--scale_dtype 'fp32' \
--eval_bs 32 \
--output_dir "./tmp_autoround" \
--amp
