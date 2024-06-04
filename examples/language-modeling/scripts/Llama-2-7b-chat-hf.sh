## Due to licensing restrictions, we are unable to release the model. To run the int4 model on GPU, please install transformers==4.32.0, currently the 4.38.1 version has some issues.
python3 main.py \
--model_name  meta-llama/Llama-2-7b-chat-hf \
--device 0 \
--group_size 128 \
--bits 4 \
--iters 1000 \
--deployment_device 'gpu' \
--output_dir "./tmp_autoround"
