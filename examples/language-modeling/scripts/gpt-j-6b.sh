python3 main.py \
--model_name  EleutherAI/gpt-j-6b \
--device 0 \
--group_size 128 \
--bits 4 \
--iters 1000 \
--disable_quanted_input \
--nsamples 512 \
--deployment_device 'gpu' \
--output_dir "./tmp_autoround"
