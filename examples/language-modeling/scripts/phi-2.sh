python3 main.py \
--model_name  microsoft/phi-2 \
--device 0 \
--group_size 128 \
--bits 4 \
--iters 1000 \
--sym \
--nsamples 512 \
--deployment_device 'gpu' \
--output_dir "./tmp_autoround"