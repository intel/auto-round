## Due to licensing restrictions, we are unable to release the model.
python3 main.py \
--model_name 01-ai/Yi-6B-Chat \
--group_size 128 \
--bits 4 \
--iters 1000 \
--deployment_device 'gpu' \
--minmax_lr 2e-3 \
--use_quant_input
