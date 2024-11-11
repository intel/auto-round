## This recipe is outdated, we recommend using the latest recipe for best accuracy in homepage
## Due to licensing restrictions, we are unable to release the model.
python3 main.py \
--model_name 01-ai/Yi-6B-Chat \
--group_size 128 \
--bits 4 \
--iters 1000 \
--format 'auto_round,auto_gptq' \
--minmax_lr 2e-3


