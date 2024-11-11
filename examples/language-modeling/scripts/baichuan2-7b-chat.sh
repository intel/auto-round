## Due to licensing restrictions, we are unable to release the model.
## This recipe is outdated, we recommend using the latest recipe for best accuracy in homepage
python3 main.py \
--model_name baichuan-inc/Baichuan2-7B-Chat \
--group_size 128 \
--bits 4 \
--iters 1000 \
--nsamples 512 \
--format 'auto_round,auto_gptq' \
--minmax_lr 2e-3
