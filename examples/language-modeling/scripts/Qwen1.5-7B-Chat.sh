## Due to licensing restrictions, we are unable to release the model. 
#Recipe 1
python3 main.py \
--model_name Qwen/Qwen1.5-7B-Chat\
--bits 4 \
--group_size 128 \
--use_quant_input \
--iters 200 \
--deployment_device gpu \
--sym \
--minmax_lr 0.01 

