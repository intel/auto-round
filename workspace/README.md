 
 ```bash
############################### Gaudi model path #############################################
deepspeed --include="localhost:0,1,2,3" --master_port=29500 quantize.py   --autoround --batch_size  16 --accuracy  --dtype mx_fp4  --mp_ratio 4/7   2>&1 |tee mxfp4_ratio_2_8b.log

deepspeed --include="localhost:4,5,6,7" --master_port=29510 quantize.py   --autoround --batch_size  16 --accuracy  --dtype mx_fp4  --mp_ratio 3/7   2>&1 |tee mxfp4_ratio_3_8b.log

deepspeed  --include="localhost:0,1,2,3,4,5,6,7" --master_port=29510  quantize.py  --model_name_or_path /git_lfs/data/pytorch/llama3.3/Meta-Llama-3.3-70B-Instruct/  --batch_size  8 --accuracy  --dtype mx_fp4   --mp_ratio 4/7 --autoround 2>&1 |tee mxfp4_op_4_3.3_70b.log

deepspeed  --include="localhost:0,1,2,3,4,5,6,7" --master_port=29510  quantize.py  --model_name_or_path /git_lfs/data/pytorch/llama3.3/Meta-Llama-3.3-70B-Instruct/  --batch_size  8 --accuracy  --dtype mx_fp4   --mp_ratio 5/7 --autoround 2>&1 |tee mxfp4_op_5_3.3_70b.log


############################### H20 model path #############################################
deepspeed --include="localhost:4,5,6,7" --master_port=29500 quantize.py   --model_name_or_path  /ssd/xinhe/Llama-3.1-8B-Instruct/ --autoround --batch_size  64 --accuracy  --dtype mx_fp4  --device cuda --mp_ratio 3/7   2>&1 |tee mxfp4_ratio_3_8b.log

deepspeed --include="localhost:2,3" --master_port=29510 quantize.py   --model_name_or_path  /ssd/xinhe/Llama-3.1-8B-Instruct/ --autoround --batch_size 32 --accuracy  --dtype mx_fp4  --device cuda --mp_ratio 3/7   2>&1 |tee mxfp4_ratio_3_8b.log


deepspeed --include="localhost:0,1,2,3" --master_port=29500  quantize.py  --model_name_or_path /ssd/xinhe/Llama-3.3-70B-Instruct/  --batch_size  32 --accuracy  --dtype mx_fp4   --device cuda   --mp_ratio 4/7 --autoround   2>&1 |tee mxfp4_ratio_2_3.3_70b.log 


deepspeed --include="localhost:4,5,6,7" --master_port=29510  quantize.py  --model_name_or_path /ssd/xinhe/Llama-3.3-70B-Instruct/  --batch_size  32 --accuracy  --dtype mx_fp4   --device cuda   --mp_ratio 3/7 --autoround   2>&1 |tee mxfp4_ratio_3_3.3_70b.log 


```
python quantize.py  --model_name_or_path facebook/opt-125m  --batch_size 8 --accuracy  --dtype mx_fp4   --mp_ratio 1/3 --autoround
