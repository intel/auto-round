#--enable_torch_compile 
model_path=/mnt/disk8/yiliu7/zai-org/GLM-4.5-Air
model_path=/mnt/disk8/zai-org/GLM-4.5/
AR_LOG_LEVEL=TRACE  PT_HPU_LAZY_MODE=0 python  auto_round/__main__.py --model ${model_path} --scheme FP8_STATIC  --iters 0 --enable_torch_compile --disable_opt_rtn  --batch_size 64 --output_dir /mnt/disk2/GLM45_air_compile_fp8_static
# AR_LOG_LEVEL=TRACE PT_HPU_LAZY_MODE=1 python  auto_round/__main__.py --model ${model_path} --scheme FP8_STATIC  --iters 0  --disable_opt_rtn
