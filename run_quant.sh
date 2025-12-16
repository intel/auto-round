#--enable_torch_compile 
model_path=/mnt/disk8/yiliu7/zai-org/GLM-4.5-Air
# model_path=/mnt/disk8/zai-org/GLM-4.5/
# model_path=/mnt/disk8/Qwen/Qwen3-8B-FP8
# model_path=/mnt/disk8/meta-llama/Llama-3.2-3B-Instruct
output_dir=/mnt/disk2/Llama-3.2-3B-Instruct-test-no-lm-head-fp8-attn
# output_dir=/mnt/disk8/yiliu7/zai-org/GLM-4.5-lm-head-fp8-attn-g2
# output_dir=/mnt/disk8/yiliu7/zai-org/GLM-4.5-air-no-lm-head-fp8-attn-g2
echo "Start AutoRound quantization with model ${model_path}"
AR_LOG_LEVEL=TRACE  PT_HPU_LAZY_MODE=0 python \
    auto_round/__main__.py --model ${model_path} \
    --scheme FP8_STATIC  \
    --iters 0 \
    --disable_opt_rtn  \
    --batch_size 8  \
    --quant_lm_head \
    --low_gpu_mem_usage \
    --static_attention_dtype fp8 \
    --output_dir ${output_dir} \
       --enable_torch_compile
echo "Finish AutoRound quantization with model ${model_path}"
# AR_LOG_LEVEL=TRACE PT_HPU_LAZY_MODE=1 python  auto_round/__main__.py --model ${model_path} --scheme FP8_STATIC  --iters 0  --disable_opt_rtn

    # --quant_lm_head \