export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_ENABLE_V1_MULTIPROCESSING=0



model_path="quantized_models/DeepSeek-V2-Lite-Chat-MXFP4/"
model_path="quantized_models/DeepSeek-V2-Lite-Chat-MXFP4"
model_path="quantized_models/Qwen3-30B-A3B-Base-MXFP4"


# VLLM_ATTENTION_BACKEND=TRITON_ATTN \
VLLM_USE_DEEP_GEMM=0 \
VLLM_LOGGING_LEVEL=DEBUG  \
VLLM_ENABLE_V1_MULTIPROCESSING=1 \
python generate.py \
    --model ${model_path} \
    --max-tokens 64 \
    --enforce-eager