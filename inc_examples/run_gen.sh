export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_ENABLE_V1_MULTIPROCESSING=0



model_path="quantized_models/DeepSeek-V2-Lite-Chat-MXFP4/"
model_path="quantized_models/DeepSeek-V2-Lite-Chat-MXFP4"
model_path="quantized_model_qwen_mxfp8"
# model_path="quantized_model_ds_mxfp8"
# model_path="quantized_model_ds_mxfp4"
# model_path="quantized_model_qwen_mxfp4"
# model_path="quantized_model_qwen_mxfp4"
# model_path="quantized_models/Qwen3-235B-A22B-MXFP4"
# model_path="quantized_models/Qwen3-30B-A3B-Base-MXFP4"


# /home/yiliu7/workspace/torchutils/examples

# VLLM_ATTENTION_BACKEND=TRITON_ATTN \
# VLLM_LOGGING_LEVEL=DEBUG  \
VLLM_AR_MXFP4_MODULAR_MOE=1 \
VLLM_ENABLE_STATIC_MOE=0 \
VLLM_USE_DEEP_GEMM=0 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
python generate.py \
    --model ${model_path} \
    --max-tokens 16 \
    # --enforce-eager \
    # --tensor_parallel_size 4