### Quantize Model

- MXFP8
```bash
export QWEN_MODEL=Qwen/Qwen3-235B-A22B
export DS_MODEL=deepseek-ai/DeepSeek-R1

python quantize.py --model $QWEN_MODEL -t qwen_mxfp8 --use_autoround_format  
python quantize.py --model $DS_MODEL -t ds_mxfp8 --use_autoround_format  


- MXFP4
```bash
export QWEN_MODEL=Qwen/Qwen3-235B-A22B
export DS_MODEL=deepseek-ai/DeepSeek-R1
python quantize.py --model $QWEN_MODEL -t qwen_mxfp4 --use_autoround_format 
python quantize.py --model $DS_MODEL -t qwen_mxfp4 --use_autoround_format 
```


### Prompt Tests
- MXFP8
```bash
export model_path=/path/to/quantized_model
tp_size=8
VLLM_AR_MXFP4_MODULAR_MOE=0 \
VLLM_ENABLE_AR_EXT=1 \
VLLM_MXFP4_PRE_UNPACK_TO_FP8=0 \
VLLM_ENABLE_STATIC_MOE=0 \
VLLM_MXFP4_PRE_UNPACK_WEIGHTS=0 \
VLLM_USE_DEEP_GEMM=0 \
VLLM_ENABLE_V1_MULTIPROCESSING=1 \
    python generate.py \
    --model ${model_path} \
    --tensor_parallel_size $tp_size \
    --max-tokens 16 \
    --max-num-seqs 4  \
    --gpu_memory_utilization 0.75 \
    --no-enable-prefix-caching \
    --enable_expert_parallel
```

- MXFP4
```bash
export model_path=/path/to/quantized_model
tp_size=8
VLLM_AR_MXFP4_MODULAR_MOE=1 \
VLLM_ENABLE_AR_EXT=1 \
VLLM_MXFP4_PRE_UNPACK_TO_FP8=1 \
VLLM_ENABLE_STATIC_MOE=0 \
VLLM_MXFP4_PRE_UNPACK_WEIGHTS=0 \
VLLM_USE_DEEP_GEMM=0 \
VLLM_ENABLE_V1_MULTIPROCESSING=1 \
    python generate.py \
    --model ${model_path} \
    --tensor_parallel_size $tp_size \
    --max-tokens 16 \
    --max-num-seqs 4 \
    --gpu_memory_utilization 0.75 \
    --no-enable-prefix-caching \
    --enable_expert_parallel
```

### Evaluation Tests

WIP



