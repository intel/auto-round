### Prerequisite
- https://github.com/yiliu30/transformers/tree/457-ds32
- https://github.com/intel/auto-round/tree/ds-v32
  
### Quantize
```bash
python quant_ds_v32.py
```

### Eval

- https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_2.html#launching-deepseek-v32
```
uv venv
source .venv/bin/activate
uv pip install git+https://github.com/deepseek-ai/DeepGEMM.git@v2.1.1.post3 --no-build-isolation
git clone https://github.com/vllm-project/vllm.git
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
```

```bash
VLLM_ALLREDUCE_USE_SYMM_MEM=0  NCCL_NVLS_ENABLE=0  VLLM_USE_FUSED_MOE_GROUPED_TOPK=0 \
  vllm serve /storage/yiliu7/DeepSeek-V3.2-fp8-w4a16/ \
   --tensor-parallel-size 4 \
   --tokenizer-mode deepseek_v32 \
   --tool-call-parser deepseek_v32 \
   --enable-auto-tool-choice \
   --reasoning-parser deepseek_v3
```

```bash
lm_eval --model local-completions \
    --model_args "model=/storage/yiliu7/DeepSeek-V3.2-fp8-w4a16/,base_url=http://0.0.0.0:8000/v1/completions,max_length=8192,tokenized_requests=False,tokenizer_backend=None,num_concurrent=32" \
    --tasks gsm8k \
    --num_fewshot 5
```