### Prerequisite
- https://github.com/yiliu30/transformers/tree/457-ds32
- https://github.com/intel/auto-round/tree/ds-v32
  
### Quantize
```bash
export MODEL_NAME=/storage/yiliu7/deepseek-ai/DeepSeek-V3.2/
export OUTPUT_DIR=/storage/yiliu7/deepseek-ai/DeepSeek-V3.2-W416
python quant_ds_v32.py --model_name $MODEL_NAME --output_dir $OUTPUT_DIR
```

### Eval

- https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_2.html#launching-deepseek-v32
```
uv venv
source .venv/bin/activate
uv pip install git+https://github.com/deepseek-ai/DeepGEMM.git@v2.1.1.post3 --no-build-isolation
git clone https://github.com/vllm-project/vllm.git
git checkout 773d7073a
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
```

```bash
VLLM_ALLREDUCE_USE_SYMM_MEM=0  NCCL_NVLS_ENABLE=0  VLLM_USE_FUSED_MOE_GROUPED_TOPK=0 \
  vllm serve /storage/yiliu7/ds-v32-exp/ \
   --tensor-parallel-size 4 \
   --tokenizer-mode deepseek_v32 \
   --tool-call-parser deepseek_v32 \
   --enable-auto-tool-choice \
   --reasoning-parser deepseek_v3
```

```bash
lm_eval --model local-completions \
    --model_args "model=/storage/yiliu7/ds-v32-exp/,base_url=http://0.0.0.0:8000/v1/completions,max_length=8192,tokenized_requests=False,tokenizer_backend=None,num_concurrent=32" \
    --tasks gsm8k \
    --num_fewshot 5


VLLM_ALLREDUCE_USE_SYMM_MEM=0  NCCL_NVLS_ENABLE=0  VLLM_USE_FUSED_MOE_GROUPED_TOPK=0 vllm serve /storage/yiliu7/ds-v32-exp/    --tensor-parallel-size 4 
lm_eval --model local-completions --model_args "model=/storage/yiliu7/ds-v32-exp/"
# lm-eval --model local-completions --tasks gsm8k   --model_args model=/storage/yiliu7/ds-v32-exp/,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=100,max_retries=3,tokenized_requests=False
# lm-eval --model local-completions --tasks gsm8k   --model_args model=deepseek-ai/DeepSeek-V3.2-Exp,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=100,max_retries=3,tokenized_requests=False
```