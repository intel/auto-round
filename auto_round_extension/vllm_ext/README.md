-  Build and Install vLLM 

```
git clone --branch fused-moe-ar https://github.com/yiliu30/vllm-fork.git
VLLM_USE_PRECOMPILED=1 pip install --editable . -vvv
```


- Enable vLLM-Ext at Runtime
```bash
VLLM_ENABLE_AR_EXT=1 vllm serve ...
```