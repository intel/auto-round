### The GGUF results

We tested with a fake model because the main branch currently has layer name mismatches between Transformers and GGUF.

~~~bash
python3 -m auto_round Qwen/Qwen3-8B     --options "gguf:q2_k_s,gguf:q4_k_s"     --target_bits 3.5     --ignore_scale_zp_bits     --iters 0     --format fake     --output_dir "./test_gguf"
~~~


eval

~~~
# Start vLLM serve
vllm serve ./test_gguf/Qwen3-8B-w2g16/ --port 8000 --max-model-len 8192 --host 127.0.0.1 --served-model-name qwen3-8b

# Perform five repeated evaluations on math_500 and gpqa_diamond.
evalscope eval --model qwen3-8b --api-url http://127.0.0.1:8000/v1   --api-key EMPTY   --datasets math_500 gpqa_diamond  --eval-batch-size 64  --generation-config "{\"n\":5}"

# Evaluate mmlu_pro
evalscope eval --model qwen3-8b --api-url http://127.0.0.1:8001/v1   --api-key EMPTY   --datasets mmlu_pro  --eval-batch-size 128
~~~
| evalscope, options q2ks,q4ks avgbits 3.5, ignore_scale_zp | math_500 (repeat=5) | gpqa_diamond (repeat=5) | mmlu_pro |
|-----------------------------------------------------------|---------------------|-------------------------|----------|
| qwen3-8b: BF16                                            | 0.8083              | 0.4586                  | 0.6934   |
| qwen3-8b: Fake quantized model                            | 0.7924              | 0.4313                  | 0.6751   |

| evalscope, options q2ks,q4ks avgbits 3.5, ignore_scale_zp | math_500 (repeat=5) | gpqa_diamond (repeat=5) | mmlu_pro |
|-----------------------------------------------------------|---------------------|-------------------------|----------|
| qwen3.5-4b: BF16                                          | 0.5365              | 0.3263                  | 0.5891   |
| qwen3.5-4b: Fake quantized model                          | 0.505               | 0.3172                  | 0.5948   |


### Other results

We use **lm-eval** for evaluation. For LLaMA, we enabled `add_bos_token` and
`removed @use_kernel_forward_from_hub("RMSNorm")`
in [modeling_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L52C1-L52C40)
to stabilize accuracy during evaluation. All other settings follow the default configurations of AutoRound and lm-eval.

We ignore the scale and zp bits in the tables below. The accuracy may change a little as we modified a little of the
implementation. We will rerun all the experiments.

For mxfp experiment, we use fake model while for weight only model we use real model. **No tuning is applied unless
explicitly stated.**

*Average accuracy across `lambada_openai`, `hellaswag`, `piqa`, `winogrande`, and `mmlu`.*

### Table 1 MXFP4/8 mixed accuracy.

| Average bits     | Llama3.1-8B-I  |  Qwen2.5-7B-I  |    Qwen3-8B    |   Qwen3-32B    |
|:-----------------|:--------------:|:--------------:|:--------------:|:--------------:|
| **BF16**         | 0.7076 (100%)  | 0.7075 (100%)  | 0.6764 (100%)  | 0.7321 (100%)  |
| **Pure 4-bit**   | 0.6626 (93.6%) | 0.6550 (92.6%) | 0.6316 (93.4%) | 0.6901 (94.3%) |
| **Ours 4.5-bit** | 0.6808 (96.2%) | 0.6776 (95.8%) | 0.6550 (96.8%) | 0.7176 (98.0%) |
| **Ours 5-bit**   | 0.6857 (96.9%) | 0.6823 (96.4%) | 0.6594 (97.5%) | 0.7201 (98.3%) |
| **Ours 6-bit**   | 0.6975 (98.6%) | 0.6970 (98.5%) | 0.6716 (99.3%) | 0.7303 (99.8%) |

We compare the proposed method against naive layer-wise bit allocation strategies, such as assigning higher
precision to the network’s head((near lm-head) or tailad(close to embedding)) layers, to demonstrate its relative
performance advantages.

### Table 2  Comparison with other recipes at an average of 5 bits of mxfp datatype

| Avg. bits = 5         |   Llama3.1-8B-I    |    Qwen2.5-7B-I    |      Qwen3-8B      |
|:----------------------|:------------------:|:------------------:|:------------------:|
| **Tail layers 8-bit** |   0.6671 (94.3%)   |   0.6616 (93.5%)   |   0.6410 (94.8%)   |
| **Head layers 8-bit** |   0.6657 (94.1%)   |   0.6686 (94.5%)   |   0.6356 (94.0%)   |
| **Ours**              | **0.6857 (96.9%)** | **0.6823 (96.4%)** | **0.6594 (97.5%)** |

### Table 3  Comparison with other recipes at an average of 4.5 bits of mxfp datatype

| Avg. bits = 4.5       |   Llama3.1-8B-I    |    Qwen2.5-7B-I    |      Qwen3-8B      |
|:----------------------|:------------------:|:------------------:|:------------------:|
| **Tail layers 8-bit** |   0.6614 (93.5%)   |   0.6535 (92.4%)   |   0.6373 (94.2%)   |
| **Head layers 8-bit** |   0.6568 (92.8%)   |   0.6642 (93.9%)   |   0.6305 (93.2%)   |
| **Ours**              | **0.6808 (96.2%)** | **0.6776 (95.5%)** | **0.6550 (95.8%)** |

### Table4   Comparison with other recipes at an average of 3 bits of W2G128 and W4G128

| Avg. bits = 4.5       | Llama3.1-8B-I | Qwen2.5-7B-I | Qwen3-8B |
|:----------------------|:-------------:|:------------:|:--------:|
| **Tail layers 4-bit** |    0.6058     |    0.3798    |  0.4536  |
| **Head layers 4-bit** |    0.3198     |    0.3270    |  0.3196  |
| **Ours**              |    0.6148     |    0.4058    |  0.4862  |
