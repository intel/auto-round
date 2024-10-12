W2G32 nsamples 512,iter 200, average accuracy of 10 tasks

| Models                     | gptq_sym | asym       | full_range_sym |
|----------------------------|----------|------------|----------------|
| Meta-Llama-3.1-8B-Instruct | 0.4500   | 0.52802    | **0.5381**     |
| Qwen2-7B                   | 0.5229   | **0.5559** | 0.5486         |

W4G128 nsamples 128,iter 200, average accuracy of 10 tasks

| Models                     | asym       | full_range_sym |
|----------------------------|------------|----------------|
| Meta-Llama-3.1-8B-Instruct | 0.6342     | **0.6370**     |
| Qwen2-7B                   | 0.6143     | **0.6167**     |
| Mistral-7B-Instruct-v0.2   | 0.6606     | **0.6635**     |
| Phi-3-mini-4k-instruct     | **0.6475** | 0.6432         |
