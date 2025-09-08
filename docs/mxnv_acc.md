Average accuracy of hellaswag,lambada_openai,mmlu,piqa,winogrande.

We evaluated using a fake model since we currently have no access to devices for running the real models. However, we have verified that in most cases the fake model closely matches the real model.

| mxfp4 g32         | llama3.1-8B-Instruct | Qwen2-7.5-Instruct | Phi4    | Qwen3-32B |
|-------------------|----------------------|--------------------|---------|-----------|
| RTN               | 0.62124              | 0.65502            | 0.71674 | 0.69006   |
| AutoRound         | 0.66862              | 0.67588            | 0.72472 | 0.72106   |
| AutoRound+alg_ext | 0.6732               | 0.68094            | 0.72252 | 0.72012   |

| nvfp4  g16        | llama3.1-8B-Instruct | Qwen2-7.5-Instruct | Phi4    | Qwen3-32B |
|-------------------|----------------------|--------------------|---------|-----------|
| RTN               | 0.68756              | 0.6906             | 0.72962 | 0.71636   |
| AutoRound         | 0.69184              | 0.69728            | 0.73058 | 0.73062   |
| AutoRound+alg_ext | 0.69648              | 0.6989             | 0.7318  | 0.72948    |