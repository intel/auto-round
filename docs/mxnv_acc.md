Average accuracy of hellaswag,lambada_openai,mmlu,piqa,winogrande.

We evaluated using a fake model since we currently have no access to devices for running the real models. However, we have verified that in most cases the fake model closely matches the real model.

| mxfp4 g32         | llama3.1-8B-Instruct | Qwen2-7.5-Instruct | Phi4    | Qwen3-32B |
|:-------------------|:----------------------:|:--------------------:|:---------:|:-----------:|
| RTN               | 0.6212               | 0.6550            | 0.7167 | 0.6901   |
| AutoRound         | 0.6686               | 0.6758            | 0.7247 | 0.7211   |
| AutoRound+alg_ext | 0.6732               | 0.6809            | 0.7225 | 0.7201   |

| nvfp4  g16        | llama3.1-8B-Instruct | Qwen2-7.5-Instruct | Phi4    | Qwen3-32B |
|:-------------------|:----------------------:|:--------------------:|:---------:|:-----------:|
| RTN               | 0.6876              | 0.6906             | 0.7296 | 0.7164      |
| AutoRound         | 0.6918              | 0.6973             | 0.7306 | 0.7306      |
| AutoRound+alg_ext | 0.6965              | 0.6989             | 0.7318  | 0.7295     |
