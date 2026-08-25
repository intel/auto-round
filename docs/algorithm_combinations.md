# Algorithm Combinations

AutoRound can be combined with several weight/activation transform algorithms
before (or during) quantization. This page summarizes each combination and rates
it along two dimensions:

- **Accuracy Gain** — does the transform improve the accuracy of the quantized
  model compared with plain AutoRound?
- **Deployment** — can the resulting model actually be deployed/served today
  (kernel support, export path, real inference engine)?

## Legend

| Light | Meaning                                                         |
|:-----:|:----------------------------------------------------------------|
|  🟢   | Good — clear benefit / ready to deploy                          |
|  🟡   | Partial — conditional benefit / limited or experimental support |
|  🔴   | Poor — no measurable benefit / not deployable yet               |

> **Note:** A Partial/Poor **Accuracy Gain** rating may stem from two factors:
> (1) limitations in our current implementation, and (2) our own internal,
> subjective evaluation. Both are subject to change as the implementation matures
> and more benchmarks become available.

## Matrix

| Combination                                            | Accuracy Gain | Deployment | Details                                 | Reference                                            | Comments                                                                                         | CLI Usage                        |
|:-------------------------------------------------------|:-------------:|:----------:|:----------------------------------------|:-----------------------------------------------------|:-------------------------------------------------------------------------------------------------|:---------------------------------|
| AutoRound + AWQ (activation-aware scaling)             |      🟢       |     🟢     | [awq_details](awq_details.md)           | [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) | Recommended when activations are quantized (e.g., W4A4).                                         | `--algorithm awq,signround`      |
| AutoRound + Hadamard rotation                          |      🟢       |     🔴     | [rotation_details](rotation_details.md) | [arXiv:2404.00456](https://arxiv.org/abs/2404.00456) | Especially helpful for INT4 (W4A4) and some MXFP4 scenarios.   no production kernel.             | `--algorithm hadamard,signround` |
| AutoRound + SpinQuant                                  |      🟡       |     🔴     | [rotation_details](rotation_details.md) | [arXiv:2405.16406](https://arxiv.org/abs/2405.16406) | Learns rotation matrices; higher accuracy at extra training cost. no production kernel.          | Python API only                  |
| AutoRound + LFQ (logit-aware final-block quantization) |      🔴       |     🟢     | [lfq_acc](lfq_acc.md)                   | [arXiv:2605.29756](https://arxiv.org/abs/2605.29756) | Refines the final block to lift low-bit generation quality                                       | `--enable_lfq`                   |
| AutoRound + MX Attention (MX-format attention)         |      🟡       |     🔴     | [mxnv_acc](mxnv_acc.md)                 | [arXiv:2607.24377](https://arxiv.org/abs/2607.24377) | Applies MX formats to attention; experimental                                                    | Python API only                  |
| AutoRound + SVDQuant (low-rank outlier absorption)     |      🟡       |     🟡     | [svdquant_details](svdquant_details.md) | [arXiv:2411.05007](https://arxiv.org/abs/2411.05007) | Recommended for diffusion models; currently only FLUX is supported.                              | `--algorithm svdquant,signround` |

