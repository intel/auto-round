# SVDQuant（实验性功能）

SVDQuant 将 Linear 权重拆成一个量化残差分支和一个较小的浮点低秩分支：

```text
W ~= Q(R) + U @ V
Linear(x) ~= QuantizedLinear(x, Q(R)) + Linear(Linear(x, V), U)
```

该功能采用 [SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit
Diffusion Models](https://arxiv.org/abs/2411.05007) 提出的分解方法。
AutoRound 可以将该变换与 RTN 或 SignRound 组合，并导出 Nunchaku 可直接加载的
MXFP4 FLUX pipeline。

> 目前端到端导出流程仅在 FLUX.1-dev 上完成验证。其他使用兼容 Diffusers
> FLUX block 类的模型仍属于实验性支持，运行时会输出警告。

## 快速开始

请在隔离环境中安装 AutoRound 和 diffusion 依赖：

```bash
pip install --no-build-isolation -e .
pip install -r test/test_cuda/requirements_diffusion.txt
```

### 默认 RTN 流程

建议先从 rank 32、一次 residual iteration、关闭 smooth、使用普通 RTN 的默认配置开始：

```bash
CUDA_VISIBLE_DEVICES=0 auto-round-rtn \
  --model /path/to/FLUX.1-dev \
  --model_dtype bf16 \
  --scheme MXFP4 \
  --algorithm svdquant \
  --device 0 \
  --output_dir ./flux-dev-mxfp4-svdquant-rtn
```

选择 `svdquant` 且未指定 `--format` 时，CLI 会自动使用
`svdquant_nunchaku`。

### 默认 SignRound 流程

选择 `auto_round` 即可使用 SignRound 作为最终量化算法：

```bash
CUDA_VISIBLE_DEVICES=0 auto-round \
  --model /path/to/FLUX.1-dev \
  --model_dtype bf16 \
  --scheme MXFP4 \
  --algorithm svdquant,auto_round \
  --dataset /path/to/coco2017-captions.tsv \
  --batch_size 1 \
  --device 0 \
  --low_gpu_mem_usage \
  --output_dir ./flux-dev-mxfp4-svdquant-signround
```

Caption dataset 是包含 `id` 和 `caption` 两列的 TSV 文件：

```text
id	caption
0	A photo of a cat
1	A city street at night
```

### 可选的质量参数

Smooth 搜索和更多 residual iterations 默认不开启，可按需添加：

```bash
--enable-svdquant-smooth \
--svdquant-smooth-num-grids 20 \
--svdquant-smooth-max-calibration-calls 128 \
--svdquant-residual-iters 20 \
--enable-svdquant-residual-early-stop
```

Diffusion calibration 中，`--nsamples` 控制 prompt 数量，
`--num_inference_steps` 控制去噪步数。建议先用小配置跑通端到端流程，再运行完整质量配置。

## 处理流程

SVDQuant 是结构变换预处理器，后面必须接一个最终量化算法：

```text
原始 Linear 权重
  -> 可选的 activation-aware smooth 搜索
  -> 对相关 projection 分组
  -> 低秩分解和 residual iteration
  -> 用 SVDQuantLinear 替换 nn.Linear
       -> 量化 residual_linear
       -> 浮点 lora_down
       -> 浮点 lora_up
       -> 可选 input smooth factor
  -> RTN 或 SignRound
  -> Nunchaku 导出
```

FLUX 的 Q/K/V projection 会作为一组处理：共享一个低秩 down factor，并保留各自的
up factor，以匹配运行时布局。

## Residual iteration

默认 `residual_iters=1` 时：

1. 对分组权重 `W` 计算指定 rank 的低秩分解。
2. 使用 `low_rank_dtype` 物化 `U` 和 `V`。
3. 计算残差 `R = W - U @ V`。
4. 由最终的 RTN 或 SignRound 量化 residual linear。

当 `residual_iters > 1` 时，每次 outer iteration 会交替执行两步：对
`W - Q(R_previous)` 拟合新的低秩分支，再对新残差执行与部署一致的 RTN QDQ。
Early stop 会保留最佳候选，并在目标误差变差时停止。

关闭 smooth 时使用权重重建误差；开启 smooth 时使用 calibration output error。
Residual iterations 与 SignRound 的 `--iters` 是两个独立参数。

## Smooth 搜索

令 `G = --svdquant-smooth-num-grids`，候选集合为：

```text
(alpha, beta) = (0, 0)
(alpha, 0)，其中 alpha 属于 {1/G, ..., (G-1)/G}
(alpha, 1-alpha)，alpha 取值同上
```

默认 `G=20`，共生成 39 组候选。每组候选计算
`x_span**alpha / w_span**beta`，并与浮点 group output 比较，选择有限且最小的
output error。

`--svdquant-smooth-max-calibration-calls` 使用 reservoir sampling，限制每个
group 保留的 calls 数量。未被选中的 call 不会复制到 smooth CPU cache。该参数不会
代替 SignRound 自身的 calibration samples。

## 配置项

| Python 字段 | CLI 参数 | 默认值 | 含义 |
|---|---|---:|---|
| `rank` | `--svdquant-rank` | `32` | 低秩分支 rank。 |
| `smooth_enabled` | `--enable-svdquant-smooth` | `False` | 开启 Alpha/Beta 搜索。 |
| `smooth_num_grids` | `--svdquant-smooth-num-grids` | `20` | 候选网格精度。 |
| `smooth_max_calibration_calls` | `--svdquant-smooth-max-calibration-calls` | `128` | 每组保留的 calls 上限。 |
| `residual_iters` | `--svdquant-residual-iters` | `1` | Residual outer iterations。 |
| `residual_early_stop` | `--enable-svdquant-residual-early-stop` | `False` | 目标误差变差后停止。 |
| `low_rank_dtype` | `--svdquant-low-rank-dtype` | `bf16` | BF16、FP16 或 FP32 factor。 |
| `target_modules` | `--svdquant-target-modules` | `None` | 需要处理的模块名子串。 |
| `exclude_modules` | `--svdquant-exclude-modules` | `None` | 排除的模块名子串。 |
| `model_adapter` | `--svdquant-model-adapter` | `auto` | 运行时分组和导出 adapter。 |

## Python API

```python
from auto_round import AutoRound
from auto_round.algorithms.quantization.rtn.config import RTNConfig
from auto_round.algorithms.transforms.svdquant import SVDQuantConfig

autoround = AutoRound(
    "/path/to/FLUX.1-dev",
    scheme="MXFP4",
    model_dtype="bf16",
    alg_configs=[
        SVDQuantConfig(rank=32, model_adapter="flux"),
        RTNConfig(disable_opt_rtn=True),
    ],
    device_map=0,
    low_gpu_mem_usage=True,
)
autoround.quantize_and_save(
    "./flux-dev-mxfp4-svdquant-rtn",
    format="svdquant_nunchaku",
)
```

## 导出与推理

`svdquant_nunchaku` 当前要求 MXFP4 E2M1 W4A4、group size 32、weight 和
activation 均为对称量化、dynamic activation quantization，并使用受支持的运行时
adapter。量化过程不会导入 Nunchaku；只有推理时需要安装 Nunchaku。

输出目录是一个完整的 Diffusers pipeline，可以直接加载：

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "./flux-dev-mxfp4-svdquant-signround",
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
pipe.enable_model_cpu_offload()

image = pipe(
    "A cat holding a sign that says Hello world",
    num_inference_steps=20,
    guidance_scale=3.5,
    generator=torch.Generator(device="cuda").manual_seed(12345),
    height=512,
    width=512,
).images[0]
image.save("flux-svdquant-mxfp4.png")
```

## 当前限制

- Runtime-loadable export 目前只在 FLUX.1-dev 上完成验证。
- SVDQuant 要求 `nblocks=1`。
- Smooth 搜索会针对每组 Alpha/Beta 候选重放所有保留 calls。
- 增加 residual iterations 会重复执行分解和 QDQ。
- Smoke 图片只能验证加载和数值稳定性，不能代替数据集级的生成质量评估。
