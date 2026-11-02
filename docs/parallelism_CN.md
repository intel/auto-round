# QAD 并行支持

AutoRound 将张量并行（TP）交给 Transformers：由 Transformers 加载权重分片、
安装原生前向逻辑并完成布局重分布。AutoRound 仅适配本地权重分片的
量化与反量化（QDQ）计算，不另行实现一套张量并行后端。

## 公共配置

`ParallelConfig` 独立于算法配置，提供三个正整数维度：`tp_size`、`pp_size`
和 `dp_size`，默认值均为 `1`。Python 接口也支持通过 `parallel_config`
传入等价的映射。

```python
from auto_round import AutoRound, ParallelConfig
from auto_round.algorithms.quantization.qad.config import QADConfig

compressor = AutoRound(
    "Qwen/Qwen3-0.6B",
    scheme="W4A16",
    alg_configs=QADConfig(iters=200, group_size=128),
    parallel_config=ParallelConfig(tp_size=4),
    enable_torch_compile=False,
    low_cpu_mem_usage=False,
)
compressor.quantize_and_save("./quantized", format="auto_round")
```

将示例保存为 `train_qad_tp.py`，启动全部 TP rank：

```bash
torchrun --nproc-per-node 4 train_qad_tp.py
```

CLI 提供相同的配置维度：

```bash
torchrun --nproc-per-node 4 -m auto_round quantize Qwen/Qwen3-0.6B --alg qad --tp_size 4 --scheme W4A16
```

TP 要求 Transformers 版本同时提供 `DistributedConfig` 和原生 DTensor
训练路径，且模型配置定义了 `base_model_tp_plan`。AutoRound 将
`DistributedConfig(tp_size=..., tp_plan="auto")` 传给原生加载接口，
不使用已弃用的顶层 `tp_plan` 参数。传入已加载的模型时，模型必须已经
完成原生 TP 配置，并同时提供 tokenizer。

不要在启用 TP 时传入自定义 `device_map`。AutoRound 根据 `LOCAL_RANK`
绑定各进程的设备，且 `WORLD_SIZE` 必须等于 `tp_size`。必要时由原生加载
流程初始化分布式 mesh；AutoRound 不会销毁调用方持有的进程组。
建议使用 NVLink 等高速互连。

## 支持范围

- 采用普通列切分和行切分 Linear 方案的稠密 decoder-only 语言模型，
  包括原生 embedding／输出边界以及复制的归一化层。
- 下文所述的标准 decoder-only fused MoE 布局，已使用 Qwen3-MoE 测试。
- INT、MXFP4 和 NVFP4 仅权重量化的 QAD 及其硬标签 QAT 变体；TP 暂不支持 block 预处理器
  或多阶段量化流水线。
- 一维 TP mesh，权重分片必须大小一致且非空。
- 正整数 group size。行切分的本地分片必须包含完整的量化组；
  `group_size=-1` 仅支持列切分权重。
- 梯度检查点及现有 QAD 优化器。暂不支持非量化参数联合训练、norm／bias
  调优、激活量化、`torch.compile`、非标准 fused MoE 布局和 AutoScheme。

不支持的组合会明确报错，不会静默退回单卡或改变量化方案。
公共 QDQ 适配层可以复用，但逐 block 算法仍需独立验证校准与调度路径，
才能声明支持原生 TP。

## FP4 仅权重量化

`MXFP4` 和 `NVFP4` 预设默认采用 W4A4。当前原生 TP 必须显式覆盖
`act_bits=16, act_data_type="float"`；W4A4 会报错，不会被静默转换。
必须同时设置两个字段，否则 NVFP4 中带位宽的激活类型会恢复 `act_bits=4`。
例如，将上文的方案和算法配置替换为：

```python
scheme = "MXFP4"
algorithm = QADConfig(iters=200, act_bits=16, act_data_type="float", group_size=32)
```

将它们作为 `scheme=scheme, alg_configs=algorithm` 传给 `AutoRound`。
使用 NVFP4 时，设置 `scheme="NVFP4"` 和 `group_size=16`。两种格式均要求
`bits=4` 和 `sym=True`；此 TP 适配层尚未开放其他 FP4 变体。
行切分后的本地输入宽度必须能被相应的 block size 整除。
不兼容的布局会明确报错，不会切开 block 或改变其 scale。

MXFP4 的 block scale 在本地计算。NVFP4 通过 TP mesh 内的标量 `MAX`
归约，为每个 Linear 计算统一的权重 global scale，再执行本地 block QDQ，
不会为计算该 scale 聚合完整权重。打包检查点导出也会保留 global scale。

## Fused MoE 与原生 TP

支持标准拼接式、非转置、无 bias 的 experts，其三维权重为 `gate_up_proj`
和 `down_proj`，原生 plan 使用 `packed_colwise`、`rowwise` 和
`moe_tp_experts`。中间维分片必须均匀且与量化组对齐。
AutoRound 将本地 packed gate/up shard 拆成逐 expert 的 Linear 视图，
不会先重建全局张量。原始 experts 模块及原生 TP forward 保持不变，
继续负责输入梯度和输出归约；子 Linear 仅执行本地计算，避免重复 TP 通信。

TP 固定使用 experts 的 `linear_loop` 实现，而不是 AutoRound 的 grouped
后端。冻结的 fused 参数保留为零拷贝别名，以维持原生 TP 的布局判断；
CPU 导出排除这些别名并重建拆分的 expert 模块树。
通用 unfuse 路径遇到 DTensor 时会在修改权重前报错。暂不支持 EP、转置／
交错布局、expert bias／额外 norm，以及未被 TP 分片的 fused experts。

双进程测试比较 fused／unfused 输出、输入／路由／QDQ 梯度和未命中 expert
的行为，再运行 Qwen3-MoE 的 INT／MXFP4／NVFP4 仅权重调优、CPU 导出
logits 和 FP4 打包。这不代表已验证 fused kernel 性能或 CUDA/NCCL 执行。

## 校准与检查点导出

同一 TP 组的所有 rank 使用相同的校准样本和优化采样索引。
teacher 前向由全部 rank 参与，各 rank 使用独立的 teacher 缓存目录，
并协同判断缓存是否命中，避免某个分片缓存缺失导致 collective 执行路径不一致。
该方案会有意复制较小的 teacher Top-K 缓存，而不是完整的 teacher logits。

必须在 **所有 TP rank** 上调用 `quantize()`、`quantize_and_save()` 和
`save_quantized()`。检查点聚合是 collective 操作，但只有 rank 0 写文件。
导出完成状态和写入失败信息会同步给所有参与的 rank。
导出先将本地分片复制到 CPU，再以有界字节块（每块 4 MiB）传输，
不会对加速器权重调用 `DTensor.full_tensor()`。仅 rank 0 重建完整 CPU 张量，
后续打包也在 CPU 执行。
导出会重建包含完整权重、量化元数据及非持久化 buffer 的非 TP CPU 模型。
写入进程需要足够容纳完整模型的主机内存；目前没有实现流式／分片量化检查点导出。
原始 TP 模型保持不变。非写入 rank 调用
`save_quantized(return_folders=True)` 时，返回 TP 模型和空目录列表。
应在 collective 聚合完成后，仅由 rank 0 评测导出的模型。

离线双进程 CPU/Gloo 测试覆盖 FP4 前向及 STE 梯度一致性、QAD/QAT 生命周期、
普通导出模型的 logits 和真实 `auto_round` 打包结果。
CUDA/NCCL 显存、性能及部署后端推理仍未验证。

## PP 与后续 DP

`ParallelConfig(pp_size=N)` 将现有单进程设备放置路径配置到前 `N` 张可见
CUDA 设备。这不是微批次流水线训练调度，QAD 目前也不支持此路径。
不启用并行配置维度时，现有 `device_map` 调用方式保持不变。

`dp_size` 预留独立的数据并行维度；大于一的值及 TP／PP 混合组合目前
均抛出 `NotImplementedError`。本接口不替换现有实验性的 block 级 DDP 行为。
后续 DP 支持需要显式管理数据切分、副本进程组和梯度同步，
而不能将 TP 进程组当成 DP 进程组使用。
