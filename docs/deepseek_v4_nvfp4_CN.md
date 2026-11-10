# DeepSeek V4 NVFP4

需要包含 `DeepseekV4GroupedLinear` 的 Transformers（CPU 回归测试使用
Transformers 5.14.1）。原生 FP8 检查点使用完整的 Transformers 反量化加载器，
不会使用通用 meta/磁盘流式加载器，即使设置了 `AR_DISK_STREAM_MODEL=1`。
需要预留大量主机内存，包括 MoE 拆分时的副本。

## 使用方法

请使用有代表性的分词校准数据，并为模型提供足够的主机和设备内存：

```python
from auto_round import AutoRound

quantizer = AutoRound(
    "/path/to/DeepSeek-V4-Flash",
    scheme="NVFP4",
    iters=0,
    disable_opt_rtn=True,
    disable_model_free=True,
    batch_size=1,
    nsamples=128,
    seqlen=2048,
    device_map="auto",
)
quantizer.quantize_and_save("./DeepSeek-V4-Flash-NVFP4", format="auto_round:llm_compressor")
```

`format="auto_round"` 也支持分组打包。解码器重放会保留用于哈希路由的真实
`input_ids`，并缓存字典形式的 RoPE 输入。

## 支持范围

- 分组 `wo_a` / Transformers `self_attn.o_a_proj`：权重仅支持 NVFP4，4 位，
  分组大小为 16，无偏置；输入宽度必须能被 16 整除，输出宽度必须能被组数整除。
  不支持的分组格式会明确报错。
- 激活支持分组大小为 16、带校准全局尺度的 NVFP4，或仅量化权重
  （层配置中设置 `act_bits=16`）。不支持其他分组激活格式。
- 两种导出格式均保留 NVFP4 打包字节、分组尺度和全局尺度。兼容的 Transformers
  保存接口会对打包参数和缓冲区设置 `save_original_format=False`，不修改源转换元数据。
- V4 保存的权重 key 和模块名元数据使用原始 Hub 命名：
  `layers.N.ffn.experts.E.w1/w2/w3`、`layers.N.attn.wo_a`、`embed`、`head`，
  而不是 Transformers 加载后的 `model.layers.N.mlp/self_attn` 命名。
  未量化的融合专家拆回逐专家矩阵；只修改名称或拆分视图，不重新应用原生 FP8 量化。
  完整保存和即时分片保存遵循相同命名约定，内存中的模型仍使用 Transformers 命名。
- AutoRound 推理在 backend 为 `auto` 时，为分组层选择 `auto_round:torch_nvfp4`。
  该实现解包权重并以输入 dtype 执行分组 BMM，仅用于正确性参考，不是快速服务内核。

## 验证与服务

CPU 测试覆盖原生加载器选择和反量化参数、路由 ID、字典 RoPE、两种分组打包器、
打包状态重加载，以及一个真实的小型 V4 注意力 RTN/导出流程。小型流程保留 MoE
权重为全精度；不验证完整检查点加载、专家校准覆盖率或精度。

使用 vLLM 服务该导出模型需要独立且兼容的 DeepSeek V4 NVFP4 适配器，以处理
量化元数据和分组投影。原始 Hub 命名本身不保证服务兼容性。
本改动不提供该适配器，也不证明原生 vLLM 服务支持。
本次未测试 GPU 内核、完整模型生成或 thinking 精度。

## AutoRound / vLLM 职责边界

AutoRound 侧工作统一放在 PR #2606。目标是导出自描述的标准 compressed-tensors
检查点，而不是依靠运行时适配器修补检查点。

| 职责 | 归属 | 状态 |
|---|---|---|
| 原始 Hub 权重名称，包括专家和分组投影 | AutoRound 导出 | 已实现并完成名称审计 |
| NVFP4 打包、分组布局、尺度与打包字节保护 | AutoRound 量化/导出 | 已有 CPU 回归 |
| compressed-tensors 格式、targets 与 ignore 元数据 | AutoRound 导出 | 已有导出路径；完整服务契约尚待验证 |
| 原生 FP8/FP4 反量化与有意保留的全精度层 | AutoRound 加载/导出 | 已有加载器回退；服务前需审计残留原始辅助张量 |
| vLLM 融合分区的独立尺度 | vLLM 运行时 | 后端未保留时必须支持 |
| 分组 NVFP4 wo_a、TP 本地分组、输出 RoPE/后端分派 | vLLM 运行时 | 需要兼容的运行时支持 |
| 融合专家尺度兼容性校验 | 导出验证与运行时加载器 | 不支持的独立尺度必须拒绝，不能静默合并 |

旧评测适配器在运行时执行名称修补、配置重建、原生 FP8 检查/反量化和内核分派。
它是集成原型，不是预期的上游接口。新导出不应要求服务启动时扫描原始检查点，
也不应依赖私有的 `local_deepseek_v4_nvfp4_adapter` 标记。
导出问题应在 AutoRound 修复，而不是交给 vLLM 模型权重加载器兜底。

仅修改权重名称不能消除运行时计算要求：保留 gate/up 或其他融合分区的独立尺度、
执行 grouped GEMM 都属于数值/计算语义。不能为避免运行时支持而重新舍入/量化权重，
或把不同全局尺度替换成共同最大值。推理引擎内部的共享专家名称可能不同于 Hub，
这类内部映射应由引擎正常的原始检查点加载器处理。

宣称最小改动服务兼容前，还需验证新导出的标准 compressed-tensors 配置与全部排除层，
确认不需要原生 FP8 修补，再对精简后的运行时适配执行 TP2 加载、分组数值测试、
语义生成和精度验证。当前 CPU 测试与旧集成检查点的评测结果不证明迁移已完整完成。