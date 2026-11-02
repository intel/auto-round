# 统一并行执行

AutoRound 将张量并行（TP）交给 Transformers：由 Transformers 加载权重分片、
安装原生前向逻辑并完成布局重分布。AutoRound 仅适配本地权重和激活分片的
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

CLI 提供相同的配置维度。省略 `--tp_size` 时，默认使用 torchrun 的
`WORLD_SIZE`（未使用 torchrun 时为 `1`）；已选择其他并行维度时不自动推断 TP：

```bash
torchrun --nproc-per-node 4 -m auto_round quantize Qwen/Qwen3-0.6B --alg qad --tp_size 4 --scheme W4A16
torchrun --nproc-per-node 4 -m auto_round quantize Qwen/Qwen3-0.6B --alg auto_round --scheme W4A16 --group_size 32
torchrun --nproc-per-node 4 -m auto_round quantize Qwen/Qwen3-0.6B --alg awq,auto_round_v2 --scheme W4A16 --group_size 32
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
- 仅权重量化的 QAD/QAT、AutoRound/SignRound、AutoRound v2、RTN 和优化 RTN。
- 通过公共 QDQ 适配层支持 MXFP4/NVFP4 激活量化，具体约束见下文。
- INT AWQ 平滑后接 RTN、优化 RTN、AutoRound 或 AutoRound v2。
  暂不支持 AWQ clipping／clip-as-init 和 FP4 AWQ 平滑。
- 共享固定量化方案的有序多量化阶段，可在前面组合已支持的逐 block 预处理器。
  没有前置逐 block 阶段时，全模型量化器不能直接接在 block 预处理器后面。
- 一维 TP mesh，权重分片必须大小一致且非空。
- 正整数 group size。行切分的本地分片必须包含完整的量化组；
  `group_size=-1` 仅支持列切分权重。
- 梯度检查点及现有 QAD 优化器。暂不支持非量化参数联合训练、norm／bias
  调优、INT 激活量化、`torch.compile`、非标准 fused MoE 布局和 AutoScheme。
- 块级 LFQ/QAD 损失和旋转预处理器尚未实现对应 TP 契约，会明确报错，
  不会退回完整权重复制执行。

不支持的组合会明确报错，不会静默退回单卡或改变量化方案。
公共 QDQ 适配层可以复用，但算法必须声明并行能力后才能使用原生 TP。

## 统一算法接口

算法通过 `BaseAlgorithm.parallel` 访问由 `QuantizationRunContext` 注入的共享
`ParallelExecution`。单设备、既有 PP 放置和原生 TP 使用同一组服务，
调度不依赖算法名：

- `dispatch_block(...)` 返回 `BlockPlacement`，包含已放置的 block、loss 设备
  和主设备内存风险判断；保留 SignRound 既有的 PP 启发式。
- `prepare_gradients(...)` 为非 TP 执行保留既有实验性 DP 同步；
  TP 由原生 autograd 通信，不能再对 TP 分片套 DDP。
- `sample_indices(...)` 从 rank 0 广播 TP 微批次索引，不依赖各 rank 本地的随机数消耗；
  `offload_block(...)` 在每个 block 完成后将权重分片及量化张量元数据释放到 CPU。
- `replicated(...)` 在缓存／损失边界将分布式**激活**转换为普通张量，
  并保留反向传播；它不是权重聚合接口。
- `tensor_ops` 提供本地权重更新、通道向量裁切、分布式通道统计和 scale 折叠。
  AWQ 通过这些服务执行自己的权重数学逻辑。

`parallel_capabilities()` 默认拒绝 TP。仅使用公共服务的算法可以继承已经验证过的
实现契约；新的自定义数学实现必须显式声明支持并验证其布局。
重写调优、放置、校准或预处理 hook 后必须重新声明能力，不能静默信任继承的支持标记。
`validate_parallel_config(config, execution)` 用于在执行前拒绝不支持的选项。
能力校验同时覆盖预处理器和量化器。

自定义 QDQ wrapper 还需要在自己的类体声明 `supports_tensor_parallel=True`。
`tensor_parallel_wrapper(...)` 工厂将分片传输／原生 forward 与原有本地 wrapper
组合，不会用标准舍入替换算法自定义的权重逻辑。未适配的 wrapper 会明确报错。

## 算法阶段与执行策略

TP 使用与本地执行相同的有序算法流水线，不对特定算法组合做特殊处理。
每个预处理器和量化器都必须满足自身的并行能力及选项检查。例如，逐块
AutoRound 阶段可以后接另一个逐块阶段或全模型 QAD/QAT；加入尚未支持的成员
仍会在调优前报错。已有的多阶段限制继续生效，包括不支持即时保存／打包和断点恢复。

每个算法可以重写 `create_parallel_execution(config, default_execution)`，
返回基于已加载拓扑的专用 `ParallelExecution`。这样各算法可采用不同的 block
放置、流式调度和阶段局部 hook 策略，而无需在 composer 中按算法名分支。
`algorithm_stage(algorithm, model)` 是安装这些 hook 的上下文管理器，
应通过 `finally` 在正常结束及初始化／调优异常时移除 hook。
提前校验时也会选择策略，因此工厂不能修改模型或安装 hook。

Composer 进入当前成员的策略，只准备这些成员，执行阶段后依次完成成员清理和
策略退出，再启动下一阶段。预处理器仅在首个逐块阶段运行。校准、block forward
和 compressor 放置使用当前阶段的执行服务；阶段边界清空缓存的 block 放置状态，
异常时恢复原执行引用。如果后续算法需要原始 FP 权重，前序阶段保留这些权重，
并将分片本地的 warm-start 参数（包括优化版初始 scale）保存在 native 层上，
而不是临时本地视图上，以便跨阶段传递。

Transformers 原生 TP forward 和 DTensor placement 属于模型的通信／分片基础，
不是可随算法丢弃的 hook。阶段切换不会移除或重新应用这层基础，不会聚合完整权重，
也不会重建进程组。目前所有策略必须保留已加载的 `ParallelConfig`。
不同 TP/PP 度数、TP 切换到 PP，以及修改原生分片方案，都需要显式重分片权重、
元数据和 warm-start 状态并重建模型；当前阶段接口尚未实现这些功能。
选择不同拓扑的策略会明确报错，不会静默将本地 shard 当作未分片模型执行。

## 逐块 TP 与 CPU 驻留

逐块算法中，每个 rank 将非当前 block 的**本地权重分片**保留在 CPU 内存。
只有当前 block 的分片搬到该 rank 的加速器上执行校准、forward、backward 和优化，
完成后再搬回 CPU。校准前向也采用同样的逐块流式调度；block 之外的 embedding、
norm 和输出模块可在这次前向期间保持驻留。不会重建完整 block 权重。
各 rank 使用相同的样本和调优索引。通道统计可以归约／聚合，
但权重矩阵在整个调优过程中始终保持分片。

TP 使用上述主机分片驻留替代单进程磁盘 offloader。
即时打包／保存被禁用：调优结束后以 collective 方式导出，
只在写入 rank 上打包完整 CPU 模型。多进程运行时禁用跨设备搜索借卡，
防止某个 rank 使用另一个 rank 的加速器。
QAD/QAT 保留原来的全模型执行方式，不使用上述逐块调度。
这里改变的是校准／调优阶段的驻留，不改变 Transformers 原生加载过程：
初次加载仍可能先把全部本地分片放到加速器上，再 offload。
此执行服务尚未实现加载期的 CPU 流式方案。
逐块调优结束后，若要对返回的 TP 模型执行前向，全部 rank 需要先将分片搬回
各自的本地加速器；导出的模型则不带 TP。

离线双 rank 测试覆盖稠密 Llama 的逐块算法、AWQ 通道统计／scale 折叠、
导出 logits，以及调优期间禁止完整权重聚合。
这不代表已经验证逐块 MoE 精度或 CUDA/NCCL 显存行为。

## FP4 权重与激活量化

`MXFP4` 和 `NVFP4` 预设默认采用 W4A4，原生 TP 无需覆盖激活配置。
同时支持 `MXFP4_RCEIL`。如果仅量化权重，请显式设置
`act_bits=16, act_data_type="float"`。
必须同时设置两个字段，否则 NVFP4 中带位宽的激活类型会恢复 `act_bits=4`。
例如，将上文的方案和算法配置替换为：

```python
from auto_round.algorithms.quantization.qad.config import QADConfig

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

激活 QDQ 保留输入的原生 DTensor 布局。MXFP 激活使用大小为 32 的本地量化组，
支持 4、6 或 8 位（`mx_fp` / `mx_fp_rceil`）；NVFP4 激活使用大小为 16 的
量化组，支持 4 位（`nv_fp4` / `nv_fp4_with_static_gs`）。两者均要求
`act_sym=True` 和 `act_dynamic=True`；NVFP4 的静态 global scale 变体仍使用
动态 block scale。其他激活类型或静态 block scale 会明确报错。
行切分后的本地宽度也必须包含完整的激活量化组。

NVFP4 的校准最大值在权重的 TP mesh 内同步。如果没有校准或显式提供的
global scale，则每次前向通过标量 `MAX` 归约计算激活最大值，不会为计算此 scale
聚合完整激活或权重。可调优的激活裁剪系数保持复制，其分片局部梯度在 mesh 内
求和。解包后保留激活 QDQ；导出时重建不依赖 TP collective 的普通 CPU 激活包装层。

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
的行为，再运行 Qwen3-MoE 的 INT／MXFP4／NVFP4 仅权重调优、
MXFP4／NVFP4 W4A4 调优、CPU 导出
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

CLI 在仅由 rank 0 评估之前，会由全部 TP rank 关闭训练进程组，
并临时清除 torchrun／Accelerate 的分布式状态。
fake 和打包检查点评估都适用；Python API 不会自动销毁调用方持有的进程组。

离线双进程 CPU/Gloo 测试覆盖 FP4 前向及 STE 梯度一致性、QAD/QAT 生命周期、
普通导出模型的 logits 和真实 `auto_round` 打包结果。
激活测试包括稠密和 MoE W4A4、MXFP6/8 激活、零输入和空输入、并列全局最大值、
裁剪系数梯度及激活量化组边界。逐块测试还覆盖 AutoRound/SignRound、
AutoRound v2、RTN 和优化 RTN 的 FP4 激活量化，包括元数据 offload 和导出 logits。
CUDA/NCCL 显存、性能及部署后端推理仍未验证。

## PP 与后续 DP

`ParallelConfig(pp_size=N)` 将现有单进程设备放置路径配置到前 `N` 张可见
CUDA 设备。这不是微批次流水线训练调度，QAD 目前也不支持此路径。
不启用并行配置维度时，现有 `device_map` 调用方式保持不变。

`dp_size` 预留独立的数据并行维度；大于一的值及 TP／PP 混合组合目前
均抛出 `NotImplementedError`。本接口不替换现有实验性的 block 级 DDP 行为。
后续 DP 支持需要显式管理数据切分、副本进程组和梯度同步，
而不能将 TP 进程组当成 DP 进程组使用。
