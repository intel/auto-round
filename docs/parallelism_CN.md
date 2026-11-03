# 多 GPU 量化指南

## 选哪种并行？

| 模式 | 什么时候用 | CLI | Python 的 `parallel_config` |
| --- | --- | --- | --- |
| DP 数据并行 | 单个 block 能放入单卡，希望提高校准和调优吞吐 | `--data_parallel 2` 或 `auto` | `ParallelConfig(data_parallel=2)` |
| TP 张量并行 | 希望把同一层的权重和计算分到多卡 | `--tensor_parallel 2` | `ParallelConfig(tensor_parallel=2)` |
| PP 朴素流水线并行 | 将模型的层或 block 划分到多卡，顺序执行各阶段 | `--pipeline_parallel 2` | `ParallelConfig(pipeline_parallel=2)` |

一次只选一种模式，暂不支持 DP、TP、PP 混合。PP 是**朴素流水线并行
（naive pipeline parallelism）**：将层划分到不同 GPU，在单进程中顺序执行
各阶段，尚无微批次重叠调度。TP 使用 Transformers 原生分片。

## 命令行：直接指定并行度

```bash
# DP：默认把全局 batch 分到各卡，无需额外的 --parallel_batch split
CUDA_VISIBLE_DEVICES=0,1 auto_round quantize Qwen/Qwen3-0.6B --data_parallel 2

# TP：自动启动两个本地 worker，无需手写 torchrun
CUDA_VISIBLE_DEVICES=0,1 auto_round quantize Qwen/Qwen3-0.6B --tensor_parallel 2

# PP：两卡朴素流水线并行
CUDA_VISIBLE_DEVICES=0,1 auto_round quantize Qwen/Qwen3-0.6B --pipeline_parallel 2
```

先用 `CUDA_VISIBLE_DEVICES` 选择设备，再设置并行度。数量对应**可见 GPU**；
例如可见设备是物理卡 `0,2`，进程内编号仍是 `cuda:0,cuda:1`。
`--tp_size`、`--pp_size` 是旧别名，新命令推荐使用上表的名字。

## Python：先创建 AutoRound 对象

DP 和 PP 不需要多进程启动器。所有并行设置都放在 `parallel_config` 中，
也可以传入等价字典。

```python
from auto_round import AutoRound, ParallelConfig

compressor = AutoRound(
    "Qwen/Qwen3-0.6B",
    scheme="W4A16",
    parallel_config=ParallelConfig(data_parallel=2),
)
compressor.quantize_and_save("./quantized", format="auto_round")
```

使用 PP 时，将配置替换为 `ParallelConfig(pipeline_parallel=2)`。
未设置 `parallel_config` 时，原有单卡和 `device_map` 用法保持不变。

### TP 的对象式 API

TP 要求每个 rank 都参与量化和保存。将下面的代码存为 `quantize_tp.py`：

```python
from auto_round import AutoRound, ParallelConfig

if __name__ == "__main__":
    compressor = AutoRound(
        "Qwen/Qwen3-0.6B",
        scheme="W4A16",
        parallel_config=ParallelConfig(tensor_parallel=2),
    )
    compressor.quantize_and_save("./quantized", format="auto_round")
```

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2 quantize_tp.py
```

### Python 自动启动 TP（可选）

不想手写 `torchrun` 时，可使用下面的**模块级包装函数**。它不是对象方法：
它会在每个 worker 内部创建 `AutoRound` 对象，再调用该对象的 `quantize_and_save()`。
父进程只得到输出文件夹列表，不会得到分布式模型对象。

```python
from auto_round import ParallelConfig, quantize_and_save

if __name__ == "__main__":
    folders = quantize_and_save(
        "Qwen/Qwen3-0.6B",
        "./quantized",
        parallel_config=ParallelConfig(tensor_parallel=2),
        scheme="W4A16",
        save_kwargs={"format": "auto_round"},
    )
```

启动方式是普通的 `python quantize_tp.py`；必须保留 main 保护。
模型路径和构造配置需可序列化。预加载模型或自定义 tokenizer 请使用对象式 API。
已有 torchrun 环境会被复用，不会再次启动 worker。

## Batch、显存和保存

- **DP 默认拆分全局 batch**：两张卡、`batch_size=8` 时每卡处理 4 个样本。
  `parallel_batch="split"` 是默认值，不必填写。只有想让每卡都处理 8 个样本时，
  才显式使用 `parallel_batch="per_replica"`；此时全局 batch 变为 16。
- DP 每卡持有完整 block 和调优状态，不会降低单个 block 的权重显存需求。
  有迭代调优时，副本数必须为 2 的幂，全局 batch 要能均分；不满足运行时
  样本／batch 条件的 block 会记录原因并串行调优。`auto` 会选择可用的副本数。
- TP 对象式 API 的量化／保存方法必须在所有 rank 上调用，只有 rank 0 写文件。
  导出会重建完整 CPU 模型，因此写入 rank 需要足够的主机内存。

## 使用前确认

- DP 适用于已支持的逐块校准／调优；不要与 torchrun、多卡 `device_map`
  （有迭代时）或 `model_free=True` 混用。LFQ 和字典式 diffusion 校准池不支持该 DP 路径。
- TP 需要提供 `DistributedConfig` 和原生 TP plan 的 Transformers 版本与模型；
  不要再指定自定义 `device_map`。自动启动目前只支持单节点 CUDA。
- TP 支持已适配的 AutoRound、AutoRound v2、RTN、优化 RTN、QAD/QAT 及受限的 AWQ 组合；
  不支持 AutoScheme、`torch.compile`、块级 LFQ/QAD 损失、旋转及 TP/DP/PP 混合。
  MXFP4/NVFP4 要求分片内包含完整量化组，不支持的配置会明确报错。
- PP 目前采用朴素的顺序流水线执行；QAD 暂不支持该路径。
