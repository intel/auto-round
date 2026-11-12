# Multi-GPU quantization guide

**Experimental:** Data parallelism (DP), tensor parallelism (TP),
quantization-aware training (QAT), and quantization-aware distillation (QAD)
are experimental; their APIs and supported combinations may change.

## Which mode should I use?

| Mode | When to use it | CLI | Python `parallel_config` |
| --- | --- | --- | --- |
| DP: data parallelism | A block fits on one GPU; increase calibration/tuning throughput | `--data_parallel 2` or `auto` | `ParallelConfig(data_parallel=2)` |
| TP: tensor parallelism | Split a layer's weights and computation across GPUs | `--tensor_parallel 2` | `ParallelConfig(tensor_parallel=2)` |
| PP: naive pipeline parallelism | Partition model layers/blocks across GPUs and execute stages sequentially | `--pipeline_parallel 2` | `ParallelConfig(pipeline_parallel=2)` |

Select one mode per run; hybrid DP/TP/PP is not supported. PP is **naive pipeline
parallelism**: layers are partitioned across GPUs and stages execute sequentially
in one process, without overlapping microbatch scheduling.
TP uses native Transformers sharding.

## CLI: select the parallel degree

```bash
# DP: split the global batch by default; no --parallel_batch split needed
CUDA_VISIBLE_DEVICES=0,1 auto_round quantize Qwen/Qwen3-0.6B --data_parallel 2

# TP: start two local workers automatically, without a separate torchrun command
CUDA_VISIBLE_DEVICES=0,1 auto_round quantize Qwen/Qwen3-0.6B --tensor_parallel 2

# PP: naive pipeline parallelism across two GPUs
CUDA_VISIBLE_DEVICES=0,1 auto_round quantize Qwen/Qwen3-0.6B --pipeline_parallel 2
```

Select devices with `CUDA_VISIBLE_DEVICES` before setting the degree. Counts refer
to **visible GPUs**: physical GPUs `0,2` appear inside the process as
`cuda:0,cuda:1`. `--tp_size` and `--pp_size` remain legacy aliases; prefer the names
in the table for new commands.

## Python: create an AutoRound object

DP and PP do not require a multi-process launcher. Put all parallel settings in
`parallel_config`; an equivalent dictionary is also accepted.

```python
from auto_round import AutoRound, ParallelConfig

compressor = AutoRound(
    "Qwen/Qwen3-0.6B",
    scheme="W4A16",
    parallel_config=ParallelConfig(data_parallel=2),
)
compressor.quantize_and_save("./quantized", format="auto_round")
```

For PP, replace the config with `ParallelConfig(pipeline_parallel=2)`.
Without `parallel_config`, existing single-device and `device_map` calls are unchanged.

### Object-oriented TP API

Every TP rank must participate in quantization and saving. Save this as `quantize_tp.py`:

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

### Automatic TP launch from Python (optional)

To avoid a separate `torchrun` command, use this **module-level convenience
function**. It is not an instance method: each worker internally constructs an
`AutoRound` object and calls that object's `quantize_and_save()` method.
The parent receives output folders only, not a distributed model object.

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

Run it with ordinary `python quantize_tp.py`; keep the main guard.
The model path and constructor configuration must support serialization.
Use the object-oriented API for preloaded models/custom tokenizers.
An existing torchrun environment is reused without launching more workers.

## Batches, memory and saving

- **DP splits the global batch by default**: with two GPUs and `batch_size=8`, each
  GPU processes 4 samples. `parallel_batch="split"` is the default and can be omitted.
  Only request `parallel_batch="per_replica"` if every GPU should process 8 samples;
  the effective global batch then becomes 16.
- DP keeps the complete block and tuning state on every GPU; it does not reduce
  a block's weight memory requirement. Iterative tuning requires a power-of-two
  replica count and an evenly split global batch. Blocks that fail runtime
  sample/batch eligibility checks tune serially with a reason logged.
  `auto` selects an eligible replica count.
- With object-oriented TP APIs, all ranks must call quantization/saving methods;
  only rank zero writes files. Export reconstructs a complete CPU model, so the
  writer needs sufficient host RAM.

## Before you start

- DP applies to supported block calibration/tuning. Do not combine it with torchrun,
  multi-device `device_map` (for iterative tuning), or `model_free=True`.
  LFQ and dictionary-shaped diffusion calibration pools are not supported by this DP path.
- TP requires a Transformers version exposing `DistributedConfig` and a model with
  a native TP plan. Do not supply a custom `device_map`. Automatic launch currently
  supports single-node CUDA only.
- TP supports adapted AutoRound, AutoRound v2, RTN, optimized RTN, QAD/QAT and limited
  AWQ combinations. AutoScheme, `torch.compile`, block-level LFQ/QAD losses,
  rotation and hybrid TP/DP/PP are not supported. MXFP4/NVFP4 shards must contain
  complete quantization groups; unsupported configurations fail explicitly.
- PP currently uses naive, sequential pipeline execution; QAD does not support this path.
