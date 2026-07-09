# SYCL TLA Commit Benchmark 对比

这份说明对比了原始坏 build、较早的好上游点，以及本地 no-LSE sparse epilogue 修复后的 sparse `HND` benchmark 表现。

## Build

- `auto_round_kernel/xbuild-0630-fedbb`
  基于坏上游点 `fedbba40`
- `auto_round_kernel/xbuild-25ace5-0406`
  基于好上游点 `25ace5`
- `auto_round_kernel/xbuild-0630-fedbb-nolse`
  同样基于 `fedbba40`，但把 sparse 路径改成了本地 no-LSE epilogue

## 范围

通用 benchmark 配置：

- layout: `HND`
- `q_tile_override=256`
- `sparse_q_block_tokens=256`
- `sparse_k_block_tokens=64`
- device: `ZE_AFFINITY_MASK=1`

覆盖范围：

- `xbuild-0630-fedbb` 和 `xbuild-25ace5-0406`
  对比了 `topk=0.5`、`0.3`、`0.1`
- `xbuild-0630-fedbb-nolse`
  只对目标 case `topk=0.5` 做了 spot check

## Loader 说明

`test/bench_sparse_topk.py` 现在支持：

- `--xbuild-dir`
- `--xpu-so`

因此 benchmark 运行时可以显式指定要测的 `.so`，不会再默默回退到默认 build。

## 基线结果

### xbuild-0630-fedbb

- dense torch: `2110.954 ms`
- dense sage: `1213.869 ms`
- `topk=0.5`: kernel `1812.282 ms`, e2e `2015.860 ms`
- `topk=0.3`: kernel `1094.826 ms`, e2e `1296.630 ms`
- `topk=0.1`: kernel `376.031 ms`, e2e `576.689 ms`

CSV：

- [bench_sparse_topk_xbuild-0630-fedbb_hnd_gpu1.csv](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/bench_sparse_topk_xbuild-0630-fedbb_hnd_gpu1.csv)

### xbuild-25ace5-0406

- dense torch: `2111.008 ms`
- dense sage: `1218.095 ms`
- `topk=0.5`: kernel `637.953 ms`, e2e `841.896 ms`
- `topk=0.3`: kernel `394.873 ms`, e2e `598.060 ms`
- `topk=0.1`: kernel `145.836 ms`, e2e `348.595 ms`

CSV：

- [bench_sparse_topk_xbuild-25ace5-0406_hnd_gpu1_rerun.csv](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/bench_sparse_topk_xbuild-25ace5-0406_hnd_gpu1_rerun.csv)

## 修复后抽样结果

### xbuild-0630-fedbb-nolse

目标 case：

- layout: `HND`
- `topk=0.5`
- `q_tile_override=256`
- `sparse_q_block_tokens=256`
- `sparse_k_block_tokens=64`

结果：

- dense torch: `2111.371 ms`
- dense sage: `1213.433 ms`
- sparse kernel-only: `643.092 ms`
- sparse e2e: `847.798 ms`

CSV：

- [bench_sparse_topk_xbuild-0630-fedbb-nolse_hnd_topk05_gpu1.csv](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/bench_sparse_topk_xbuild-0630-fedbb-nolse_hnd_topk05_gpu1.csv)

## 差异

对目标 `topk=0.5` case：

- 坏 `0630`：kernel `1812.282 ms`，e2e `2015.860 ms`
- 修复后 `0630-nolse`：kernel `643.092 ms`，e2e `847.798 ms`
- 好 `25ace5`：kernel `637.953 ms`，e2e `841.896 ms`

解释：

- 本地 no-LSE sparse epilogue 基本恢复了丢失的稀疏性能
- 修复后的 `fedbba40` build 已经非常接近 `25ace5`
- 剩余差距很小，原先的大回退已经不再是主要问题

## Unitrace Spill 检查

主要 profile 的目标 case 是：

- layout: `HND`
- `topk=0.5`
- `q_tile_override=256`
- `sparse_q_block_tokens=256`
- `sparse_k_block_tokens=64`
- `warmup=0`
- `iters=1`

Spill 汇总：

| Build | Sparse Epilogue | Sparse Private Mem / Thread | Sparse Spill / Thread | Dense Spill / Thread | Sparse Register File / Thread |
|---|---|---:|---:|---:|---:|
| `xbuild-0630-fedbb` | `FMHAFwdEpilogue` | `2048 B` | `28992 B` | `128 B` | `256` |
| `xbuild-0630-fedbb-nolse` | `SparseFMHAFwdEpilogue` | `2048 B` | `640 B` | `128 B` | `256` |
| `xbuild-25ace5-0406` | 旧的无状态形态 | `2048 B` | `640 B` | `128 B` | `256` |

解释：

- 坏 `0630` build 出现了明显的 sparse-only spill 爆炸
- 改成 no-LSE sparse epilogue 后，sparse spill 回到和 `25ace5` 一样的 `640 B/thread`
- dense spill 保持不变，说明影响局限在 sparse 路径

相关 unitrace 日志：

- [unitrace_0630_hnd_topk05_d0.1254552](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/unitrace_0630_hnd_topk05_d0.1254552)
- [unitrace_nolse_hnd_topk05.2005344](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/unitrace_nolse_hnd_topk05.2005344)
- [unitrace_25ace5_hnd_topk05_d0.1262743](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/unitrace_25ace5_hnd_topk05_d0.1262743)

## 更低层的检查

为了确认这确实是 epilogue / codegen 问题，而不只是 benchmark 层面的现象，我们又往下检查了一层编译后的 kernel。

### 上游 Epilogue 形态

- `25ace5` 的 epilogue 是无状态的：
  [xe_fmha_fwd_epilogue.hpp](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/xbuild-25ace5-0406/_deps/sycl_tla-src/applications/flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp:106)
- `fedbba40` 的 epilogue 增加了 `lse_ptr`、`seq_len_qo`、`num_heads_q`，并且为了可选 LSE 写回多了 `head_q` / `idx_b` 输入：
  [xe_fmha_fwd_epilogue.hpp](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/xbuild-0630-fedbb/_deps/sycl_tla-src/applications/flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp:106)
  [xe_fmha_fwd_epilogue.hpp](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/xbuild-0630-fedbb/_deps/sycl_tla-src/applications/flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp:156)
- 本地 sparse 修复把它恢复成 stateless 的 sparse-only epilogue：
  [xe_sparse_fmha_fwd_epilogue.hpp](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/wrapper/include/stla/xe_sparse_fmha_fwd_epilogue.hpp:71)

### 嵌入式 SPIR-V 检查

我们把每个 `.so` 里的 embedded SPIR-V bundle 抽出来，直接比较里面的类型名。

| Build | SPIR-V Bundle 大小 | `SparseFMHAFwdEpilogue` 次数 | `FMHAFwdEpilogue` 次数 | `XeSparseSageFwdKernel` 次数 |
|---|---:|---:|---:|---:|
| `xbuild-0630-fedbb` | `11010120` | `0` | `756` | `148` |
| `xbuild-0630-fedbb-nolse` | `10946344` | `196` | `752` | `148` |

可以这样解读：

- 两个 build 里仍然是同一批 sparse kernel 家族
- 只有修复后的 build，才真正包含 `SparseFMHAFwdEpilogue` 的 sparse kernel 实例
- 坏 build 的 sparse 仍然绑在 dense `FMHAFwdEpilogue` 上

这说明 device image 的变化方向和根因判断完全一致。

### 运行时计数器的限制

再往下一步，本来应该做采样式硬件计数器对比，但这台机器上目前还拿不到：

- `/proc/sys/dev/xe/observation_paranoid` 当前是 `1`
- `unitrace -d -k -g ComputeBasic ...` 会报 `Unable to open metric streamer for sampling`
- 生成出来的 metrics CSV 目前只有表头

所以目前最强的已确认证据链是：

- 上游 epilogue 源码形态变了
- sparse device image specialization 也变了
- 只有坏 build 的 sparse spill 爆炸
- 恢复 stateless sparse epilogue 后，spill 爆炸消失，性能恢复

## 结论

- 对 sparse `HND` 来说，`fedbba40` 原始形态就是坏性能 case
- 回退来自 sparse kernel 的 codegen 形态，而不是 dense 路径
- 把 sparse 改回 no-LSE epilogue 后，spill 爆炸消失
- 对目标 `topk=0.5` benchmark，修复后的 `fedbba40` 已经基本恢复到 `25ace5` 水平
