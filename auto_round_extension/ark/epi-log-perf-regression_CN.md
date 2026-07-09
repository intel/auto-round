# 稀疏 Epilogue 性能回退说明

这份说明解释了为什么 `HND` 稀疏回退最终定位到 epilogue 选择问题，并补充了一些面向不熟悉 CUTE / SYCL-TLA 模板代码读者的背景。

## 简短结论

是的，这次稀疏性能回退的根因是：稀疏内核错误地用了带 LSE 能力的新版 dense `FMHAFwdEpilogue`，而不是稀疏场景实际需要的 no-LSE epilogue。

重点不是 “LSE 运行时本身很贵”，而是 “epilogue 类型改变了最终生成的 kernel 形状”。在 CUTE 风格代码里，一个模板类型的变化，往往会同时改变寄存器生命周期、临时 fragment、kernel 参数和内联后的代码结构，即使某个运行时分支几乎不会走到。

## 这里的 Epilogue 是什么

这类 kernel 大致由两个编译期大组件组成：

- `CollectiveMainloop`
  负责分块加载 Q/K/V、执行 MMA、累积注意力结果。
- `CollectiveEpilogue`
  负责最后的归一化、fragment 重排以及把输出 tile 写回全局内存。

在 CUTE / CUTLASS 风格里，这两者都是 C++ 模板类型。kernel 是由这些类型实例化出来的，所以改 epilogue 并不是小范围 helper 函数替换，而是会真正生成一个不同的设备内核。

## 为什么 CUTE 下这件事很敏感

如果不常看 CUTE，可以用一个简单心智模型理解：

- shape 是类型
- layout 是类型
- copy 策略是类型
- mainloop 是类型
- epilogue 是类型
- 最终 GPU kernel 也是由这些类型组合出来的

所以“稀疏路径用了错误的 epilogue”并不只是“调用了另一个函数”，而是“整个稀疏 kernel 的编译期对象图变了”。

这会影响：

- 哪些值在 epilogue 阶段仍然存活
- `Params` 里是否需要携带更多状态
- 是否存在额外条件分支
- 编译器保留多少 fragment 临时变量
- 寄存器压力是否越过 spill 阈值

真正出问题的就是最后一点。

## 上游改了什么

在好的上游点 `25ace5` 和坏的上游点 `fedbba40` 之间，`FMHAFwdEpilogue` 的结构变了：

- `25ace5`
  对我们的用法来说，epilogue 基本是无状态的。
- `fedbba40`
  epilogue 增加了 LSE 相关状态，比如 `lse_ptr`、`seq_len_qo`、`num_heads_q`，并且包含可选的 LSE 写回逻辑。

对于 dense attention，这个改动本身是合理的，因为 dense kernel 可能确实需要 LSE 输出。

但对我们的 sparse prefill 路径来说并不合适，因为 sparse Sage 当前并不消费 LSE 输出。

## 为什么稀疏路径会回退

我们的 sparse wrapper 当时仍然实例化的是新版 dense epilogue：

- 从语义上看，sparse 并不使用 LSE
- 但从结构上看，kernel 仍然带着支持 LSE 的 epilogue 类型

这意味着编译器仍然要把更重的 epilogue 形态编进 sparse kernel。

而这个 sparse kernel 本来就已经比较接近寄存器压力上限。额外 epilogue 状态和代码路径把它推过了临界点，于是 spill 急剧增加。

## 为什么 Dense 没有一起变慢

Dense 仍然使用 `FMHAFwdEpilogue`，而且 dense 性能保持正常。

这并不和根因判断矛盾，反而是支持它的证据：

- dense kernel 的资源平衡和 sparse 不同
- dense kernel 能承受这个 epilogue 变化
- sparse kernel 在这个配置下对寄存器更敏感

所以问题不是“新 epilogue 一定慢”，而是“这个 dense epilogue 不适合当前 sparse 实例化”。

## 证据

### 修复前

坏 build：`xbuild-0630-fedbb`

- sparse kernel 类型里是 `FMHAFwdEpilogue`
- sparse spill：`28992 B/thread`
- dense spill：`128 B/thread`
- `topk=0.5` 稀疏 kernel-only 延迟约 `1812 ms`
- `topk=0.5` 稀疏 e2e 延迟约 `2016 ms`

### 修复后

修复 build：`xbuild-0630-fedbb-nolse`

- sparse kernel 类型里是 `SparseFMHAFwdEpilogue`
- sparse spill：`640 B/thread`
- dense spill：`128 B/thread`
- `topk=0.5` 稀疏 kernel-only 延迟 `643.092 ms`
- `topk=0.5` 稀疏 e2e 延迟 `847.798 ms`

### 怎样解读 Unitrace

关键信号不只是“性能变好了”，而是“生成出来的 sparse kernel 按预期变轻了”：

- kernel 名字里现在是 `SparseFMHAFwdEpilogue`
- spill 从 `28992` 降到 `640 B/thread`
- dense 路径保持不变

如果 epilogue 类型就是根因，这正是最符合预期的现象。

### 更底层的确认

我们还往下做了一层确认，不只是停留在 top-down benchmark。

#### 1. 源码层面的形态变化

好的旧上游点 `25ace5` 里的 epilogue 是无状态的：

- [auto_round_kernel/xbuild-25ace5-0406/_deps/sycl_tla-src/applications/flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/xbuild-25ace5-0406/_deps/sycl_tla-src/applications/flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp:106)
  `Arguments {}` 和 `Params {}` 为空，`operator()` 只接收 `(O, tArA, tA_max, tA_sum, blk_qv, thr_id)`。

坏上游点 `fedbba40` 里的 epilogue 已经不是无状态了：

- [auto_round_kernel/xbuild-0630-fedbb/_deps/sycl_tla-src/applications/flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/xbuild-0630-fedbb/_deps/sycl_tla-src/applications/flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp:106)
  `Arguments` / `Params` 新增了 `lse_ptr`、`seq_len_qo`、`num_heads_q`。
- [auto_round_kernel/xbuild-0630-fedbb/_deps/sycl_tla-src/applications/flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/xbuild-0630-fedbb/_deps/sycl_tla-src/applications/flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp:156)
  `operator()` 还新增了 `head_q` 和 `idx_b`，函数体里包含可选的 LSE 写回逻辑。

我们本地的 sparse-only epilogue 则把它恢复成无状态形态：

- [auto_round_kernel/wrapper/include/stla/xe_sparse_fmha_fwd_epilogue.hpp](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/wrapper/include/stla/xe_sparse_fmha_fwd_epilogue.hpp:71)
  `Arguments {}` 和 `Params {}` 再次为空。

这说明根因判断不只是“看起来像”。编译期 kernel 模板形态，确实在我们怀疑的位置发生了变化。

#### 2. 设备镜像层面的确认

我们把两个 `.so` 里的嵌入式 SPIR-V device image 抽出来直接对比了。

- 坏 `xbuild-0630-fedbb`：`/tmp/fedbb.bundle`，大小 `11010120`
- 修复后 `xbuild-0630-fedbb-nolse`：`/tmp/nolse.bundle`，大小 `10946344`
- 差值：修复后 build 少了 `63776` 字节

更关键的是，嵌入式 SPIR-V 里的类型名变化完全符合预期：

- 坏 bundle：
  `SparseFMHAFwdEpilogue` 计数 `0`
  `FMHAFwdEpilogue` 计数 `756`
  `XeSparseSageFwdKernel` 计数 `148`
- 修复后 bundle：
  `SparseFMHAFwdEpilogue` 计数 `196`
  `FMHAFwdEpilogue` 计数 `752`
  `XeSparseSageFwdKernel` 计数 `148`

可以这样理解：

- 两个 build 里仍然是同一批 sparse kernel 家族
- 只有修复后的 build，才真正包含 `SparseFMHAFwdEpilogue` 的 sparse kernel 实例
- 坏 build 里的 sparse kernel 实例，仍然绑定的是 dense `FMHAFwdEpilogue`

所以连设备镜像本身都能确认：sparse kernel 的 specialization 确实改了。这比只看运行时名字更强。

#### 3. 当前这台机器上还不能证明的部分

再往下一层，本来还可以继续比硬件计数器，例如：

- stall 百分比
- occupancy
- memory traffic
- stall reason sampling

但这一层目前被系统权限挡住了：

- `/proc/sys/dev/xe/observation_paranoid` 当前是 `1`
- `unitrace -d -k -g ComputeBasic ...` 会报：
  `Unable to open metric streamer for sampling`
- 生成出来的 metrics CSV 只有表头，没有有效采样行

因此目前这条证据链是：

- 源码层面的 epilogue 形态变了
- 嵌入式 device image 也对应变了
- 生成出来的 sparse kernel 名字也对应变了
- 只有坏 build 的 sparse spill 爆炸
- 把 sparse 恢复成 stateless no-LSE epilogue 后，spill 爆炸消失，性能恢复

在拿不到完整硬件采样计数器的前提下，这已经是相当强的低层确认，足以支撑 epilogue 根因判断。

## 我们改了什么

我们新增了一个仅供 sparse 使用的无状态 epilogue：

- [auto_round_kernel/wrapper/include/stla/xe_sparse_fmha_fwd_epilogue.hpp](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/wrapper/include/stla/xe_sparse_fmha_fwd_epilogue.hpp:1)

它的目的很明确：让 sparse kernel 保持旧版 no-LSE 的 codegen 形状。

然后把活跃的 sparse launch path 改成使用这个 epilogue：

- [auto_round_kernel/wrapper/include/sycl_tla_sdpa_sparse.hpp](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/wrapper/include/sycl_tla_sdpa_sparse.hpp:979)
- [auto_round_kernel/wrapper/include/sycl_tla_sdpa_sparse.hpp](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/wrapper/include/sycl_tla_sdpa_sparse.hpp:995)
- [auto_round_kernel/wrapper/include/sycl_tla_sdpa_sparse.hpp](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/wrapper/include/sycl_tla_sdpa_sparse.hpp:1010)

Dense 路径保持使用正常的 dense epilogue。

## 一个实用的 CUTE 经验

看这类模板代码时，可以记住一个简单规则：

- 只要某个类型参与 kernel 模板实例化，就把它看成 kernel ABI 和 code shape 的一部分

即使两个 epilogue 在高层功能上看起来很像，最终生成的 kernel 也可能差很多。

在 CUTE 代码里，下面这些因素往往比一个小的运行时分支更重要：

- `Arguments` / `Params` 里多了哪些字段
- 多了哪些 fragment 变换
- 是否存在可选输出路径
- shared memory staging 结构是否变化
- copy helper 类型是否变化

## 最终结论

这次稀疏性能回退的根因，是稀疏路径使用了带 LSE 能力的 dense epilogue 类型。

真正的修复不是“优化 LSE”，而是把 sparse 路径恢复成 no-LSE epilogue 类型，让编译器重新生成更轻的 sparse kernel。
