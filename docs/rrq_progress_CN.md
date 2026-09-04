# RRQ 实现进度（对照 RFC 分阶段目标）

- 更新时间：2026-09-03
- 分支：`feat/rrq-phase1`
- 对照文档：[`rrq_rfc_CN.md`](./rrq_rfc_CN.md) / [`rrq_rfc.md`](./rrq_rfc.md)（英文版）
- 状态来源：基于当前未提交 diff 与新增文件的静态审查 + 语法校验（本环境无 `torch`/`pytest`，**未实际运行验证**）
- 本次更新：**residual 改为 packed-INT2 存储**（3 个 INT2 AutoRound model 打包到一个 `auto_round:rrq` artifact），forward/dequant 复用既有 W2A16 代码

## 变更清单（未提交）

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `auto_round/__init__.py` | 修改 | 导出 `RRQConfig`；PEP 562 `__getattr__` 懒导出 `load_rrq_model` |
| `auto_round/algorithms/registry.py` | 修改 | 注册 `rrq` alias 与 pipeline member（config + quantizer） |
| `auto_round/algorithms/quantization/rrq/` | 新增 | `config.py` / `quantizer.py` / `__init__.py` |
| `auto_round/inference/rrq_linear.py` | 新增 | `RRQLinear` = base + N 个 residual `QuantLinear` 子模块；`forward` 先算 base 再累加 residual；`set_rrq_bits` |
| `auto_round/inference/rrq_model.py` | 新增 | **组合加载入口** `load_rrq_model`（从 base packed tensors 重建 base `QuantLinear` + residual `QuantLinear` → `RRQLinear`） |
| `auto_round/export/export_to_autoround/export_to_rrq.py` | 新增 | **residual 导出** `save_quantized_rrq`（只落 3 个 packed INT2 plane 到单个 `auto_round:rrq` artifact）/ `save_rrq_base_model` + `quantization_config` 构建 |
| `auto_round/export/formats/backends/rrq.py` | 新增 | `RRQFormat`（注册 `auto_round:rrq`） |
| `auto_round/export/formats/backends/__init__.py` | 修改 | 导入并导出 `RRQFormat` |
| `auto_round/compressors/model_free.py` | 修改 | `auto_round:rrq` 加入 accepted_formats |
| `auto_round/inference/backend.py` | 修改 | 新增 `RRQ_FORMAT` 格式常量 |
| `auto_round/export/export_to_gguf/conversion/base.py` | 修改 | `auto-round-rrq` → fail-fast 拒绝 |
| `auto_round/export/export_to_mlx/export.py` | 修改 | `auto-round-rrq` → fail-fast 拒绝 |
| `docs/rrq_rfc_CN.md` | 新增 | RFC 草案 |
| `test/unit/test_cpu/algorithms/test_rrq.py` | 新增 | Phase 1 单元测试（量化 + 导出配置/重命名/校验） |

**整体判断**：Phase 1 核心算法 + **导出/加载管线集成、ABI、组合 loader、非 RRQ 后端拒绝** 均已落地，且 residual 已实现 **packed-INT2 存储**。剩余主要缺口是端到端运行时验证（本环境无 `torch`，未实跑）。整体交付约 **85–90%**。

---

## 关键决策：residual 采用 packed-INT2（3 个 INT2 AutoRound model 打包到 `auto_round:rrq`）

RFC 要求 residual 用 **packed INT2**（`qweight_k`/`scales_k`/`qzeros_k`）。实现时关键点：现有导出栈对 **对称**（`qlinear_torch_zp`）与 **非对称**（`qlinear_torch`）使用**不同的 QuantLinear 类**，两者的零点对/反量化约定不同（对称存 `zp-1`、dequant `zeros += 1`；非对称直接存 `zp`），因此 **load 时必须按 `sym` 选择同一个类**，不能统一用某一个。

处理方式（正确性 by construction，复用既有 W2A16 代码）：
- **quantizer 侧**：`RRQRTNQuantizer` 每个 residual plane 通过 W2A16 `QuantLinear.pack`（sym→`qlinear_torch_zp`，asym→`qlinear_torch`）打包成标准 INT2 layout。喂入的是 RTN 原始 `scale`/`zp`（已 reshape 成 `pack` 约定形状 `(out, num_groups)`），使 pack 自洽：`code = round(W/scale + zp)`，`forward` 还原 `scale*(code - zp)` —— 即该平面的反量化值（sym/asym 均精确）。base 平面同法存 `weight/scale/zp`（标准 INT2 layout），走标准 `auto_round` 路径导出。
- **导出侧**：`save_quantized_rrq` **只保存 3 个 residual plane** 的 packed 张量（`qweight_k/scales_k/qzeros_k`，k=1..3），打包进**单个** `auto_round:rrq` artifact（sharded safetensors），不含 base、不含非 RRQ 参数（embeddings/layernorm），保证 artifact 体积紧凑。residual 目录为独立 safetensors（仅含 `qweight_1..3` 等，不含 base）。
  > 注：`save_quantized_rrq` 不再走 `save_model`（那会把完整模型含 base 权重/非 RRQ 参数一起 dump 进 residual），改为 `_save_state_dict_sharded` + `_write_quantization_config` 只落 residual 张量 + `quantization_config`。
- **推理侧**：`RRQLinear` = base（`QuantLinear`）+ 若干 residual（`QuantLinear`）子模块；`forward` = **先算 base 结果，再依次累加每个 active residual 的结果**（bias 由 base 加一次），dequant 全部复用 stock `QuantLinear.forward`，无需自写 pack/dequant。
- **加载侧**：`load_rrq_model` 从 base 的 packed tensors（`qweight`/`scales`/`qzeros`/`bias`）重建 base `QuantLinear`（`from_pretrained` 无法重建 packed QuantLinear，故手动重建；架构仍由 `from_pretrained` 加载），residual plane 按 `qweight_k` 键重建；两侧都按 `sym` 选同一 `QuantLinear` 类，dequant 逐位一致。

> `quantization_config` ABI：`quant_method="auto-round-rrq"`、`format_version=1`、`base_bits=2`、`residual_planes=[2,2,2]`、`supported_effective_bits=[4,6,8]`、`total_planes`、`group_size`、`sym`、`packing_format="auto_round:rrq"`。

---

## 一、已完成（对照 Phase 1 清单）

| RFC 条目 | 状态 | 说明 |
| --- | --- | --- |
| 1. `RRQConfig` + 注册 `rrq` alias + 从 `__init__` 导出 | ✅ | `config.py` 继承 `RTNConfig`，固定 `bits=2 / data_type=int / act_bits=16 / num_residual_planes=3`，覆盖时抛 `ValueError`；`registry.py` 已加 alias `"rrq"/"rrq_rtn"` 与 pipeline member；`__init__.py` 已导出 `RRQConfig` |
| 2. `RRQRTNQuantizer` 4 轮顺序 RTN（`disable_opt_rtn=True`、无 calib） | ✅（packed） | `_quantize_layer_rrq` 实现 `W → 4 轮 QDQ 累加`；base 写入 `weight/scale/zp`（标准 INT2 layout），每个 residual plane 经 W2A16 `QuantLinear.pack`（sym/asym 各选类）打包成 `qweight_k/scales_k/qzeros_k` 存入 buffer |
| 3. 导出（residual + base 分开） | ✅（packed INT2） | `export_to_rrq.py`：`save_quantized_rrq` 只落 3 个 packed INT2 plane（`qweight_1..3/scales/qzeros`）到**单个** `auto_round:rrq` sharded artifact（不含 base/非 RRQ 参数），含 `quantization_config`（`quant_method="auto-round-rrq"`）；`save_rrq_base_model` 走标准 `auto_round` 路径。`RRQFormat` 已注册 `auto_round:rrq` |
| 4. `RRQLinear` 推理参考模块 + `set_rrq_bits` | ✅ | `rrq_linear.py`：`RRQLinear` = base `QuantLinear` + N 个 residual `QuantLinear` 子模块，`forward` 先算 base 再累加 active residual（dequant 复用 stock `QuantLinear.forward`）；`set_rrq_bits`(2/4/6/8 → 1/2/3/4 平面)；被 `rrq_model.py` 使用 |
| 5. 组合加载入口（base + residual → `RRQLinear`） | ✅ | `rrq_model.py::load_rrq_model`：校验 `bits/group_size/sym` 一致（fail-fast）；从 base packed tensors 重建 base `QuantLinear`、从 `qweight_k` 重建 residual `QuantLinear`（均按 `sym` 选类），将 eligible 层替换为 `RRQLinear`，按 `active_bits` 设置活跃 |
| 6. 非 RRQ 后端显式报错 | ✅ | `export_to_gguf` 与 `export_to_mlx` 对 `quant_method="auto-round-rrq"` raise `NotImplementedError`（fail-fast，不静默丢弃残差） |

**管线集成**：`auto_round:rrq` 已注册 `OutputFormat`、加入 `model_free.py` accepted_formats、`backend.py` 定义 `RRQ_FORMAT` 常量、`__init__.py` 懒导出 `load_rrq_model`。

---

## 二、剩余缺口 / 偏离 RFC

1. **端到端运行时验证**：本环境无 `torch`/`pytest`，所有改动仅为**语法校验 + 静态审查**，未实跑。`load_rrq_model` 的真实 HF 加载（含 base packed `QuantLinear` 手动重建）、`RRQLinear` 2/4/6/8-bit 输出正确性（`QuantLinear.forward` 累加）、`save_quantized_rrq` 的 packed 张量落盘 + `load_state_dict` 往返均需在有 `torch` 环境跑通。
2. **base 加载路径**：`load_rrq_model` 目前用 `from_pretrained` 加载架构 + 从 packed tensors 手动重建 base `QuantLinear`。在真实模型上需确认 `from_pretrained` 对 packed checkpoint 的状态加载行为（packed 层加载为 garbage 权重后被替换，非 packed 层正常加载），以及 `set_module` 替换后 model 可正常 forward。
3. **Phase 2（`generate_rrq_residual`）**：旧的 `algorithms/quantization/rrq/export.py`（含 `generate_rrq_residual`、旧的 `rrq_config.json`/`rrq_residual.safetensors` 布局）已**删除**（与新导出 ABI 冲突且为死代码）。Phase 2 增量生成需基于新 `auto_round:rrq` 布局重新实现（dequant base → `E_1 = W − Ŵ_0` → 3 轮 RTN → packed 导出）。
4. **Phase 3（OPT sign-SGD）**：完全未实现（`RRQSignRoundQuantizer`、`iters/lr/minmax_lr` 字段、每轮 sign-SGD 调优）。

---

## 三、已修复的关键缺陷（历次）

1. ~~`export.py` `get_quant_func` API 不匹配~~ → 改为 `quant_func, _ = get_quant_func(dtype, bits, sym, disable_opt_rtn=True, group_size, iters=0)`（该旧文件后续已删除）。
2. ~~`test_rrq.py` 缺 `quant_tensor_rtn_sym` import~~ → 已补。
3. ~~`test_4bit_better_than_2bit` 命名疑虑~~ → 核对确认语义正确（base+1 残差=4-bit）。
4. ~~测试重复 `import os`/`import sys`~~ → 已清理。
5. ~~packed-INT2 手搓 dequant 的约定风险~~ → 改由 **quantizer 侧用 W2A16 `QuantLinear.pack` 打包**（sym/asym 各自选类，喂 RTN scale/zp 自洽），load/forward 复用 stock `QuantLinear`，不手搓 dequant（见"关键决策"）。
6. ~~residual 导出误将完整模型（含 base/非 RRQ 参数）dump 进 artifact~~ → `save_quantized_rrq` 改为只落 3 个 packed residual plane + `quantization_config`（`_save_state_dict_sharded`/`_write_quantization_config`）。
7. ~~`RRQLinear` bias 重复（base `QuantLinear` 加一次 + `RRQLinear` 再加一次）~~ → base 平面自带 bias，`RRQLinear` 传 `bias=None` 不再重复加。
8. ~~residual 统一用 `qlinear_torch_zp` 的 asym 偏移~~ → load 侧按 `sym` 选 `QuantLinear` 类（与导出一致），避免 `zeros+=1` 偏移。

---

## 四、进度结论

- **Phase 1**：核心算法 + **packed-INT2 residual 存储**（3 个 INT2 AutoRound model 打包到单个 `auto_round:rrq`）+ 导出/加载管线集成 + ABI（`quant_method="auto-round-rrq"`、`quantization_config`）+ 组合 loader（复用 W2A16 `QuantLinear.forward`）+ 非 RRQ 后端拒绝 + 单元测试（配置/量化-`packed`/导出-`packed`/`RRQLinear`/校验）**均已落地**。剩余：端到端运行时验证 + base 加载路径在真实模型上确认。**整体交付约 85–90%**。
- **Phase 2**：旧实现已删除，需基于新 ABI 重新实现增量生成。
- **Phase 3**：完全未实现。
- **验收标准**：单元层面（配置拒绝、残差单调递减、4bit>2bit、marker 属性、`quantization_config` 字段、buffer 重命名、base/residual 校验 fail-fast）已覆盖；`bit-exact vs RTN W2A16`、`base 可被现有 runtime 独立加载`、`residual 缺 base 报错`、`2/4/6/8-bit 输出正确性` 需在有 `torch` 环境实跑验证。

---

## 五、建议的补全优先级

1. ✅ 修 `get_quant_func` 签名、补测试 import、核对 4-bit 语义、清理冗余 import（已完成）。
2. ✅ 打通主管线：注册 `OutputFormat`（`auto_round:rrq`）+ `quantization_config`（`auto-round-rrq`）+ accepted_formats + `RRQ_FORMAT` + 懒导出 `load_rrq_model`（已完成）。
3. ✅ 组合 loader（base+residual → `RRQLinear`，复用 W2A16 `QuantLinear`）（已完成）。
4. ✅ 非 RRQ 后端（GGUF/MLX）对 `auto-round-rrq` fail-fast（已完成）。
5. ✅ **Packed-INT2 存储**：quantizer 用 W2A16 `QuantLinear.pack` 打包 residual（sym/asym 各选类），导出只落 3 个 packed plane 到单个 `auto_round:rrq`，`RRQLinear`/load 复用 stock `QuantLinear.forward`（已完成，待运行时验证）。
6. ⏳ **端到端验证**：在有 `torch` 环境运行 `test_rrq.py` + 用真实 HF 模型验证 `load_rrq_model`（base packed 重建）与 2/4/6/8-bit 输出（当前阻塞：环境无 `torch`）。
7. ⏳ Phase 2 `generate_rrq_residual`（基于新 `auto_round:rrq` 布局）+ Phase 3 OPT sign-SGD。
