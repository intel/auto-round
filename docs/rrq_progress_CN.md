# RRQ 实现进度（对照 RFC 分阶段目标）

- 更新时间：2026-09-04
- 分支：`feat/rrq-phase1`
- 对照文档：[`rrq_rfc_CN.md`](./rrq_rfc_CN.md) / [`rrq_rfc.md`](./rrq_rfc.md)（英文版）
- 状态来源：`ar-xpu` conda 环境（torch 2.14.0+xpu + 2× Arc Pro B60）内**实际运行验证**（单测 + Qwen3-0.6B 端到端 + lm-eval）
- 本次更新：**Phase 2 完成** —— `generate_rrq_residual`（从已有 INT2 base 模型 + 原始 FP 权重增量生成 residual，无需重算 base）；Phase 1 端到端运行时验证全部通过

## 变更清单

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `auto_round/__init__.py` | 修改 | 导出 `RRQConfig`；PEP 562 `__getattr__` 懒导出 `load_rrq_model` + `generate_rrq_residual` |
| `auto_round/algorithms/registry.py` | 修改 | 注册 `rrq` alias 与 pipeline member（config + quantizer） |
| `auto_round/algorithms/quantization/rrq/` | 新增 | `config.py` / `quantizer.py` / `__init__.py` |
| `auto_round/inference/rrq_linear.py` | 新增 | `RRQLinear` = base + N 个 residual `QuantLinear` 子模块；`forward` 先算 base 再累加 residual；`set_rrq_bits` |
| `auto_round/inference/rrq_model.py` | 新增 | **组合加载入口** `load_rrq_model`（从 base packed tensors 重建 base `QuantLinear` + residual `QuantLinear` → `RRQLinear`） |
| `auto_round/export/export_to_autoround/export_to_rrq.py` | 新增 | **residual 导出** `save_quantized_rrq`（只落 3 个 packed INT2 plane 到单个 `auto_round:rrq` artifact）/ `save_rrq_base_model` / **Phase 2 `generate_rrq_residual`**（已有 base + FP 权重增量生成）+ `quantization_config` 构建 |
| `auto_round/export/formats/backends/rrq.py` | 新增 | `RRQFormat`（注册 `auto_round:rrq`） |
| `auto_round/export/formats/backends/__init__.py` | 修改 | 导入并导出 `RRQFormat` |
| `auto_round/compressors/model_free.py` | 修改 | `auto_round:rrq` 加入 accepted_formats |
| `auto_round/inference/backend.py` | 修改 | 新增 `RRQ_FORMAT` 格式常量 |
| `auto_round/utils/common.py` | 修改 | `auto_round:rrq` 加入 `SUPPORTED_FORMATS` |
| `auto_round_extension/torch/qlinear_torch.py` | 修改 | asym `pack_248_bits` `self.device` → `device` 参数（bug 修复） |
| `auto_round/export/export_to_gguf/conversion/base.py` | 修改 | `auto-round-rrq` → fail-fast 拒绝 |
| `auto_round/export/export_to_mlx/export.py` | 修改 | `auto-round-rrq` → fail-fast 拒绝 |
| `docs/rrq_rfc_CN.md` / `rrq_rfc.md` / `rrq_progress_CN.md` | 新增 | RFC 草案 + 进度 |
| `test/unit/test_cpu/algorithms/test_rrq.py` | 新增 | 单元测试（Phase 1 量化/导出/加载 + Phase 2 `generate_rrq_residual`），**28 个** |
| `test_rrq_qwen3_06b.py` / `test_rrq_lm_eval.py` | 新增 | Qwen3-0.6B 端到端量化+校验 / lm-eval 精度 benchmark 脚本 |

**整体判断**：Phase 1 + Phase 2 核心算法、**导出/加载管线集成、ABI、组合 loader、非 RRQ 后端拒绝、packed-INT2 存储、增量生成** 均已落地，并在 `ar-xpu`（torch 2.14.0+xpu）环境**实跑验证通过**（28 单测 + Qwen3-0.6B 端到端 + HellaSwag 精度）。剩余：Phase 3（OPT sign-SGD 调优）。

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

1. ~~**端到端运行时验证**~~ ✅ **已完成**：`ar-xpu` 环境（torch 2.14.0+xpu）内实跑通过。`test_rrq.py` 28/28 单测通过；`test_rrq_qwen3_06b.py` 对 Qwen3-0.6B 完成量化 + base/residual 分块保存 + 文件布局/大小校验（PASS）；`load_rrq_model` 真实 HF 加载 + 2/4/6/8-bit forward 均正常（`--verify-load`）。
2. ~~**base 加载路径**~~ ✅ **已完成**：`load_rrq_model` 在真实 Qwen3-0.6B 上验证：`from_pretrained` 加载架构 + 非量化权重，packed 层被手动重建为 `QuantLinear` 后替换（garbage 权重被丢弃），`set_module` 替换后 model 正常 forward + `generate` 可用。
3. ~~**Phase 2（`generate_rrq_residual`）**~~ ✅ **已完成**：基于新 `auto_round:rrq` 布局实现增量生成（`export_to_rrq.py::generate_rrq_residual`）。流程：dequant base（`QuantLinear.forward(identity)`）→ `E_1 = W_fp − Ŵ_0` → 3 轮 RTN INT2 → packed 导出。支持本地目录 / HF 模型名；`group_size`/`sym`/`bits` 与 base 校验一致（fail-fast）；`from auto_round import generate_rrq_residual` 已暴露；5 个单测（结构/残差单调递减/配置 fail-fast/顶层导出）全部通过。
4. **Phase 3（OPT sign-SGD）**：⏳ 未实现（`RRQSignRoundQuantizer`、`iters/lr/minmax_lr` 字段、每轮 sign-SGD 调优）。

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

- **Phase 1**：核心算法 + **packed-INT2 residual 存储**（3 个 INT2 AutoRound model 打包到单个 `auto_round:rrq`）+ 导出/加载管线集成 + ABI（`quant_method="auto-round-rrq"`、`quantization_config`）+ 组合 loader（复用 W2A16 `QuantLinear.forward`）+ 非 RRQ 后端拒绝 + 单元测试 **均已落地并通过运行时验证**。Qwen3-0.6B 端到端验证通过（量化 + 分块保存 + `load_rrq_model` + 2/4/6/8-bit forward + HellaSwag 精度 6-bit≈fp32）。**✅ 已完成**。
- **Phase 2**：✅ **已完成**。`generate_rrq_residual` 从已有 INT2 base 模型 + 原始 FP 权重增量生成 residual（无需重算 base），5 个单测通过，`from auto_round import generate_rrq_residual` 已暴露。
- **Phase 3**：⏳ 未实现（OPT sign-SGD 调优）。
- **验收标准**：单元层面 + `load_rrq_model` 真实加载 + 2/4/6/8-bit 输出 + HellaSwag 精度均已实跑验证（`ar-xpu` 环境）。Phase 3 验收待 OPT sign-SGD 落地后补充。

---

## 五、建议的补全优先级

1. ✅ 修 `get_quant_func` 签名、补测试 import、核对 4-bit 语义、清理冗余 import（已完成）。
2. ✅ 打通主管线：注册 `OutputFormat`（`auto_round:rrq`）+ `quantization_config`（`auto-round-rrq`）+ accepted_formats + `RRQ_FORMAT` + 懒导出 `load_rrq_model`（已完成）。
3. ✅ 组合 loader（base+residual → `RRQLinear`，复用 W2A16 `QuantLinear`）（已完成）。
4. ✅ 非 RRQ 后端（GGUF/MLX）对 `auto-round-rrq` fail-fast（已完成）。
5. ✅ **Packed-INT2 存储**：quantizer 用 W2A16 `QuantLinear.pack` 打包 residual（sym/asym 各选类），导出只落 3 个 packed plane 到单个 `auto_round:rrq`，`RRQLinear`/load 复用 stock `QuantLinear.forward`（已完成，待运行时验证）。
6. ✅ **端到端验证**：`ar-xpu` 环境（torch 2.14.0+xpu）实跑通过。`test_rrq.py` 28/28；`test_rrq_qwen3_06b.py` 量化 + base/residual 分块保存 + `load_rrq_model` + 2/4/6/8-bit forward；HellaSwag 精度 6-bit≈fp32。
7. ✅ Phase 2 `generate_rrq_residual`（基于新 `auto_round:rrq` 布局，从已有 base + FP 权重增量生成）。
8. ⏳ Phase 3 OPT sign-SGD 调优（`RRQSignRoundQuantizer` + `iters/lr/minmax_lr`）。
