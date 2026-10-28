# 量化 Recipe

Recipe 是 Python API 与 CLI 共用的推荐配置来源。不传 `recipe` 等价于选择 `ar`。
Recipe 提供完整算法组合、校准默认值和可选的逐层推荐配置。
Scheme 定义和算法实现保持独立。

## 内置方案

| Recipe | 初始算法组合 | 迭代次数 | 样本数 |
| --- | --- | --- | --- |
| `ar` | SignRound | 200 | 128 |
| `ar-light` | SignRound，学习率 0.005 | 50 | 128 |
| `ar-best` | SignRound | 1000 | 512 |
| `rtn` | 普通 RTN | — | 1 |
| `opt-rtn` | 优化 RTN | — | 128 |
| `awq` | AWQ 平滑 → 普通 RTN | — | 128 |

所有初始方案的序列长度为 2048，batch size 为 8。`default`、`light`、`best`、
`opt_rtn` 分别是 `ar`、`ar-light`、`ar-best`、`opt-rtn` 的别名。
三个 `ar` 档位表达成本与效果目标，未来可增加经过验证的 scheme 专属推荐组合，
但不保证每个模型上的精度严格按档位排序。核心校准要求仍然适用，例如 RTN
在需要时仍要进行静态激活校准。

## 使用与覆盖

```python
from auto_round import AutoRound, SignRoundConfig

compressor = AutoRound("Qwen/Qwen3-0.6B", scheme="MXFP4", recipe="ar-best")

# 显式值优先，即使它等于旧 API 默认值。
compressor = AutoRound("Qwen/Qwen3-0.6B", recipe="ar-best", nsamples=128)

# 整体替换算法组合，同时继承 ar-best 的校准默认值。
compressor = AutoRound("Qwen/Qwen3-0.6B", recipe="ar-best", alg_configs=SignRoundConfig(iters=20))
```

```bash
auto_round Qwen/Qwen3-0.6B --scheme MXFP4 --recipe ar-best
auto_round Qwen/Qwen3-0.6B --recipe ar-best --nsamples 128 --iters 20
auto_round Qwen/Qwen3-0.6B --recipe ar-best --algorithm awq
auto_round list recipes
auto_round list recipe awq
```

显式算法配置整体替换 recipe pipeline，不自动追加。其算法参数不会被 recipe
阶段参数覆盖。旧 Python `algorithm=` 和 CLI `--algorithm` 保留已有算法选择语义。
显式 `iters=0` 仍选择 RTN。显式逐层配置按模式、按字段覆盖 recipe 配置；
字符串 scheme 值整体替换对应条目。无论如何覆盖，能力校验仍会拒绝不支持的组合。

`auto_round_best`、`auto_round_light`、`auto_round_rtn`、`auto_round_opt_rtn`
选择对应 recipe；连字符形式行为一致。显式 `--recipe` 优先于快捷命令的默认选择。

## 检查与扩展

```python
from auto_round.recipes import get_recipe, list_recipes, resolve_recipe

print(list_recipes())
definition = get_recipe("ar-best")
plan = resolve_recipe(definition)
print(plan.pipeline, plan.calibration, plan.applied_overlays)
configs = plan.build_configs()
```

定义和解析结果均为独立快照。每次 `build_configs()` 都构造新的算法配置。
构造后的 compressor 通过 `compressor.resolved_recipe` 暴露所选推荐配置；
它不替代显式覆盖和核心校验后的最终有效配置。

自定义 recipe 可以使用 `auto_round.recipes` 的 `Recipe`、`AlgorithmSpec`、
`RecipeOverlay`、`RecipeContext`。可以直接把 `Recipe` 传给 `AutoRound`，
或者通过 `register_recipe` 注册后在 Python 和同进程 CLI 中按名称使用。
注册拒绝重复名称和别名。Overlay 按声明顺序执行，后匹配项覆盖前匹配项。
Overlay 的 `pipeline` 整体替换算法组合；`stage_params` 按稳定阶段名称修改参数，
并拒绝不存在的阶段。校准字段为 `dataset`、`seqlen`、`nsamples`、`batch_size`。

公共 resolver 接受调用者提供的模型元信息，不加载模型。入口从已加载模型
提取元信息；存在 overlay 时才为字符串模型输入查询配置元信息。
配置查询不可用时，上下文不包含模型元信息。已有模型逐层策略仍在加载模型后
执行，Python 和 CLI 均适用。其定义位于 `auto_round/recipes/layer_policies.py`，
原入口保留兼容适配。Shape 限制和模型兼容补丁不归 recipe 管理。

未来 `ar-auto` 可以通过这一解析边界扩展上下文并选择推荐方案，目前未注册该入口。
AutoScheme 仍是独立的混合位宽选择功能。
