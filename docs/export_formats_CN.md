# 输出格式与导出实现

所有输出格式实现统一位于 `auto_round/export/formats/`。旧 `export_to_*` 和
`formats/backends` 目录已删除，不保留导入兼容层；直接导入旧内部模块的代码需要更新。
高层 `AutoRound.save_quantized` API 与支持的格式字符串保持不变。

## 目录结构

```text
auto_round/export/
  base.py                       OutputFormat 接口与能力检查
  registry.py                   注册、别名、公开格式名称
  resolver.py                   格式选择与 scheme 解析
  common/
    packing.py                  模型准备、层遍历、公共打包工具
    artifacts.py                tokenizer 与 processor 保存
    serialization.py            checkpoint/分片写出及输出布局
  formats/
    gptq/                       format.py、packing.py、config.py
    awq/                        format.py、packing.py、config.py、utils.py
    autoround/                  format.py、integer.py、fp8.py、float.py、qlinear 模块
    compressed_tensors/         format.py、integer.py、float.py、static_fp.py、config.py
    gguf/                       format.py、writer.py、conversion 与 GGUF 工具
    mlx/                        format.py、packing.py、config.py
    svdquant/                   format.py、打包/导出工具与模型适配器
    fake.py
    fp8.py
```

每种格式拥有校验、打包实现、checkpoint 元数据及保存行为。复杂格式使用包，简单格式
可使用单个模块。公共工具复用确实相同的操作，不引入第二套路由系统。

## 新增格式

1. 在 `formats/` 下创建模块或包，继承 `OutputFormat`，用
   `@register_format("my_format")` 注册。
2. 在该格式内实现 `pack_layer` 和 `save_quantized`。专用打包与配置工具
   同样放在这个格式包内。
3. 在 `registry.load_builtin_formats` 中导入模块，使注册执行。

普通格式无需另建 exporter，也无需修改名称列表、resolver 或 compressor。内置格式
延迟加载，避免工具模块提前导入轻量 GGUF 配置时产生循环依赖。可选依赖在实际使用
它们的函数内部导入。

```python
from auto_round.export import OutputFormat, register_format


@register_format("my_format")
class MyFormat(OutputFormat):
    def pack_layer(self, layer_name, model, device=None, **kwargs):
        # 将目标层替换为本格式的打包表示。
        ...

    def save_quantized(
        self,
        output_dir,
        model=None,
        tokenizer=None,
        layer_config=None,
        inplace=True,
        device="cpu",
        serialization_dict=None,
        **kwargs,
    ):
        # 构建本格式元数据并写出 checkpoint。
        ...
        return model
```

`pack_layer` 原地修改目标层，返回值不参与后续处理。保存方法返回导出的模型。
`layer_config` 是逐层配置，`serialization_dict` 是导出元数据；compressor 通过关键字
传入 tokenizer、device 等选项。支持逐层打包的格式在完整保存时也使用同一实例方法。

通过 `check_scheme_args` 校验解析后的 scheme；仅提供 `support_schemes` 列表不会
校验 scheme 实例。SVDQuant 等整模型格式需要覆盖即时打包/保存能力。GGUF 保留流式
writer、converter 生命周期，通过 `immediate_pack` 提供额外上下文，不套用 Hugging Face
checkpoint writer。

## 公共操作与格式差异

- `prepare_model(model, inplace, device=...)` 在 `inplace=False` 时先复制后移动，
  保持调用者模型不变。
- `pack_layers(model, names, pack_layer, device=...)` 固定名称快照，跳过不可用的 meta
  张量，并复用格式的打包回调。
- `save_artifacts(...)` 仅在提供输出目录时保存支持的 tokenizer/processor。
- `save_model(...)` 保留分片、即时保存与 pipeline 支持。dtype 和配置差异由格式明确决定。

AutoRound 和 compressed-tensors 一起选择 packing/writer，消除 pack/save 独立维护的
重复分支。AutoRound 原生格式可以复用 GPTQ/AWQ 打包，但保留自己的元数据。
请求名称（`auto_round`）、解析后端（`auto_round:auto_gptq`）和 checkpoint 的
`packing_format` 语义不同，不能仅按请求名称推导元数据。

`SUPPORTED_FORMATS` 来自公开注册名与 `GGUF_CONFIG` 别名。内部别名可使用
`public=False`。GGUF+fake 仍先导出 fake。checkpoint 布局与元数据由各格式管理。
回归测试覆盖两项一致性修复：`inplace=False` 先复制后移动；原生 E5M3 整模型保存使用
与即时打包一致的 packer。AWQ 在无输出目录且 `inplace=False` 时返回打包后的副本。

## 验证

common 导出测试覆盖注册、别名、解析、数值打包工具、转换工具及不存在旧 exporter 导入。
小型 CPU 测试覆盖真实整数/FP8 张量、元数据、checkpoint 保存、无输出目录与副本隔离。
`test_custom_format_save.py` 本地构造微型 OPT 模型及 tokenizer，执行真实 RTN，验证仅
实现两个统一方法的自定义格式可以完成保存，无需网络下载。

当前开发环境验证：主要 common/CPU 导出与 CLI 测试组 520 个通过（42 个 compressed-tensors
测试因缺少该可选依赖跳过），另外 18 个 fake/GGUF 生命周期 CPU 测试通过。wheel
构建完成，确认包含新格式包、GGUF 版本标记，且不包含旧 exporter 路径。pre-commit
通过；因 sandbox 内 black 多文件进程池挂起，black hook 按文件分别运行。

当前环境完整 GGUF 转换测试未全绿：已安装 GGUF 缺少 `MODEL_ARCH.DFLASH`，
transformers 缺少 `PrefixChange`，迁移前已观察到这些失败。CPU 验证结果不代表
外部推理引擎或硬件相关测试已经验证。
