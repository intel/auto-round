# AGENTS.md

AutoRound 智能体使用说明。

## 项目概览
- AutoRound 是面向 LLM/VLM 的低比特量化 Python 工具包。
- 主要入口是 CLI 命令，例如 `auto-round`、`auto-round-best`、`auto-round-light`。

## 仓库结构
- `auto_round/`: 核心库与量化逻辑
- `docs/`: 使用指南与技术文档
- `test/`: 单元测试
- `.azure-pipelines/`: CI 脚本与测试编排

## 架构概览
- CLI 入口：`auto_round/__main__.py`
- 核心量化 API：`auto_round/autoround.py`
- 压缩器实现：`auto_round/compressors/`（LLM/MLLM/Diffusion）
- 方案与预设：`auto_round/schemes.py`
- 导出格式与流程：`auto_round/formats.py`、`auto_round/export/`
- AutoScheme 生成器：`auto_round/auto_scheme/`
- 数据类型定义：`auto_round/data_type/`
- 模型特定补丁：`auto_round/modeling/`
- 通用工具：`auto_round/utils/`
- 评测入口：`auto_round/eval/`

## 环境设置
- 需要 Python 3.10+。
- 开发安装：`pip install -e .`
- 仅安装运行时依赖：`pip install -r requirements.txt`

## 构建与测试命令
- 从源码构建：`pip install .`
- 运行单元测试：`pytest test`

## 常用命令
- CLI 帮助：`auto-round -h`
- 列出支持的导出格式：`auto_round list format`

## 最小示例
- CLI 量化：

```bash
auto-round --model Qwen/Qwen3-0.6B --scheme "W4A16" --format "auto_round" --output_dir ./tmp_autoround
```

- API 量化：

```python
from auto_round import AutoRound

ar = AutoRound("Qwen/Qwen3-0.6B", scheme="W4A16")
ar.quantize_and_save(output_dir="./qmodel", format="auto_round")
```

- GGUF 导出（单一格式）：

```bash
auto-round --model Qwen/Qwen3-0.6B --scheme "W4A16" --format "gguf:q4_k_m" --output_dir ./tmp_autoround_gguf
```

- CompressedTensors 导出（LLM-Compressor 格式）：

```bash
auto-round --model Qwen/Qwen3-0.6B --scheme "NVFP4" --format "llm_compressor" --output_dir ./tmp_autoround_ct
```

## 测试
- 本地快速运行：`pytest test`
- CI 通过 `.azure-pipelines/scripts/ut/run_ut.sh` 拆分运行 CPU 测试，并在部分场景下
  运行 LLM-compressor 测试（依赖 `uv`、`numactl`，会安装额外依赖）。运行时间更长且
  对系统资源要求更高。
- 优先运行与你改动相关的定向测试。

## 代码风格与检查
- Python 格式化遵循 Black/Ruff，行宽 120。
- 新增字符串优先使用双引号（Ruff 默认）。
- 保持导入排序一致（isort profile: black）。

## 文档与翻译
- 如果修改了带 `_CN` 对应版本的 Markdown，请同步更新中文文件以保持内容和结构一致
  （例如 `README.md` 与 `README_CN.md`）。

## 贡献
- 提交需要 DCO 签名（参见 `CONTRIBUTING.md`）。

## 数据与大文件
- 避免提交模型权重、大型二进制或数据集。
- 需要样例数据时，优先使用小型 fixture 或公开 URL。

## 智能体常见错误防范
- 不要把 GGUF 导出与其他格式混用；GGUF 只选一种格式。
- 修改带 `_CN` 对应版本的 Markdown 时，请同步更新两份文件。
- 非必要不要在本地跑完整 CI；优先定向测试。
- 不要提交大型产物（模型、二进制、数据集）。
