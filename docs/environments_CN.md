# AutoRound 环境变量配置

[English](./environments.md) | 简体中文

本文档介绍 AutoRound 使用的环境变量及其配置说明。

## 概述

AutoRound 通过 `envs.py` 模块提供统一的环境变量管理系统，支持懒加载求值与程序化配置。

## 可用环境变量

### AR_LOG_LEVEL
- **描述**：控制 AutoRound 默认日志级别
- **默认值**：`"INFO"`
- **有效值**：`"TRACE"`、`"DEBUG"`、`"INFO"`、`"WARNING"`、`"ERROR"`、`"CRITICAL"`
- **用途**：通过设置该变量控制 AutoRound 的日志详细程度

```bash
export AR_LOG_LEVEL=DEBUG
```

### AR_ENABLE_COMPILE_PACKING
- **描述**：启用编译打包优化
- **默认值**：`False`（等价于 `"0"`）
- **有效值**：`"1"`、`"true"`、`"yes"`（不区分大小写）表示启用；其他值表示禁用
- **用途**：启用后可在将 FP4 张量打包为 `uint8` 时获得性能优化

```bash
export AR_ENABLE_COMPILE_PACKING=1
```

### AR_USE_MODELSCOPE
- **描述**：控制是否使用 ModelScope 下载模型
- **默认值**：`False`
- **有效值**：`"1"`、`"true"`（不区分大小写）表示启用；其他值表示禁用
- **用途**：启用后将使用 ModelScope 替代 Hugging Face Hub 下载模型

```bash
export AR_USE_MODELSCOPE=true
```

### AR_WORK_SPACE
- **描述**：设置 AutoRound 操作的工作目录
- **默认值**：`"ar_work_space"`
- **用途**：指定 AutoRound 存储临时文件和输出结果的自定义目录

```bash
export AR_WORK_SPACE=/path/to/custom/workspace
```

### AR_DISABLE_OFFLOAD
- **描述**：强制禁用 `OffloadManager` 中的权重卸载功能。在开发和调试时可跳过所有卸载/重载开销。
- **默认值**：`False`（等价于 `"0"`）
- **有效值**：`"1"`、`"true"`、`"yes"`（不区分大小写）表示禁用卸载；其他值保持默认行为
- **用途**：设置后将完全绕过权重卸载

```bash
export AR_DISABLE_OFFLOAD=1
```

### AR_DISABLE_DATASET_SUBPROCESS
- **描述**：禁用子进程方式进行数据集预处理。默认情况下，AutoRound 使用子进程确保所有临时内存在进程退出后被操作系统回收。
- **默认值**：`False`
- **有效值**：`"1"`、`"true"`（不区分大小写）表示禁用子进程；其他值表示启用子进程
- **用途**：设置后数据集预处理将在主进程中运行

```bash
export AR_DISABLE_DATASET_SUBPROCESS=true
```

## 使用示例

### 设置环境变量

#### 通过 Shell 命令
```bash
# 将日志级别设置为 DEBUG
export AR_LOG_LEVEL=DEBUG

# 启用编译打包
export AR_ENABLE_COMPILE_PACKING=1

# 使用 ModelScope 下载模型
export AR_USE_MODELSCOPE=true

# 设置自定义工作目录
export AR_WORK_SPACE=/tmp/autoround_workspace
```

#### 通过 Python 代码
```python
from auto_round.envs import set_config

# 同时配置多个环境变量
set_config(
    AR_LOG_LEVEL="DEBUG",
    AR_USE_MODELSCOPE=True,
    AR_ENABLE_COMPILE_PACKING=True,
    AR_WORK_SPACE="/tmp/autoround_workspace",
)
```

### 查看环境变量

#### 通过 Python 代码
```python
from auto_round import envs

# 访问环境变量（懒加载求值）
log_level = envs.AR_LOG_LEVEL
use_modelscope = envs.AR_USE_MODELSCOPE
enable_packing = envs.AR_ENABLE_COMPILE_PACKING
workspace = envs.AR_WORK_SPACE

print(f"日志级别: {log_level}")
print(f"使用 ModelScope: {use_modelscope}")
print(f"启用编译打包: {enable_packing}")
print(f"工作目录: {workspace}")
```

#### 检查变量是否显式设置
```python
from auto_round.envs import is_set

# 检查环境变量是否被显式设置
if is_set("AR_LOG_LEVEL"):
    print("AR_LOG_LEVEL 已被显式设置")
else:
    print("AR_LOG_LEVEL 正在使用默认值")
```

## 配置最佳实践

1. **开发环境**：设置 `AR_LOG_LEVEL=TRACE` 或 `AR_LOG_LEVEL=DEBUG` 以获取详细日志
2. **生产环境**：使用 `AR_LOG_LEVEL=WARNING` 或 `AR_LOG_LEVEL=ERROR` 减少日志噪声
3. **中国用户**：建议设置 `AR_USE_MODELSCOPE=true` 以获得更好的模型下载速度
4. **性能优化**：如有足够算力，可启用 `AR_ENABLE_COMPILE_PACKING=1`
5. **自定义工作目录**：将 `AR_WORK_SPACE` 设置为磁盘空间充足的目录

## 注意事项

- 环境变量采用懒加载方式，仅在首次访问时读取
- `set_config()` 函数提供了便捷的程序化多变量配置方式
- `AR_USE_MODELSCOPE` 的布尔值会自动转换为适当的字符串表示
- 所有环境变量名称区分大小写
- 通过 `set_config()` 所做的修改将影响当前进程及其子进程
