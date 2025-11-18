# AutoRound Environment Variables Configuration

This document describes the environment variables used by AutoRound for configuration and their usage.

## Overview

AutoRound uses a centralized environment variable management system through the `envs.py` module. This system provides lazy evaluation of environment variables and programmatic configuration capabilities.

## Available Environment Variables

### AR_LOG_LEVEL
- **Description**: Controls the default logging level for AutoRound
- **Default**: `"INFO"`
- **Valid Values**: `"TRACE"`,  `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`
- **Usage**: Set this to control the verbosity of AutoRound logs

```bash
export AR_LOG_LEVEL=DEBUG
```

### AR_ENABLE_COMPILE_PACKING
- **Description**: Enables compile packing optimization
- **Default**: `False` (equivalent to `"0"`)
- **Valid Values**: `"1"`, `"true"`, `"yes"` (case-insensitive) for enabling; any other value for disabling
- **Usage**: Enable this for performance optimizations during packing FP4 tensors into `uint8`.

```bash
export AR_ENABLE_COMPILE_PACKING=1
```

### AR_USE_MODELSCOPE
- **Description**: Controls whether to use ModelScope for model downloads
- **Default**: `False`
- **Valid Values**: `"1"`, `"true"` (case-insensitive) for enabling; any other value for disabling
- **Usage**: Enable this to use ModelScope instead of Hugging Face Hub for model downloads

```bash
export AR_USE_MODELSCOPE=true
```

### AR_WORK_SPACE
- **Description**: Sets the workspace directory for AutoRound operations
- **Default**: `"ar_work_space"`
- **Usage**: Specify a custom directory for AutoRound to store temporary files and outputs

```bash
export AR_WORK_SPACE=/path/to/custom/workspace
```

## Usage Examples

### Setting Environment Variables

#### Using Shell Commands
```bash
# Set logging level to DEBUG
export AR_LOG_LEVEL=DEBUG

# Enable compile packing
export AR_ENABLE_COMPILE_PACKING=1

# Use ModelScope for downloads
export AR_USE_MODELSCOPE=true

# Set custom workspace
export AR_WORK_SPACE=/tmp/autoround_workspace
```

#### Using Python Code
```python
from auto_round.envs import set_config

# Configure multiple environment variables at once
set_config(
    AR_LOG_LEVEL="DEBUG",
    AR_USE_MODELSCOPE=True,
    AR_ENABLE_COMPILE_PACKING=True,
    AR_WORK_SPACE="/tmp/autoround_workspace",
)
```

### Checking Environment Variables

#### Using Python Code
```python
from auto_round import envs

# Access environment variables (lazy evaluation)
log_level = envs.AR_LOG_LEVEL
use_modelscope = envs.AR_USE_MODELSCOPE
enable_packing = envs.AR_ENABLE_COMPILE_PACKING
workspace = envs.AR_WORK_SPACE

print(f"Log Level: {log_level}")
print(f"Use ModelScope: {use_modelscope}")
print(f"Enable Compile Packing: {enable_packing}")
print(f"Workspace: {workspace}")
```

#### Checking if Variables are Explicitly Set
```python
from auto_round.envs import is_set

# Check if environment variables are explicitly set
if is_set("AR_LOG_LEVEL"):
    print("AR_LOG_LEVEL is explicitly set")
else:
    print("AR_LOG_LEVEL is using default value")
```

## Configuration Best Practices

1. **Development Environment**: Set `AR_LOG_LEVEL=DEBUG` for detailed logging during development
2. **Production Environment**: Use `AR_LOG_LEVEL=WARNING` or `AR_LOG_LEVEL=ERROR` to reduce log noise
3. **Chinese Users**: Consider setting `AR_USE_MODELSCOPE=true` for better model download performance
4. **Performance Optimization**: Enable `AR_ENABLE_COMPILE_PACKING=1` if you have sufficient computational resources
5. **Custom Workspace**: Set `AR_WORK_SPACE` to a directory with sufficient disk space for model processing

## Notes

- Environment variables are evaluated lazily, meaning they are only read when first accessed
- The `set_config()` function provides a convenient way to configure multiple variables programmatically
- Boolean values for `AR_USE_MODELSCOPE` are automatically converted to appropriate string representations
- All environment variable names are case-sensitive
- Changes made through `set_config()` will affect the current process and any child processes
