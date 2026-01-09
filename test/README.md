# Unit Test (UT) Guide

This project uses `pytest` for unit testing. All test cases are under the `test/` directory.

## 1. Environment Setup
- Recommended Python 3.8 or above.
- Install dependencies:
  ```sh
  pip install -r ../requirements.txt
  pip install pytest
  ```

## 2. Test Directory Structure

Tests are organized by hardware backend (`test_cpu/`, `test_cuda/`) and functionality:

- **core/** - Core AutoRound API and quantization workflows
- **quantization/** - Quantization techniques (mixed-bit, MXFP, NVFP4, activation quant)
- **export/** - Model serialization (GGUF, AutoGPTQ, AutoRound format)
- **backends/** - Inference backends (Torch, Marlin, Triton, ExLlamaV2)
- **models/** - Architecture-specific tests (MLLMs, VLMs, MoE, Diffusion)
- **integrations/** - Third-party frameworks (vLLM, SGLang, LLMC, Transformers)
- **schemes/** - Quantization scheme selection and configuration
- **utils/** - Calibration datasets, logging, CLI, model loading
- **advanced/** - Multi-GPU, FP8 input, custom pipelines

## 3. Shared Test Utilities

### conftest.py
Pytest configuration file that:
- Adds parent directory to `sys.path` for easy debugging without installation
- Defines HPU-specific test options (`--mode=compile/lazy`)
- Imports all fixtures from `fixtures.py`

### fixtures.py
Provides reusable pytest fixtures for testing:

**Model Fixtures:**
- `tiny_opt_model_path` - OPT-125M model with 2 layers (session scope)
- `tiny_qwen_model_path` - Qwen-0.6B model with 2 layers
- `tiny_lamini_model_path` - LaMini-GPT-124M with 2 layers
- `tiny_gptj_model_path` - Tiny GPT-J model
- `tiny_phi2_model_path` - Phi-2 model with 2 layers
- `tiny_deepseek_v2_model_path` - DeepSeek-V2-Lite with 2 layers
- `tiny_qwen_moe_model_path` - Qwen-1.5-MoE with 2 layers
- `tiny_qwen_vl_model_path` - Qwen2-VL-2B with 2 layers (vision model)
- `tiny_qwen_2_5_vl_model_path` - Qwen2.5-VL-3B with 2 layers

**Data Fixtures:**
- `dataloader` - Simple calibration dataloader with 4 text samples

All model fixtures:
- Use session scope to avoid reloading models for each test
- Automatically save tiny models to `./tmp/` directory
- Clean up temporary files after test session ends

### helpers.py
Utility functions for testing:

**Model Path Resolution:**
```python
get_model_path(model_name)  # Automatically finds local or remote model path
```

**Predefined Model Paths:**
```python
opt_name_or_path  # facebook/opt-125m
qwen_name_or_path  # Qwen/Qwen3-0.6B
lamini_name_or_path  # MBZUAI/LaMini-GPT-124M
qwen_vl_name_or_path  # Qwen/Qwen2-VL-2B-Instruct
# ... and more
```

**Model Manipulation:**
```python
get_tiny_model(model_path, num_layers=2)  # Create tiny model by slicing layers
save_tiny_model(model_path, save_path)  # Save tiny model to disk
```

**Model Inference:**
```python
model_infer(model, tokenizer, input_text)  # Run inference and return output
is_model_outputs_similar(out1, out2)  # Compare two model outputs
```

**Data Utilities:**
```python
DataLoader()  # Simple dataloader for calibration datasets
```

## 4. Writing New Tests

### Basic Example
```python
# test_cpu/quantization/test_new_method.py
import pytest
from auto_round import AutoRound
from ...helpers import opt_name_or_path


class TestNewQuantMethod:
    def test_quantization(self, tiny_opt_model_path, dataloader):
        """Test new quantization method."""
        autoround = AutoRound(model=tiny_opt_model_path, bits=4, group_size=128, iters=2, dataset=dataloader)
        autoround.quantize()
        assert autoround is not None
```

### Using Helpers and Fixtures
```python
from ...helpers import model_infer, opt_name_or_path, get_model_path


def test_model_inference(tiny_opt_model_path):
    # Use predefined model path
    model_name = opt_name_or_path

    # Or resolve custom model path
    custom_model = get_model_path("custom/model-name")

    # Run inference using helper
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(tiny_opt_model_path)
    tokenizer = AutoTokenizer.from_pretrained(tiny_opt_model_path)
    output = model_infer(model, tokenizer, "Hello world")
```

### Placement Guidelines
- **CPU-specific** → `test_cpu/<category>/`
- **CUDA-specific** → `test_cuda/<category>/`
- **Cross-platform** → Choose most relevant directory
- Import from parent: `from ...helpers import ...`

## 5. Running Tests

```sh
# Run all tests
pytest

# Run specific directory
pytest test_cpu/quantization/

# Run specific file
pytest test_cpu/core/test_autoround.py

# Run specific test
pytest -k "test_layer_config"

# Run with verbose output
pytest -v -s
```

## 6. Hardware-Specific Requirements
- **test_cpu/**: Install `pip install -r test_cpu/requirements.txt`
- **test_cuda/**: Install `pip install -r test_cuda/requirements.txt`
  - VLM: `pip install -r test_cuda/requirements_vlm.txt`
  - Diffusion: `pip install -r test_cuda/requirements_diffusion.txt`
  - LLMC: `pip install -r test_cuda/requirements_llmc.txt`
  - SGLang: `pip install -r test_cuda/requirements_sglang.txt`

## 7. Contributing
When adding new tests:
1. Place in appropriate category subdirectory
2. Use existing fixtures and helpers
3. Clean up resources in teardown methods
4. Use descriptive names and docstrings

For questions, open an issue.
