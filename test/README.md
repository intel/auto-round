# Unit Test (UT) Guide

This project uses `pytest` for unit testing. All test cases are under the `test/` directory. Below is a comprehensive guide for new users to write and run UTs:

## 1. Environment Setup
- Recommended Python 3.8 or above.
- Install dependencies:
  ```sh
  pip install -r ../requirements.txt
  pip install pytest
  ```

## 2. Test Directory Structure

The test suite is organized by hardware backend and functionality. Tests are categorized into the following subdirectories:

### `test_cpu/` and `test_cuda/`
Both CPU and CUDA test directories follow the same organizational structure:

#### **core/**
Core AutoRound functionality tests including:
- Basic AutoRound operations and workflows
- Model quantization accuracy tests
- Core API functionality
- Examples: `test_autoround.py`, `test_autoround_acc.py`, `test_autoopt.py`, `test_init.py`

#### **quantization/**
Quantization techniques and data type tests:
- Activation quantization
- Asymmetric and mixed-bit quantization
- MXFP, NVFP4 and other floating-point formats
- Quantization linear layers
- Examples: `test_act_quantization.py`, `test_mix_bits.py`, `test_mxfp_nvfp.py`, `test_2_3bits.py`

#### **export/**
Model export and serialization format tests:
- AutoGPTQ, GGUF, and other export formats
- Model saving and loading
- Format-specific features
- Examples: `test_export.py`, `test_gguf_format.py`, `test_auto_round_format.py`

#### **backends/**
Backend implementation tests for different inference engines:
- Torch backend
- Marlin, Triton, ExLlamaV2 backends
- Backend-specific optimizations
- Examples: `test_torch_backend.py`, `test_marlin_backend.py`, `test_triton_backend.py`

#### **models/**
Model-specific tests for various architectures:
- Multimodal LLMs (MLLM)
- Vision-Language Models (VLMs)
- Mixture-of-Experts (MoE) models
- Diffusion models
- Special layer types (Conv1D, etc.)
- Examples: `test_mllm.py`, `test_vlms.py`, `test_moe_model.py`, `test_diffusion.py`

#### **integrations/**
Third-party framework integration tests:
- LLMC (LLM Compressor) integration
- vLLM and SGLang integration
- Transformers library integration
- Examples: `test_llmc_integration.py`, `test_vllm.py`, `test_sglang.py`, `test_transformers.py`

#### **schemes/**
Quantization scheme tests:
- Custom quantization schemes
- Auto-scheme selection
- Scheme configuration and validation
- Examples: `test_scheme.py`, `test_auto_scheme.py`

#### **utils/**
Utility and helper function tests:
- Calibration dataset handling
- Model generation and inference utilities
- Loading pretrained quantized models (AWQ, GPTQ)
- Logging, CLI, and configuration utilities
- Examples: `test_calib_dataset.py`, `test_generation.py`, `test_logger.py`, `test_cli_usage.py`

#### **advanced/**
Advanced features and multi-device tests:
- Multi-GPU support
- FP8 input handling
- Custom data pipelines
- Examples: `test_multiple_card.py`, `test_fp8_input.py`, `test_customized_data.py`

### Other Test Directories
- **test_hpu/**: Intel Habana HPU-specific tests
- **test_xpu/**: Intel XPU (GPU) specific tests
- **test_ark/**: ARK-specific tests

## 3. Writing New Tests

### Guidelines for Adding Tests
When adding a new test, follow these guidelines to maintain the organized structure:

1. **Identify the correct category**: Determine which category your test belongs to based on the functionality being tested.

2. **Place in the appropriate directory**:
   - If testing CPU-specific functionality → `test_cpu/<category>/`
   - If testing CUDA-specific functionality → `test_cuda/<category>/`
   - If testing cross-platform functionality → Add to both directories or the most relevant one

3. **Naming convention**: 
   - Name test files starting with `test_`
   - Use descriptive names: `test_<feature_name>.py`
   - Example: `test_new_quantization_method.py`

4. **Category selection**:
   - **core/**: If testing fundamental AutoRound API or workflows
   - **quantization/**: If testing a new quantization technique or data type
   - **export/**: If testing model serialization or export formats
   - **backends/**: If testing a specific inference backend
   - **models/**: If testing a specific model architecture or family
   - **integrations/**: If testing integration with external frameworks
   - **schemes/**: If testing quantization scheme logic
   - **utils/**: If testing helper functions, utilities, or supporting features
   - **advanced/**: If testing multi-device or advanced optimization features

5. **Use existing fixtures and helpers**:
   - Common fixtures are defined in `conftest.py` and `fixtures.py`
   - Helper functions are available in `helpers.py`
   - Import from parent directory: `from ...helpers import model_infer`

### Example Test Structure
```python
# test_cpu/quantization/test_new_method.py
import pytest
from auto_round import AutoRound
from ...helpers import model_infer, opt_name_or_path

class TestNewQuantMethod:
    @classmethod
    def setup_class(self):
        self.model_name = opt_name_or_path
    
    def test_new_quantization_method(self, tiny_opt_model_path, dataloader):
        """Test new quantization method."""
        autoround = AutoRound(
            model=tiny_opt_model_path,
            bits=4,
            group_size=128,
            iters=2,
            dataset=dataloader
        )
        autoround.quantize()
        # Add assertions here
        assert autoround is not None
```

## 4. Running Tests

### Run all tests:
```sh
pytest
```

### Run tests in a specific directory:
```sh
pytest test_cpu/quantization/
pytest test_cuda/backends/
```

### Run a specific test file:
```sh
pytest test_cpu/core/test_autoround.py
```

### Run a specific test case:
```sh
pytest test_cpu/core/test_autoround.py::TestAutoRound::test_layer_config
pytest -k "test_layer_config"
```

### Run tests for a specific category across both CPU and CUDA:
```sh
pytest test_cpu/quantization/ test_cuda/quantization/
```

## 5. Debugging Tips
- `conftest.py` adds the parent directory to `sys.path`, so you can debug without installing the local package.
- You can directly import project source code in your test cases.
- Use pytest's `-v` flag for verbose output: `pytest -v`
- Use pytest's `-s` flag to see print statements: `pytest -s`
- Use pytest's `--pdb` flag to drop into debugger on failures: `pytest --pdb`

## 6. Hardware-Specific Requirements
- **test_cpu/**: Tests that run on CPU only. Install dependencies: `pip install -r test_cpu/requirements.txt`
- **test_cuda/**: Tests that require CUDA GPU. Install dependencies: `pip install -r test_cuda/requirements.txt`
  - Additional requirements for specific features:
    - VLM tests: `pip install -r test_cuda/requirements_vlm.txt`
    - Diffusion tests: `pip install -r test_cuda/requirements_diffusion.txt`
    - LLMC tests: `pip install -r test_cuda/requirements_llmc.txt`
    - SGLang tests: `pip install -r test_cuda/requirements_sglang.txt`

## 7. Reference
- Common fixtures are defined in `conftest.py` and `fixtures.py`
- Helper functions are in `helpers.py`
- Each test category has its own `__init__.py` for module initialization

## 8. Contributing Guidelines
When contributing new tests:
1. Follow the category structure outlined above
2. Add tests to the appropriate subdirectory
3. Ensure tests are self-contained and do not depend on execution order
4. Clean up resources (models, temporary files) in teardown methods
5. Use descriptive test names and add docstrings
6. Update this README if adding a new category

If you have any questions, feel free to open an issue.
