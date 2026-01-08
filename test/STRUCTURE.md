# Test Directory Structure

This document provides an overview of the organized test directory structure.

## Directory Layout

```
test/
├── README.md                    # Comprehensive testing guide
├── STRUCTURE.md                 # This file - structure overview
├── conftest.py                  # Pytest configuration and global fixtures
├── fixtures.py                  # Shared test fixtures
├── helpers.py                   # Helper functions for tests
├── __init__.py
│
├── test_cpu/                    # CPU-only tests
│   ├── requirements.txt         # CPU test dependencies
│   ├── __init__.py
│   │
│   ├── core/                    # Core functionality (5 tests)
│   │   ├── test_autoround.py
│   │   ├── test_autoround_acc.py
│   │   ├── test_autoround_export_to_itrex.py
│   │   ├── test_autoopt.py
│   │   └── test_init.py
│   │
│   ├── quantization/            # Quantization methods (7 tests)
│   │   ├── test_act_quantization.py
│   │   ├── test_asym.py
│   │   ├── test_mix_bits.py
│   │   ├── test_mx_quant_linear.py
│   │   ├── test_mxfp_nvfp.py
│   │   ├── test_mxfp_save_load.py
│   │   └── test_nvfp4_quant_linear.py
│   │
│   ├── export/                  # Export formats (2 tests)
│   │   ├── test_export.py
│   │   └── test_gguf_format.py
│   │
│   ├── backends/                # Backend implementations (1 test)
│   │   └── test_torch_backend.py
│   │
│   ├── models/                  # Model-specific tests (5 tests)
│   │   ├── test_block_names.py
│   │   ├── test_conv1d.py
│   │   ├── test_mllm.py
│   │   ├── test_moe_alignment.py
│   │   └── test_moe_model.py
│   │
│   ├── integrations/            # Third-party integrations (2 tests)
│   │   ├── test_llmc_integration.py
│   │   └── test_llmcompressor.py
│   │
│   ├── schemes/                 # Quantization schemes (2 tests)
│   │   ├── test_auto_scheme.py
│   │   └── test_scheme.py
│   │
│   ├── utils/                   # Utilities (9 tests)
│   │   ├── test_alg_ext.py
│   │   ├── test_calib_dataset.py
│   │   ├── test_cli_usage.py
│   │   ├── test_generation.py
│   │   ├── test_load_awq_gptq.py
│   │   ├── test_logger.py
│   │   ├── test_model_scope.py
│   │   ├── test_utils.py
│   │   └── test_woq_linear.py
│   │
│   └── advanced/                # Advanced features (0 tests)
│
├── test_cuda/                   # CUDA/GPU tests
│   ├── requirements.txt         # CUDA test dependencies
│   ├── requirements_diffusion.txt
│   ├── requirements_llmc.txt
│   ├── requirements_sglang.txt
│   ├── requirements_vlm.txt
│   ├── __init__.py
│   │
│   ├── core/                    # Core functionality (1 test)
│   │   └── test_main_func.py
│   │
│   ├── quantization/            # Quantization methods (7 tests)
│   │   ├── test_2_3bits.py
│   │   ├── test_asym.py
│   │   ├── test_mix_bits.py
│   │   ├── test_mxfp_and_nvfp_quant.py
│   │   ├── test_mxfp_nvfp.py
│   │   ├── test_packing.py
│   │   └── test_qbits.py
│   │
│   ├── export/                  # Export formats (3 tests)
│   │   ├── test_auto_round_format.py
│   │   ├── test_export.py
│   │   └── test_gguf.py
│   │
│   ├── backends/                # Backend implementations (4 tests)
│   │   ├── test_exllamav2_backend.py
│   │   ├── test_marlin_backend.py
│   │   ├── test_torch_backend.py
│   │   └── test_triton_backend.py
│   │
│   ├── models/                  # Model-specific tests (6 tests)
│   │   ├── test_conv1d.py
│   │   ├── test_diffusion.py
│   │   ├── test_get_block_name.py
│   │   ├── test_moe_model.py
│   │   ├── test_support_vlms.py
│   │   └── test_vlms.py
│   │
│   ├── integrations/            # Third-party integrations (4 tests)
│   │   ├── test_llmc_integration.py -> ../../test_cpu/integrations/test_llmc_integration.py
│   │   ├── test_sglang.py
│   │   ├── test_transformers.py
│   │   └── test_vllm.py
│   │
│   ├── schemes/                 # Quantization schemes (2 tests)
│   │   ├── test_auto_scheme.py
│   │   └── test_scheme.py
│   │
│   ├── utils/                   # Utilities (3 tests)
│   │   ├── test_alg_ext.py
│   │   ├── test_calib_dataset.py
│   │   └── test_customized_data.py
│   │
│   └── advanced/                # Advanced features (3 tests)
│       ├── test_fp8_input.py
│       ├── test_multiple_card.py
│       └── test_multiple_card_calib.py
│
├── test_hpu/                    # Intel Habana HPU tests
├── test_xpu/                    # Intel XPU (GPU) tests
└── test_ark/                    # ARK-specific tests
```

## Category Descriptions

### **core/**
Tests for fundamental AutoRound functionality:
- Basic quantization workflows
- Model accuracy after quantization
- Core API and initialization

### **quantization/**
Tests for various quantization techniques:
- Activation quantization
- Asymmetric and symmetric quantization
- Mixed-bit quantization (2-bit, 3-bit, 4-bit, etc.)
- Floating-point formats (MXFP, NVFP4)
- Quantization linear layer implementations

### **export/**
Tests for model serialization and export:
- AutoGPTQ format
- GGUF format (CPU and CUDA variants)
- AutoRound native format
- Format-specific optimizations

### **backends/**
Tests for inference backend implementations:
- Torch backend (CPU and CUDA)
- Marlin backend (CUDA optimized)
- Triton backend
- ExLlamaV2 backend

### **models/**
Tests for specific model architectures:
- Multimodal LLMs (MLLM) like Qwen-VL
- Vision-Language Models (VLMs)
- Mixture-of-Experts (MoE) models
- Diffusion models
- Models with special layers (Conv1D, etc.)

### **integrations/**
Tests for third-party framework integration:
- LLMC (LLM Compressor) integration
- vLLM inference engine
- SGLang inference framework
- Transformers library compatibility

### **schemes/**
Tests for quantization schemes:
- Custom quantization scheme definitions
- Auto-scheme selection and optimization
- Scheme configuration and validation

### **utils/**
Tests for utility functions:
- Calibration dataset handling
- Model generation and inference utilities
- Loading pretrained quantized models (AWQ, GPTQ)
- Logging, CLI, and configuration

### **advanced/**
Tests for advanced features:
- Multi-GPU and distributed quantization
- FP8 input handling
- Custom data pipelines
- Advanced optimizations

## Statistics

- **test_cpu**: 33 test files across 9 categories
- **test_cuda**: 33 test files across 9 categories  
- **Total**: 66 test files (one is a symlink)

## Running Tests by Category

```bash
# Run all core tests
pytest test_cpu/core/ test_cuda/core/

# Run all quantization tests
pytest test_cpu/quantization/ test_cuda/quantization/

# Run all backend tests
pytest test_cpu/backends/ test_cuda/backends/

# Run CPU tests only
pytest test_cpu/

# Run CUDA tests only
pytest test_cuda/

# Run a specific category on CPU
pytest test_cpu/models/

# Run a specific test file
pytest test_cpu/core/test_autoround.py
```

## Notes

- Import paths have been updated to reflect the new structure (from `..helpers` to `...helpers`)
- All `__init__.py` files are created for proper Python module structure
- The `test_llmc_integration.py` in `test_cuda/integrations/` is a symlink to the CPU version
- Requirements files remain at the root of `test_cpu/` and `test_cuda/` directories
