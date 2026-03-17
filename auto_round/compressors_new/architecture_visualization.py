# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

"""
New Architecture Visualization - Mixin Pattern Combination Table

Demonstrates all possible combinations of model types and compression algorithms.
"""


def print_architecture_table():
    """Print architecture combination table"""

    print("\n" + "=" * 100)
    print("Compressor New Architecture - Mixin Pattern Combination Table")
    print("=" * 100 + "\n")

    # Table header
    print(f"{'Model Type':<15} {'Config Type':<20} {'Algorithm':<20} {'Actual Created Class':<35}")
    print("-" * 100)

    # LLM combinations
    print(f"{'LLM':<15} {'AutoRoundConfig':<20} {'AutoRound':<20} {'CalibCompessor':<35}")
    print(f"{'LLM':<15} {'RTNConfig':<20} {'RTN + imatrix':<20} {'ImatrixCompressor':<35}")
    print(f"{'LLM':<15} {'RTNConfig':<20} {'RTN (zero-shot)':<20} {'ZeroShotCompressor':<35}")

    print()

    # MLLM combinations
    print(f"{'MLLM':<15} {'AutoRoundConfig':<20} {'AutoRound':<20} {'MLLMCalibCompressor':<35}")
    print(f"{'':<15} {'':<20} {'':<20} {'  = MLLMMixin + CalibCompessor':<35}")
    print(f"{'MLLM':<15} {'RTNConfig':<20} {'RTN + imatrix':<20} {'MLLMImatrixCompressor':<35}")
    print(f"{'':<15} {'':<20} {'':<20} {'  = MLLMMixin + ImatrixCompressor':<35}")
    print(f"{'MLLM':<15} {'RTNConfig':<20} {'RTN (zero-shot)':<20} {'MLLMZeroShotCompressor':<35}")
    print(f"{'':<15} {'':<20} {'':<20} {'  = MLLMMixin + ZeroShotCompressor':<35}")

    print()

    # Diffusion combinations
    print(f"{'Diffusion':<15} {'AutoRoundConfig':<20} {'AutoRound':<20} {'DiffusionCalibCompressor':<35}")
    print(f"{'':<15} {'':<20} {'':<20} {'  = DiffusionMixin + CalibCompessor':<35}")
    print(f"{'Diffusion':<15} {'RTNConfig':<20} {'RTN + imatrix':<20} {'DiffusionImatrixCompressor':<35}")
    print(f"{'':<15} {'':<20} {'':<20} {'  = DiffusionMixin + ImatrixCompressor':<35}")
    print(f"{'Diffusion':<15} {'RTNConfig':<20} {'RTN (zero-shot)':<20} {'DiffusionZeroShotCompressor':<35}")
    print(f"{'':<15} {'':<20} {'':<20} {'  = DiffusionMixin + ZeroShotCompressor':<35}")

    print("\n" + "=" * 100 + "\n")


def print_mixin_explanation():
    """Print Mixin pattern explanation"""

    print("=" * 100)
    print("Mixin Pattern Explanation")
    print("=" * 100 + "\n")

    print("✨ Core Components:")
    print("-" * 100)
    print("  1. MLLMMixin         - MLLM features (processor, template, etc.)")
    print("  2. DiffusionMixin    - Diffusion features (guidance_scale, pipeline, etc.)")
    print("  3. CalibCompessor    - Calibration-based compression algorithm (AutoRound)")
    print("  4. ImatrixCompressor - RTN + importance matrix")
    print("  5. ZeroShotCompressor - Zero-shot RTN")

    print("\n🎯 Combination Approach:")
    print("-" * 100)
    print("  Dynamically create combined classes through multiple inheritance:")
    print("    class MLLMCalibCompressor(MLLMMixin, CalibCompessor):")
    print("        pass")
    print("\n  MLLMMixin provides MLLM features, CalibCompessor provides compression algorithm")

    print("\n💡 Advantages:")
    print("-" * 100)
    print("  ✓ Flexible Combination: Any model feature can be combined with any compression algorithm")
    print("  ✓ Code Reuse: Mixin code is written once and can be reused multiple times")
    print("  ✓ Clear Separation: Model features and compression algorithms are completely independent")
    print("  ✓ Easy Extension: Adding new model types or new algorithms is straightforward")

    print("\n" + "=" * 100 + "\n")


def print_usage_examples():
    """Print usage examples"""

    print("=" * 100)
    print("Usage Examples")
    print("=" * 100 + "\n")

    print("Example 1: MLLM + AutoRound")
    print("-" * 100)
    print(
        """
from auto_round.compressors_new.entry import Compressor
from auto_round.algorithms.quantization.auto_round.config import AutoRoundConfig

config = AutoRoundConfig(scheme="W4A16", iters=200)
compressor = Compressor(
    config=config,
    model="/models/Qwen2-VL-2B-Instruct",
    processor=processor,
    template="qwen2_vl",
)
# Actually creates: MLLMCalibCompressor (MLLMMixin + CalibCompessor)
    """
    )

    print("\nExample 2: MLLM + RTN + imatrix")
    print("-" * 100)
    print(
        """
from auto_round.algorithms.quantization.rtn.config import RTNConfig

config = RTNConfig(scheme="W4A16")
compressor = Compressor(
    config=config,
    model="/models/Qwen2-VL-2B-Instruct",
    format="gguf_k",  # Triggers imatrix
    processor=processor,
)
# Actually creates: MLLMImatrixCompressor (MLLMMixin + ImatrixCompressor)
    """
    )

    print("\nExample 3: Diffusion + AutoRound")
    print("-" * 100)
    print(
        """
config = AutoRoundConfig(scheme="W4A16", iters=200)
compressor = Compressor(
    config=config,
    model="/models/stable-diffusion-2-1",
    guidance_scale=7.5,
)
# Actually creates: DiffusionCalibCompressor (DiffusionMixin + CalibCompessor)
    """
    )

    print("\n" + "=" * 100 + "\n")


def print_mro_example():
    """Print MRO (Method Resolution Order) example"""

    print("=" * 100)
    print("Method Resolution Order (MRO) Example")
    print("=" * 100 + "\n")

    print("For MLLMCalibCompressor(MLLMMixin, CalibCompessor):")
    print("-" * 100)
    print(
        """
MLLMCalibCompressor
    └─> MLLMMixin
        └─> CalibCompessor
            └─> BaseCompressor
                └─> object

Execution order when calling __init__():
  1. MLLMCalibCompressor.__init__() (if defined)
  2. MLLMMixin.__init__()
     - Save MLLM-specific parameters (processor, template, etc.)
     - Call super().__init__() → enters CalibCompessor
  3. CalibCompessor.__init__()
     - Save calibration-related parameters (dataset, iters, etc.)
     - Call super().__init__() → enters BaseCompressor
  4. BaseCompressor.__init__()
     - Base class initialization

Thus, MLLMCalibCompressor has both:
  ✓ MLLM features (from MLLMMixin)
  ✓ Calibration compression functionality (from CalibCompessor)
    """
    )

    print("=" * 100 + "\n")


def print_decision_tree():
    """Print decision tree"""

    print("=" * 100)
    print("Compressor Creation Decision Tree")
    print("=" * 100 + "\n")

    print(
        """
Compressor.__new__(config, model, ...)
│
├─ Step 1: Detect model type
│  model_type = detect_model_type(model)
│  ├─ is_diffusion_model() → "diffusion"
│  ├─ is_mllm_model() → "mllm"
│  └─ else → "llm"
│
├─ Step 2: Determine config type
│  │
│  ├─ AutoRoundConfig (requires calibration)
│  │  ├─ model_type == "mllm"
│  │  │  └─> class MLLMCalibCompressor(MLLMMixin, CalibCompessor)
│  │  │      return MLLMCalibCompressor(...)
│  │  │
│  │  ├─ model_type == "diffusion"
│  │  │  └─> class DiffusionCalibCompressor(DiffusionMixin, CalibCompessor)
│  │  │      return DiffusionCalibCompressor(...)
│  │  │
│  │  └─ model_type == "llm"
│  │     └─> return CalibCompessor(...)
│  │
│  └─ RTNConfig (zero-shot or imatrix)
│     │
│     ├─ enable_imatrix == True
│     │  ├─ model_type == "mllm"
│     │  │  └─> class MLLMImatrixCompressor(MLLMMixin, ImatrixCompressor)
│     │  │      return MLLMImatrixCompressor(...)
│     │  │
│     │  ├─ model_type == "diffusion"
│     │  │  └─> class DiffusionImatrixCompressor(DiffusionMixin, ImatrixCompressor)
│     │  │      return DiffusionImatrixCompressor(...)
│     │  │
│     │  └─ model_type == "llm"
│     │     └─> return ImatrixCompressor(...)
│     │
│     └─ enable_imatrix == False
│        ├─ model_type == "mllm"
│        │  └─> class MLLMZeroShotCompressor(MLLMMixin, ZeroShotCompressor)
│        │      return MLLMZeroShotCompressor(...)
│        │
│        ├─ model_type == "diffusion"
│        │  └─> class DiffusionZeroShotCompressor(DiffusionMixin, ZeroShotCompressor)
│        │      return DiffusionZeroShotCompressor(...)
│        │
│        └─ model_type == "llm"
│           └─> return ZeroShotCompressor(...)
    """
    )

    print("=" * 100 + "\n")


def main():
    """Run all visualizations"""

    print_architecture_table()
    print_mixin_explanation()
    print_usage_examples()
    print_mro_example()
    print_decision_tree()

    print("=" * 100)
    print("🎉 New architecture supports 9 combinations (3 model types × 3 compression algorithms)")
    print("=" * 100)


if __name__ == "__main__":
    main()
