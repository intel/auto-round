# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0
"""
New Architecture Visualization - Mixin Pattern Combination Table

Demonstrates all possible combinations of model types and compression algorithms.
"""


def print_architecture_table():
    """Print architecture combination table"""

    print("\n" + "=" * 110)
    print("AutoRound New Architecture - Mixin Pattern Combination Table")
    print("=" * 110 + "\n")

    print(f"{'Model Type':<15} {'Config Type':<20} {'AutoRound (dynamic class)':<40} {'Base classes':<35}")
    print("-" * 110)

    # LLM combinations
    print(f"{'LLM':<15} {'SignRoundConfig':<20} {'CalibCompressor':<40} {'CalibCompressor':<35}")
    print(f"{'LLM':<15} {'RTNConfig':<20} {'CalibratedRTNCompressor':<40} {'CalibratedRTNCompressor':<35}")
    print(f"{'LLM':<15} {'RTNConfig':<20} {'ZeroShotCompressor':<40} {'ZeroShotCompressor':<35}")

    print()

    # MLLM combinations (dynamic classes created in entry.py)
    print(f"{'MLLM':<15} {'SignRoundConfig':<20} {'MLLMCalibCompressor':<40} {'MLLMMixin + CalibCompressor':<35}")
    print(
        f"{'MLLM':<15} {'RTNConfig':<20} {'MLLMCalibratedRTNCompressor':<40} "
        f"{'MLLMMixin + CalibratedRTNCompressor':<35}"
    )
    print(f"{'MLLM':<15} {'RTNConfig':<20} {'MLLMZeroShotCompressor':<40} {'MLLMMixin + ZeroShotCompressor':<35}")

    print()

    # Diffusion combinations (dynamic classes created in entry.py)
    print(
        f"{'Diffusion':<15} {'SignRoundConfig':<20} {'DiffusionCalibCompressor':<40} "
        f"{'DiffusionMixin + CalibCompressor':<35}"
    )
    print(
        f"{'Diffusion':<15} {'RTNConfig':<20} {'DiffusionCalibratedRTNCompressor':<40} "
        f"{'DiffusionMixin + CalibratedRTNCompressor':<35}"
    )
    print(
        f"{'Diffusion':<15} {'RTNConfig':<20} {'DiffusionZeroShotCompressor':<40} "
        f"{'DiffusionMixin + ZeroShotCompressor':<35}"
    )

    print("\n" + "=" * 110 + "\n")


def print_mixin_explanation():
    """Print Mixin pattern explanation"""

    print("=" * 110)
    print("Mixin Pattern Explanation")
    print("=" * 110 + "\n")

    print("✨ Core Components:")
    print("-" * 110)
    print("  1. MLLMMixin                 - MLLM features (processor, template, quant_nontext_module, etc.)")
    print("  2. DiffusionMixin            - Diffusion features (pipeline loading, guidance_scale, etc.)")
    print("  3. CalibCompressor           - AutoRoundCompatible: gradient-based calibration quantization")
    print("  4. CalibratedRTNCompressor   - RTN with importance-matrix (imatrix) or act calibration")
    print("  5. ZeroShotCompressor        - Zero-shot RTN (no calibration data needed)")

    print("\n🎯 Combination Approach:")
    print("-" * 110)
    print("  Dynamic classes created on-the-fly inside AutoRound.__new__():")
    print("    class MLLMCalibCompressor(MLLMMixin, CalibCompressor): pass")
    print("    class MLLMCalibratedRTNCompressor(MLLMMixin, CalibratedRTNCompressor): pass")
    print("    class MLLMZeroShotCompressor(MLLMMixin, ZeroShotCompressor): pass")

    print("\n💡 Advantages:")
    print("-" * 110)
    print("  ✓ Flexible Combination: Any model type can be combined with any compression algorithm")
    print("  ✓ Code Reuse: Mixin code is written once and reused across all compression algorithms")
    print("  ✓ Clear Separation: Model-specific logic (Mixin) and compression algorithm are independent")
    print("  ✓ Easy Extension: Add new model types without touching existing compressor code")

    print("\n" + "=" * 110 + "\n")


def print_post_init_flow():
    """Print the post_init execution flow"""

    print("=" * 110)
    print("BaseCompressor.post_init() Execution Flow")
    print("=" * 110 + "\n")

    print("""
BaseCompressor.post_init()
│
├─ Step 1: Resolve formats (str → list[OutputFormat])
│  └─ get_formats(self.formats, self)
│
├─ Step 2: Apply format-specific model patches
│  └─ model_context.apply_patches(formats)
│     ├─ _patch_custom_moe_modules()    # e.g. Qwen3VL MoE top_k fix
│     ├─ update_module(model, formats)  # add gguf_pack_linear etc.
│     └─ assign global_name to all modules
│
├─ Step 3: Setup quantizer on the patched model
│  └─ quantizer = BaseQuantizers.from_config(config)
│     └─ quantizer.post_init()
│        ├─ get ModelContext / CompressContext singletons
│        ├─ _parse_scheme() → resolve final quant attrs
│        ├─ get_block_names(quant_vision=quant_nontext_module)
│        ├─ find_matching_blocks() → quant_block_list
│        └─ back-fill to_quant_block_names if it was None
│
└─ Step 4: Setup device map, torch compile, offloader
    """)

    print("=" * 110 + "\n")


def print_usage_examples():
    """Print usage examples"""

    print("=" * 110)
    print("Usage Examples")
    print("=" * 110 + "\n")

    print("Example 1: MLLM + AutoRoundCompatible (gradient-based)")
    print("-" * 110)
    print("""
from auto_round.compressors_new.entry import AutoRound
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig

config = SignRoundConfig(scheme="W4A16", iters=200, nsamples=128)
compressor = AutoRound(
    config=config,
    model="/models/Qwen2-VL-2B-Instruct",
    processor=processor,
    template="qwen2_vl",
    quant_nontext_module=False,   # set True to also quantize vision encoder
)
# Dynamically creates: class MLLMCalibCompressor(MLLMMixin, CalibCompressor)
    """)

    print("\nExample 2: MLLM + RTN with imatrix")
    print("-" * 110)
    print("""
from auto_round.algorithms.quantization.rtn.config import RTNConfig

config = RTNConfig(scheme="W4A16")
compressor = AutoRound(
    config=config,
    model="/models/Qwen2-VL-2B-Instruct",
    format="gguf_k",    # gguf_k triggers CalibratedRTNCompressor
    processor=processor,
)
# Dynamically creates: class MLLMCalibratedRTNCompressor(MLLMMixin, CalibratedRTNCompressor)
    """)

    print("\nExample 3: Diffusion + AutoRoundCompatible")
    print("-" * 110)
    print("""
config = SignRoundConfig(scheme="W4A16", iters=200)
compressor = AutoRound(
    config=config,
    model="/models/stable-diffusion-2-1",
    guidance_scale=7.5,
)
# Dynamically creates: class DiffusionCalibCompressor(DiffusionMixin, CalibCompressor)
    """)

    print("\n" + "=" * 110 + "\n")


def print_mro_example():
    """Print MRO (Method Resolution Order) example"""

    print("=" * 110)
    print("Method Resolution Order (MRO) Example")
    print("=" * 110 + "\n")

    print("For class MLLMCalibCompressor(MLLMMixin, CalibCompressor):")
    print("-" * 110)
    print("""
MLLMCalibCompressor  (dynamic, created in AutoRound.__new__)
    └─> MLLMMixin
        └─> CalibCompressor
            └─> BaseCompressor
                └─> object

Execution order when calling __init__():
  1. MLLMCalibCompressor.__init__()  → not defined, falls through
  2. MLLMMixin.__init__()
     - Save MLLM-specific attrs: processor, template, quant_nontext_module, …
     - kwargs.setdefault("quant_nontext_module", quant_nontext_module)
     - Call super().__init__() → enters CalibCompressor
  3. CalibCompressor.__init__() → BaseCompressor.__init__()
     - pops quant_nontext_module from kwargs
     - Creates ModelContext(…, quant_nontext_module=quant_nontext_module)
     - ModelContext.__init__ eagerly loads the model
     - Creates CompressContext singleton

MLLMCalibCompressor instance has:
  ✓ MLLM features from MLLMMixin (processor, template, calib() override)
  ✓ Calibration compression from CalibCompressor
  ✓ Model/context management from BaseCompressor
    """)

    print("=" * 110 + "\n")


def print_decision_tree():
    """Print decision tree"""

    print("=" * 110)
    print("AutoRound Creation Decision Tree")
    print("=" * 110 + "\n")

    print("""
AutoRound.__new__(config, model, format, **kwargs)
│
├─ Step 1: Detect model type
│  model_type = detect_model_type(model)
│  ├─ is_diffusion_model() → "diffusion"
│  ├─ is_mllm_model()      → "mllm"
│  └─ else                 → "llm"
│
├─ isinstance(config, SignRoundConfig)
│  ├─ model_type == "mllm"
│  │  └─> class MLLMCalibCompressor(MLLMMixin, CalibCompressor)
│  ├─ model_type == "diffusion"
│  │  └─> class DiffusionCalibCompressor(DiffusionMixin, CalibCompressor)
│  └─ model_type == "llm"
│     └─> CalibCompressor
│
└─ isinstance(config, RTNConfig)
   │
   ├─ enable_imatrix OR needs_act_calib  → CalibratedRTNCompressor path
   │  ├─ gguf_k format → enable_imatrix = True
   │  ├─ symmetric int RTN → enable_imatrix = True
   │  ├─ static activation quantization → needs_act_calib = True
   │  │
   │  ├─ model_type == "mllm"
   │  │  └─> class MLLMCalibratedRTNCompressor(MLLMMixin, CalibratedRTNCompressor)
   │  ├─ model_type == "diffusion"
   │  │  └─> class DiffusionCalibratedRTNCompressor(DiffusionMixin, CalibratedRTNCompressor)
   │  └─ model_type == "llm"
   │     └─> CalibratedRTNCompressor
   │
   └─ else (zero-shot)  → ZeroShotCompressor path
      ├─ model_type == "mllm"
      │  └─> class MLLMZeroShotCompressor(MLLMMixin, ZeroShotCompressor)
      ├─ model_type == "diffusion"
      │  └─> class DiffusionZeroShotCompressor(DiffusionMixin, ZeroShotCompressor)
      └─ model_type == "llm"
         └─> ZeroShotCompressor
    """)

    print("=" * 110 + "\n")


def print_quantizer_interface():
    """Print the BaseQuantizers interface contract"""

    print("=" * 110)
    print("BaseQuantizers Interface - Name-based quantize_block / quantize_layer")
    print("=" * 110 + "\n")

    print("""
All quantizers use module *names* (str) instead of module objects.
The module is retrieved internally via get_module(model, name).

  BaseQuantizers (abstract)
  ├─ quantize_block(block_name: Union[str, list[str]], input_ids, input_others, **kwargs)
  │    str       → get_module(model, block_name)
  │    list[str] → WrapperMultiblock([get_module(model, n) for n in block_name])
  │                (used when nblocks > 1 in CalibCompressor)
  │
  └─ quantize_layer(layer_name: str, **kwargs)
       → get_module(model, layer_name)

  Implementations:
  ├─ RTNQuantizer.quantize_block(block_name: str)
  ├─ OptimizedRTNQuantizer.quantize_block(block_name: str, input_ids, input_others)
  └─ SignRoundQuantizer.quantize_block(block_name: Union[str, list[str]], input_ids, input_others)
    """)

    print("=" * 110 + "\n")


def main():
    """Run all visualizations"""

    print_architecture_table()
    print_mixin_explanation()
    print_post_init_flow()
    print_usage_examples()
    print_mro_example()
    print_decision_tree()
    print_quantizer_interface()

    print("=" * 110)
    print("🎉 New architecture supports 9 combinations (3 model types × 3 compression algorithms)")
    print("    CalibratedRTNCompressor (was ImatrixCompressor) lives in calib.py")
    print("=" * 110)


if __name__ == "__main__":
    main()

    print(f"{'LLM':<15} {'RTNConfig':<20} {'RTN (zero-shot)':<20} {'ZeroShotCompressor':<35}")

    print()

    # MLLM combinations
    print(f"{'MLLM':<15} {'SignRoundConfig':<20} {'AutoRoundCompatible':<20} {'MLLMCalibCompressor':<35}")
    print(f"{'':<15} {'':<20} {'':<20} {'  = MLLMMixin + CalibCompressor':<35}")
    print(f"{'MLLM':<15} {'RTNConfig':<20} {'RTN + imatrix':<20} {'MLLMImatrixCompressor':<35}")
    print(f"{'':<15} {'':<20} {'':<20} {'  = MLLMMixin + ImatrixCompressor':<35}")
    print(f"{'MLLM':<15} {'RTNConfig':<20} {'RTN (zero-shot)':<20} {'MLLMZeroShotCompressor':<35}")
    print(f"{'':<15} {'':<20} {'':<20} {'  = MLLMMixin + ZeroShotCompressor':<35}")

    print()

    # Diffusion combinations
    print(f"{'Diffusion':<15} {'SignRoundConfig':<20} {'AutoRoundCompatible':<20} {'DiffusionCalibCompressor':<35}")
    print(f"{'':<15} {'':<20} {'':<20} {'  = DiffusionMixin + CalibCompressor':<35}")
    print(f"{'Diffusion':<15} {'RTNConfig':<20} {'RTN + imatrix':<20} {'DiffusionImatrixCompressor':<35}")
    print(f"{'':<15} {'':<20} {'':<20} {'  = DiffusionMixin + ImatrixCompressor':<35}")
    print(f"{'Diffusion':<15} {'RTNConfig':<20} {'RTN (zero-shot)':<20} {'DiffusionZeroShotCompressor':<35}")
    print(f"{'':<15} {'':<20} {'':<20} {'  = DiffusionMixin + ZeroShotCompressor':<35}")

    print("\n" + "=" * 100 + "\n")
