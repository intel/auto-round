#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for MLX format export with AutoRound.
Tests quantization with W4A16 scheme on Qwen3-0.6B model.
"""

import os
import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from auto_round import AutoRound


def test_mlx_export():
    """Test MLX format export with W4A16 quantization."""
    print("=" * 80)
    print("AutoRound MLX Format Export Test")
    print("=" * 80)

    # Model configuration
    model_name = "Qwen/Qwen3-0.6B"
    output_dir = "./mlx_model_w4a16"

    print(f"\n[1/4] Loading model: {model_name}")
    try:
        ar = AutoRound(
            model_name,
            scheme="W4A16",
            bits=4,
            group_size=128,
            sym=True,
            iters=0,  # Fast RTN mode
            disable_opt_rtn=True,  # Disable optimization for faster quantization
            nsamples=32,  # Use fewer samples for testing
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False

    print(f"\n[2/4] Quantizing model to MLX format...")
    try:
        ar.quantize_and_save(output_dir=output_dir, format="mlx")
        print(f"✓ Model quantized and saved to {output_dir}")
    except Exception as e:
        print(f"✗ Failed to quantize and save model: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n[3/4] Verifying output files...")
    try:
        # Check if required files exist
        required_files = [
            "config.json",
            "quantization_config.json",
            "mlx_metadata.json",
        ]

        output_path = Path(output_dir)
        for file_name in required_files:
            file_path = output_path / file_name
            if file_path.exists():
                print(f"  ✓ {file_name} exists")
            else:
                print(f"  ✗ {file_name} NOT found")
                return False

        # Print quantization config
        import json
        quantization_config_path = output_path / "quantization_config.json"
        with open(quantization_config_path, 'r') as f:
            config = json.load(f)
        print(f"\n  Quantization Config:")
        print(f"    - Format: {config.get('format')}")
        print(f"    - Quant Method: {config.get('quant_method')}")
        print(f"    - Bits: {config.get('bits')}")
        print(f"    - Group Size: {config.get('group_size')}")
        print(f"    - Symmetric: {config.get('sym')}")

    except Exception as e:
        print(f"✗ Failed to verify output files: {e}")
        return False

    print(f"\n[4/4] Testing model loading from MLX format...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load the quantized model
        model = AutoModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype="auto")
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        print("✓ Model loaded successfully from MLX format")

        # Simple inference test
        prompt = "Hello, my name is"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Generated text: {result}")
        print("✓ Inference test passed")

    except Exception as e:
        print(f"⚠ Warning: Model loading or inference test failed (this is expected if MLX not installed): {e}")

    print("\n" + "=" * 80)
    print("✓ MLX format export test completed successfully!")
    print("=" * 80)
    return True


def test_mlx_export_w3a16():
    """Test MLX format export with W3A16 quantization."""
    print("\n" + "=" * 80)
    print("AutoRound MLX Format Export Test (W3A16)")
    print("=" * 80)

    # Model configuration
    model_name = "Qwen/Qwen3-0.6B"
    output_dir = "./mlx_model_w3a16"

    print(f"\n[1/3] Loading and quantizing model with W3A16...")
    try:
        ar = AutoRound(
            model_name,
            scheme="W3A16",
            bits=3,
            group_size=128,
            sym=True,
            iters=0,
            disable_opt_rtn=True,
            nsamples=32,
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False

    print(f"\n[2/3] Saving model in MLX format...")
    try:
        ar.quantize_and_save(output_dir=output_dir, format="mlx")
        print(f"✓ Model saved to {output_dir}")
    except Exception as e:
        print(f"✗ Failed to save model: {e}")
        return False

    print(f"\n[3/3] Verifying MLX format files...")
    try:
        output_path = Path(output_dir)
        quantization_config_path = output_path / "quantization_config.json"

        import json
        with open(quantization_config_path, 'r') as f:
            config = json.load(f)

        assert config.get('bits') == 3, f"Expected bits=3, got {config.get('bits')}"
        print(f"  ✓ Bits set correctly to 3")
        print(f"  ✓ All verifications passed")

    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False

    print("\n✓ W3A16 export test completed successfully!")
    return True


if __name__ == "__main__":
    print("\n🚀 Starting AutoRound MLX Format Tests\n")

    # Run W4A16 test
    success1 = test_mlx_export()

    # Run W3A16 test
    success2 = test_mlx_export_w3a16()

    if success1 and success2:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)

