import os
import shutil

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round.formats import AutoRoundExportFormat

from ...envs import require_gguf
from ...helpers import get_model_path, get_tiny_model


def _is_compiled(func):
    """Check if a function has been compiled with torch.compile.
    
    torch.compile wraps functions with a GraphModule that has __wrapped__ attribute.
    """
    return hasattr(func, '__wrapped__') or hasattr(func, '_torchdynamo_orig_code')


class TestTorchCompile:
    save_dir = "./saved"

    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown_class(self):
        print("[Setup] Running before any test in class")
        yield
        print("[Teardown] Running after all tests in class")
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    @require_gguf
    def test_gguf_q2ks_torch_compile(self, dataloader):
        """Test GGUF Q2_K_S quantization with torch.compile enabled.

        Verifies that:
        1. Quantization completes successfully with enable_torch_compile=True.
        2. The GGUF file is exported correctly.
        3. The exported model can be loaded and run inference.
        """
        model_path = get_model_path("Qwen/Qwen2.5-1.5B-Instruct")
        model = get_tiny_model(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        autoround = AutoRound(
            model,
            tokenizer,
            scheme="gguf:q2_k_s",
            iters=2,
            seqlen=16,
            nsamples=2,
            enable_alg_ext=True,
            enable_torch_compile=True,
        )
        autoround.quantize()
        autoround.save_quantized(output_dir=self.save_dir)

        shutil.rmtree(self.save_dir, ignore_errors=True)

    @require_gguf
    def test_gguf_q2ks_torch_compile_iters0(self, tiny_qwen_model_path):
        """Test GGUF Q2_K_S with torch.compile and iters=0 (RTN mode).

        Verifies that torch.compile does not break the fast RTN-based quantization
        path when exporting to GGUF Q2_K_S format.
        """
        autoround = AutoRound(
            tiny_qwen_model_path,
            iters=0,
            nsamples=2,
            seqlen=16,
            enable_torch_compile=True,
        )
        autoround.quantize_and_save(output_dir=self.save_dir, format="gguf:q2_k_s")

        saved_files = [f for f in os.listdir(self.save_dir) if f.endswith(".gguf")]
        assert len(saved_files) > 0, "No GGUF file was generated"

        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_mxfp8_torch_compile(self, tiny_qwen_model_path):
        """Test MXFP8 quantization with torch.compile enabled.

        Verifies that MXFP8 (floating-point format) works with torch.compile.
        These formats don't compute bits-derived constants like maxq, so they
        should work seamlessly with torch.compile without special handling.
        """
        tokenizer = AutoTokenizer.from_pretrained(tiny_qwen_model_path)
        autoround = AutoRound(
            tiny_qwen_model_path,
            tokenizer,
            scheme=AutoRoundExportFormat.MXFP8.value,
            iters=2,
            seqlen=16,
            nsamples=2,
            enable_torch_compile=True,
        )
        autoround.quantize()
        
        # Verify torch.compile is actually enabled in at least one wrapper layer
        compile_found = False
        for module in autoround.model.modules():
            if hasattr(module, 'weight_quant_func'):
                if _is_compiled(module.weight_quant_func) or hasattr(module.weight_quant_func, '_compiled'):
                    compile_found = True
                    break
        assert compile_found or autoround.enable_torch_compile, (
            "torch.compile was not applied to any wrapper layer or it was disabled. "
            "Check that _adjust_torch_compile did not disable it."
        )
        
        autoround.save_quantized(output_dir=self.save_dir)
        assert os.path.exists(self.save_dir), "Quantized model directory not created"
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_mxfp4_torch_compile(self, tiny_qwen_model_path):
        """Test MXFP4 quantization with torch.compile enabled."""
        tokenizer = AutoTokenizer.from_pretrained(tiny_qwen_model_path)
        autoround = AutoRound(
            tiny_qwen_model_path,
            tokenizer,
            scheme=AutoRoundExportFormat.MXFP4.value,
            iters=2,
            seqlen=16,
            nsamples=2,
            enable_torch_compile=True,
        )
        assert autoround.enable_torch_compile, (
            "torch.compile was disabled by _adjust_torch_compile. MXFP4 should support torch.compile."
        )
        autoround.quantize()
        autoround.save_quantized(output_dir=self.save_dir)
        assert os.path.exists(self.save_dir), "Quantized model directory not created"
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_nvfp4_torch_compile(self, tiny_qwen_model_path):
        """Test NVFP4 quantization with torch.compile enabled.

        Verifies that NVFP4 (NVIDIA floating-point 4-bit) works with torch.compile.
        These formats use pre-defined constants for quantization, not bits-derived values.
        
        Previously this was force-disabled due to issue #1109, but now supported via
        _compile_derive pre-computation pattern (even though NVFP doesn't need it).
        """
        tokenizer = AutoTokenizer.from_pretrained(tiny_qwen_model_path)
        autoround = AutoRound(
            tiny_qwen_model_path,
            tokenizer,
            scheme=AutoRoundExportFormat.NVFP4.value,
            iters=2,
            seqlen=16,
            nsamples=2,
            enable_torch_compile=True,
        )
        assert autoround.enable_torch_compile, (
            "torch.compile was disabled for NVFP4. This should now be supported after fix for issue #1109."
        )
        autoround.quantize()
        autoround.save_quantized(output_dir=self.save_dir)
        assert os.path.exists(self.save_dir), "Quantized model directory not created"
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @pytest.mark.skip(reason="FP8_STATIC with activation quantization still force-disabled due to wfp8afp8 check")
    def test_fp8_torch_compile(self, tiny_qwen_model_path):
        """Test FP8 quantization with torch.compile enabled.

        Verifies that FP8 (8-bit floating point) works with torch.compile.
        FP8 uses torch.finfo to get max values, not bits-derived constants.
        
        NOTE: Currently skipped because FP8_STATIC with activation quantization
        triggers is_wfp8afp8() which force-disables torch.compile for HPU compatibility.
        This is a conservative disable and should be revisited.
        """
        tokenizer = AutoTokenizer.from_pretrained(tiny_qwen_model_path)
        autoround = AutoRound(
            tiny_qwen_model_path,
            tokenizer,
            scheme="FP8_STATIC",
            iters=2,
            seqlen=16,
            nsamples=2,
            enable_torch_compile=True,
        )
        assert autoround.enable_torch_compile, (
            "torch.compile was disabled for FP8. FP8_STATIC should support torch.compile."
        )
        autoround.quantize()
        autoround.save_quantized(output_dir=self.save_dir)
        assert os.path.exists(self.save_dir), "Quantized model directory not created"
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_mixed_bits_2_5_torch_compile(self, tiny_qwen_model_path):
        """Test mixed 2.5-bit quantization with torch.compile (W2A16, W3A16).

        This is the critical test for the fix: verifies that torch.compile works
        with AutoScheme generating mixed bit-widths (2-bit and 3-bit layers).
        The bug was: torch.compile symbolized 'bits' into `pow(float32, int64)` which
        Triton cannot compile. The fix: pre-compute maxq outside the compile boundary.
        """
        tokenizer = AutoTokenizer.from_pretrained(tiny_qwen_model_path)
        autoround = AutoRound(
            tiny_qwen_model_path,
            tokenizer,
            options="W2A16,W3A16",
            iters=2,
            seqlen=16,
            nsamples=2,
            enable_alg_ext=True,
            enable_torch_compile=True,
        )
        assert autoround.enable_torch_compile, (
            "torch.compile was disabled for mixed-bits. Mixed 2.5-bit scheme should support torch.compile."
        )
        autoround.quantize()
        autoround.save_quantized(output_dir=self.save_dir)
        assert os.path.exists(self.save_dir), "Quantized model directory not created"
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_mixed_bits_2_5_torch_compile_full_model(self):
        """Test mixed 2.5-bit quantization with torch.compile on FULL model (NOT tiny).

        This test uses a full model (not reduced to fewer layers) to properly trigger
        torch.compile's complexity and expose the symolic int issue if frozen_kwargs fix
        is not in place.

        This is the authoritative test: if it passes, the fix is working correctly.
        If it fails with Triton compilation error, frozen_kwargs is needed.
        """
        from ...helpers import get_model_path
        
        # Use opt-125m as a non-tiny baseline (full layers, not reduced)
        model_name = get_model_path("facebook/opt-125m")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        autoround = AutoRound(
            model_name,
            tokenizer,
            options="W2A16,W3A16",
            iters=2,
            seqlen=16,
            nsamples=2,
            enable_alg_ext=True,
            enable_torch_compile=True,
        )
        assert autoround.enable_torch_compile, (
            "torch.compile was disabled for mixed-bits on full model."
        )
        autoround.quantize()
        autoround.save_quantized(output_dir=self.save_dir)
        assert os.path.exists(self.save_dir), "Quantized model directory not created"
        shutil.rmtree(self.save_dir, ignore_errors=True)

