#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end example for the ARK XPU int3 (S3) symmetric WOQ GEMV kernel.

Pipeline:
  1. Quantize Qwen/Qwen3-0.6B to W3G128 symmetric (auto_round / GPTQ-style 3-bit packing).
  2. Build TWO copies of the model, both loaded from the same exported int3 checkpoint:
       - ARK model: each quantized nn.Linear swapped for an ARK int3 QuantLinear (the kernel).
       - Dequant reference: each quantized nn.Linear's weight replaced by the plain-PyTorch
         dequantization (q - 4) * scale of the same int3 tensors.
  3. Run a SINGLE-TOKEN forward through both (so every projection sees m==1 -> the S3 GEMV
     path) and check that the ARK kernel matches the dequant reference.

The kernel-correctness check is ARK-int3 vs dequant-of-the-same-int3-weights: it isolates the
GEMV kernel from quantization loss. Comparing against the fp16 model instead would measure how
lossy 3-bit quantization is (large, and unrelated to whether the kernel is correct), so the fp16
next-token is printed for context only, not asserted.

int3 on XPU is GEMV-only (m==1). A normal multi-token prefill would hit the kernel's
m>1 guard (std::abort), so this script deliberately drives the model one token at a time.

Run (from a venv with torch-xpu + auto_round + transformers + the compiled
auto_round_kernel_xpu module on PYTHONPATH):

    ONEAPI_DEVICE_SELECTOR=level_zero:gpu python test_int3_e2e.py
"""

import os
import sys

import torch

MODEL_ID = os.environ.get("ARK_E2E_MODEL", "Qwen/Qwen3-0.6B")
EXPORT_DIR = os.environ.get("ARK_E2E_EXPORT", "/tmp/qwen3-w3g128-sym")
BITS = 3
GROUP_SIZE = 128


def _skip(msg):
    print(f"[int3-e2e] SKIP: {msg}")
    sys.exit(0)


def ensure_quantized():
    """Quantize + export the W3G128-sym model if not already on disk. Returns the dir."""
    # The exporter writes into a subfolder named after the model + scheme.
    for root, _dirs, files in os.walk(EXPORT_DIR):
        if "config.json" in files and "model.safetensors" in files:
            print(f"[int3-e2e] reusing export at {root}")
            return root

    from auto_round import AutoRound

    print(f"[int3-e2e] quantizing {MODEL_ID} -> W{BITS}G{GROUP_SIZE} sym (RTN) ...")
    ar = AutoRound(MODEL_ID, bits=BITS, group_size=GROUP_SIZE, sym=True, iters=0, disable_opt_rtn=True)
    ar.quantize_and_save(output_dir=EXPORT_DIR, format="auto_round")
    for root, _dirs, files in os.walk(EXPORT_DIR):
        if "config.json" in files and "model.safetensors" in files:
            return root
    raise RuntimeError(f"export not found under {EXPORT_DIR}")


def load_packed_tensors(ckpt_dir):
    """Return {module_name: {"qweight","qzeros","scales"}} from the exported safetensors."""
    from safetensors import safe_open

    path = os.path.join(ckpt_dir, "model.safetensors")
    packed = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            for suffix in ("qweight", "qzeros", "scales"):
                if key.endswith("." + suffix):
                    name = key[: -len("." + suffix)]
                    packed.setdefault(name, {})[suffix] = f.get_tensor(key)
    # keep only fully-quantized linears
    return {k: v for k, v in packed.items() if {"qweight", "qzeros", "scales"} <= set(v)}


def swap_to_ark_int3(model, packed, device):
    """Replace each quantized nn.Linear with an ARK int3 QuantLinear loaded from `packed`."""
    from auto_round_kernel.qlinear import QuantLinear, ark_post_init

    replaced = 0
    for name, tensors in packed.items():
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        attr = name.rsplit(".", 1)[1]
        ref = getattr(parent, attr)
        in_features, out_features = ref.in_features, ref.out_features
        has_bias = getattr(ref, "bias", None) is not None

        ql = QuantLinear(
            bits=BITS,
            group_size=GROUP_SIZE,
            sym=True,
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
            weight_dtype=torch.float16,
        )
        ql.qweight = tensors["qweight"].to(torch.int32).contiguous()
        ql.qzeros = tensors["qzeros"].to(torch.int32).contiguous()
        ql.scales = tensors["scales"].to(torch.float16).contiguous()
        if has_bias:
            ql.bias = ref.bias.detach().to(torch.float32).contiguous()
        ql = ql.to(device)
        setattr(parent, attr, ql)
        replaced += 1

    ark_post_init(model)
    print(f"[int3-e2e] swapped {replaced} linears to ARK int3 QuantLinear")
    return model


def build_dequant_reference(model, packed, device):
    """Replace each quantized nn.Linear's weight with plain-torch (q - 4) * scale of the same
    int3 tensors. This is the kernel-independent reference the ARK output is checked against."""
    from auto_round_kernel.qlinear import unpack_3bit_signed

    for name, tensors in packed.items():
        lin = model.get_submodule(name)
        k = tensors["qweight"].shape[0] // 3 * 32
        q = unpack_3bit_signed(tensors["qweight"]).float()  # (k, n), values 0..7
        scales = tensors["scales"].float()
        gs = k // scales.shape[0]
        # symmetric: (q - 4) * scale, then transpose (k, n) -> (out, in) for nn.Linear.
        w = ((q - 4.0) * scales.repeat_interleave(gs, dim=0)).t().contiguous()
        lin.weight.data = w.to(torch.float16)
    return model.to(device).eval()


@torch.no_grad()
def main():
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        _skip("torch.xpu not available")
    try:
        import auto_round_kernel  # noqa: F401
    except Exception as e:  # pragma: no cover
        _skip(f"auto_round_kernel not importable: {e}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "xpu"
    ckpt_dir = ensure_quantized()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Single-token input -> every linear sees m==1 (the S3 GEMV path). Use the last prompt token.
    ids = tok("The capital of France is", return_tensors="pt").input_ids[:, -1:].to(device)

    packed = load_packed_tensors(ckpt_dir)
    print(f"[int3-e2e] found {len(packed)} quantized linears in checkpoint")

    # fp16 baseline (informational only — quantization is lossy, so this is not asserted).
    fp16_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device).eval()
    fp16_logits = fp16_model(ids).logits[:, -1, :].float()
    del fp16_model
    torch.xpu.empty_cache()

    # ARK int3 model (the kernel under test).
    ark_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).eval().to(device)
    ark_model = swap_to_ark_int3(ark_model, packed, device)
    ark_logits = ark_model(ids).logits[:, -1, :].float()
    del ark_model
    torch.xpu.empty_cache()

    # Dequant reference: same int3 weights, plain-torch matmul (no ARK kernel).
    dq_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).eval()
    dq_model = build_dequant_reference(dq_model, packed, device)
    dq_logits = dq_model(ids).logits[:, -1, :].float()
    torch.xpu.synchronize()

    fp16_top = int(fp16_logits.argmax(-1))
    ark_top = int(ark_logits.argmax(-1))
    dq_top = int(dq_logits.argmax(-1))

    # Kernel correctness: ARK int3 vs dequant of the SAME int3 weights.
    cos = torch.nn.functional.cosine_similarity(ark_logits, dq_logits, dim=-1).item()
    mse = torch.mean((ark_logits - dq_logits) ** 2).item()

    print(f"[int3-e2e] fp16 next-token (context): {fp16_top!r} -> {tok.decode([fp16_top])!r}")
    print(f"[int3-e2e] ARK  int3 next-token:      {ark_top!r} -> {tok.decode([ark_top])!r}")
    print(f"[int3-e2e] dequant-ref next-token:    {dq_top!r} -> {tok.decode([dq_top])!r}")
    print(f"[int3-e2e] ARK vs dequant-ref: cosine-sim={cos:.6f}  mse={mse:.3e}")

    if ark_top != dq_top or cos < 0.999:
        raise SystemExit(f"[int3-e2e] FAIL: ARK kernel disagrees with dequant ref (cos={cos:.6f}, top {ark_top} vs {dq_top})")
    print("[int3-e2e] PASS: ARK int3 GEMV matches the dequant reference end-to-end")


if __name__ == "__main__":
    main()
