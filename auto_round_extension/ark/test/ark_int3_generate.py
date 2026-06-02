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

"""Token-by-token greedy generation through the ARK XPU int3 (S3) GEMV kernel.

int3 on XPU is GEMV-only (m==1). A normal ``model.generate()`` does a multi-token prefill
(m>1) that would hit the kernel's abort guard, so this script drives the model ONE token at a
time (token-by-token prefill + greedy decode via KV cache) so every linear sees m==1.

It builds the int3 model from an exported W3G128-sym checkpoint (see test_int3_e2e.py for how to
produce one), generates N tokens for a prompt, and cross-checks the ARK kernel against a plain
PyTorch dequant of the same int3 weights by teacher-forcing both over the ARK trajectory (this
removes the greedy-decode butterfly effect, isolating kernel correctness from quantization loss).

Run:
    ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
    ARK_GEN_MODEL=Qwen/Qwen3-4B ARK_GEN_EXPORT=/tmp/qwen3-4b-w3g128-sym \
    python ark_int3_generate.py
"""

import os
import sys

import torch

MODEL_ID = os.environ.get("ARK_GEN_MODEL", "Qwen/Qwen3-0.6B")
EXPORT_DIR = os.environ.get("ARK_GEN_EXPORT", "/tmp/qwen3-w3g128-sym")
PROMPT = os.environ.get("ARK_GEN_PROMPT", "1+1=?")
NEW_TOKENS = int(os.environ.get("ARK_GEN_NEW", "30"))
BITS, GROUP_SIZE = 3, 128


def find_ckpt(root):
    for d, _dirs, files in os.walk(root):
        if "config.json" in files and any(f.endswith(".safetensors") for f in files):
            return d
    raise FileNotFoundError(f"no exported checkpoint under {root}")


def load_packed(ckpt_dir):
    from safetensors import safe_open

    packed = {}
    shards = [f for f in os.listdir(ckpt_dir) if f.endswith(".safetensors")]
    for shard in shards:
        with safe_open(os.path.join(ckpt_dir, shard), framework="pt") as f:
            for key in f.keys():
                for suffix in ("qweight", "qzeros", "scales"):
                    if key.endswith("." + suffix):
                        packed.setdefault(key[: -len("." + suffix)], {})[suffix] = f.get_tensor(key)
    return {k: v for k, v in packed.items() if {"qweight", "qzeros", "scales"} <= set(v)}


def build_ark(packed, device):
    from transformers import AutoModelForCausalLM

    from auto_round_kernel.qlinear import QuantLinear, ark_post_init

    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).eval().to(device)
    for name, t in packed.items():
        parent = m.get_submodule(name.rsplit(".", 1)[0])
        attr = name.rsplit(".", 1)[1]
        ref = getattr(parent, attr)
        ql = QuantLinear(
            bits=BITS, group_size=GROUP_SIZE, sym=True,
            in_features=ref.in_features, out_features=ref.out_features,
            bias=ref.bias is not None, weight_dtype=torch.float16,
        )
        ql.qweight = t["qweight"].int().contiguous()
        ql.qzeros = t["qzeros"].int().contiguous()
        ql.scales = t["scales"].half().contiguous()
        if ref.bias is not None:
            ql.bias = ref.bias.detach().float().contiguous()
        setattr(parent, attr, ql.to(device))
    return ark_post_init(m)


def build_dequant(packed, device):
    from transformers import AutoModelForCausalLM

    from auto_round_kernel.qlinear import unpack_3bit_signed

    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).eval()
    for name, t in packed.items():
        lin = m.get_submodule(name)
        k = t["qweight"].shape[0] // 3 * 32
        q = unpack_3bit_signed(t["qweight"]).float()
        sc = t["scales"].float()
        gs = k // sc.shape[0]
        lin.weight.data = ((q - 4.0) * sc.repeat_interleave(gs, dim=0)).t().contiguous().half()
    return m.to(device).eval()


@torch.no_grad()
def greedy_m1(model, ids, device, n_new):
    """Greedy decode feeding ONE token per forward (m==1) via KV cache."""
    out = ids[0].tolist()
    past = None
    res = None
    for i in range(ids.shape[1]):
        res = model(ids[:, i : i + 1], past_key_values=past, use_cache=True)
        past = res.past_key_values
    nxt = int(res.logits[:, -1, :].argmax(-1))
    out.append(nxt)
    for _ in range(n_new - 1):
        res = model(torch.tensor([[nxt]], device=device), past_key_values=past, use_cache=True)
        past = res.past_key_values
        nxt = int(res.logits[:, -1, :].argmax(-1))
        out.append(nxt)
    return out


@torch.no_grad()
def teacher_forced_logits(model, seq, device):
    past, logits = None, []
    for i in range(seq.shape[1]):
        res = model(seq[:, i : i + 1], past_key_values=past, use_cache=True)
        past = res.past_key_values
        logits.append(res.logits[:, -1, :].float())
    return torch.cat(logits, 0)


@torch.no_grad()
def main():
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("SKIP: torch.xpu not available")
        return
    try:
        import auto_round_kernel  # noqa: F401
    except Exception as e:  # pragma: no cover
        print(f"SKIP: auto_round_kernel not importable: {e}")
        return

    from transformers import AutoTokenizer

    device = "xpu"
    ckpt = find_ckpt(EXPORT_DIR)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    packed = load_packed(ckpt)
    print(f"[gen] model={MODEL_ID}  quantized linears={len(packed)}  new_tokens={NEW_TOKENS}")

    msgs = [{"role": "user", "content": PROMPT}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    prompt_len = ids.shape[1]

    ark = build_ark(packed, device)
    traj = greedy_m1(ark, ids, device, NEW_TOKENS)
    gen_text = tok.decode(traj[prompt_len:], skip_special_tokens=True)
    print(f"[gen] prompt: {PROMPT!r}  ({prompt_len} tokens)")
    print("[gen] ARK int3 generation:")
    print(gen_text)

    # Kernel-correctness cross-check: teacher-force ARK and dequant over ARK's trajectory.
    seq = torch.tensor([traj], device=device)
    la = teacher_forced_logits(ark, seq, device)
    del ark
    torch.xpu.empty_cache()
    dq = build_dequant(packed, device)
    ld = teacher_forced_logits(dq, seq, device)
    torch.xpu.synchronize()

    sl = slice(prompt_len - 1, prompt_len - 1 + NEW_TOKENS)
    a, d = la[sl], ld[sl]
    agree = (a.argmax(-1) == d.argmax(-1)).float().mean().item()
    cos = torch.nn.functional.cosine_similarity(a, d, dim=-1)
    print(
        f"[gen] ARK vs dequant (teacher-forced): argmax-agree={100*agree:.1f}%  "
        f"mean-cos={cos.mean().item():.6f}  min-cos={cos.min().item():.6f}"
    )
    if cos.mean().item() < 0.999:
        sys.exit(f"[gen] FAIL: kernel deviates from dequant ref (mean-cos={cos.mean().item():.6f})")
    print("[gen] PASS: ARK int3 kernel matches the dequant reference along the trajectory")


if __name__ == "__main__":
    main()
