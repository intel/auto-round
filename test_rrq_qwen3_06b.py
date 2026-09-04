#!/usr/bin/env python
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Quantize Qwen3-0.6B with RRQ (packed-INT2 2+2+2+2) and verify the split-save.

This script:
     1. Runs the RRQ quantization on ``Qwen/Qwen3-0.6B`` (RTN when
         ``iters=0``; calibrated sign-SGD tuning when ``iters>0``).
    2. Saves the **base** model (standard INT2 ``auto_round``) and the
       **residual** model (3 packed-INT2 planes, ``auto_round:rrq``) to two
       separate directories.
    3. Verifies the on-disk layout:
        - base dir   -> ``.qweight`` / ``.scales`` / ``.qzeros`` + full model
                        (embeddings / layernorms / ...), quant_method "auto-round"
        - residual   -> only ``.qweight_{1,2,3}`` / ``.scales_{...}`` /
                        ``.qzeros_{...}``, NO base ``.qweight``, NO non-RRQ
                        weights, quant_method "auto-round-rrq"
    4. Reports per-directory sizes and sanity-checks that they are reasonable
       (both far smaller than the original fp model, residual a few x the base).

Run:
    # RTN baseline
    python test_rrq_qwen3_06b.py --iters 0 --out ./rrq_output/rtn
    # Phase 3: 50 sign-SGD iterations per plane
    python test_rrq_qwen3_06b.py --iters 50 --lr 0.05 --out ./rrq_output/opt50
    # Verify an existing result without re-quantizing
    python test_rrq_qwen3_06b.py --skip-quant --out ./rrq_output/opt50
    # Also load and generate with the result
    python test_rrq_qwen3_06b.py --skip-quant --verify-load --out ./rrq_output/opt50
"""

import argparse
import json
import os
import sys


def _human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024.0 or unit == "TB":
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.2f} TB"


def _dir_total_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def _list_files(path: str):
    rows = []
    for root, _dirs, files in os.walk(path):
        for f in sorted(files):
            fp = os.path.join(root, f)
            rows.append((os.path.relpath(fp, path), os.path.getsize(fp)))
    rows.sort(key=lambda r: r[0])
    return rows


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safetensors_keys(path: str):
    """Return (num_tensors, total_bytes, {key: (shape, dtype)}) for a dir's weights.

    Handles both a single ``model.safetensors`` and sharded files with an index.
    """
    from safetensors import safe_open

    single = os.path.join(path, "model.safetensors")
    index = os.path.join(path, "model.safetensors.index.json")

    if os.path.exists(index):
        with open(index, "r", encoding="utf-8") as f:
            wmap = json.load(f)["weight_map"]
        shard_files = sorted(set(wmap.values()))
        files = [os.path.join(path, s) for s in shard_files]
    elif os.path.exists(single):
        files = [single]
    else:
        print(f"  [warn] no weight file found in {path}")
        return 0, 0, {}

    info = {}
    total_bytes = 0
    for fp in files:
        with safe_open(fp, framework="pt") as st:
            for k in st.keys():
                t = st.get_tensor(k)
                info[k] = (tuple(t.shape), str(t.dtype))
                total_bytes += t.numel() * t.element_size()
    return len(info), total_bytes, info


# ─────────────────────────────────────────────────────────────────────────────
# Quantization
# ─────────────────────────────────────────────────────────────────────────────
def run_quantize(
    model_name: str,
    out_root: str,
    group_size: int,
    sym: bool,
    device: str,
    iters: int,
    lr: float | None,
    nsamples: int,
    seqlen: int,
    batch_size: int,
):
    from auto_round import AutoRound, RRQConfig

    base_dir = os.path.join(out_root, "base")
    residual_dir = os.path.join(out_root, "residual")
    os.makedirs(out_root, exist_ok=True)

    print(f"\n=== Quantizing {model_name} with RRQ ===")
    print(f"    group_size={group_size}  sym={sym}  device={device}")
    print(f"    iters={iters}  lr={lr or 'auto'}  nsamples={nsamples}  seqlen={seqlen}")
    if iters == 0:
        print("    (RRQ RTN baseline: 4 planes of INT2, effective 2/4/6/8-bit)")
    else:
        print("    (RRQ Phase 3: 4 planes of INT2, sign-SGD tuning per plane)")

    cfg = RRQConfig(group_size=group_size, sym=sym, iters=iters, lr=lr)

    ar = AutoRound(
        model_name,
        scheme="W2A16",
        alg_configs=cfg,
        device_map=device,
        low_cpu_mem_usage=True,
        nsamples=nsamples,
        seqlen=seqlen,
        batch_size=batch_size,
        seed=42,
    )
    ar.quantize()

    # Save the residual FIRST (it reads the live rrq_*_k buffers and is
    # non-destructive), then the base (standard auto_round export packs the
    # base layers in-place, which would drop the residual buffers).
    #
    # ``save_quantized`` only re-resolves the ``format`` string when ``ar.formats``
    # is a string; after the first call it latches onto the previous format
    # (stored on the compression plan).  Setting ``ar.formats`` to the target
    # format string before each call forces the getter to return that string so
    # ``save_quantized`` re-resolves it instead of reusing the last format.
    def _save(output_dir: str, fmt: str):
        ar.formats = fmt
        ar.save_quantized(output_dir, format=fmt)

    print(f"\n    saving residual (3 planes) -> {residual_dir}")
    _save(residual_dir, "auto_round:rrq")
    print(f"    saving base (standard INT2)  -> {base_dir}")
    _save(base_dir, "auto_round")

    return base_dir, residual_dir


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────
def verify_dir(path: str, label: str):
    print(f"\n{'=' * 70}\n{label}: {path}\n{'=' * 70}")
    if not os.path.isdir(path):
        print("  [FAIL] directory does not exist")
        return None

    files = _list_files(path)
    total = sum(sz for _, sz in files)
    print(f"  files: {len(files)}   total size: {_human(total)}")
    for name, sz in files:
        print(f"    {_human(sz):>12}  {name}")

    # config / quantization_config
    qmethod = None
    if os.path.exists(os.path.join(path, "quantization_config.json")):
        qcfg = _load_config(os.path.join(path, "quantization_config.json"))
        qmethod = qcfg.get("quant_method")
        print(f"  quantization_config.json -> quant_method: {qmethod}")
    elif os.path.exists(os.path.join(path, "config.json")):
        cfg = _load_config(os.path.join(path, "config.json"))
        qcfg = cfg.get("quantization_config", {})
        qmethod = qcfg.get("quant_method")
        print(f"  config.json.quantization_config -> quant_method: {qmethod}")

    num, wbytes, keys = _safetensors_keys(path)
    print(f"  weight tensors: {num}   weight bytes: {_human(wbytes)}")
    return {
        "files": files,
        "total": total,
        "qmethod": qmethod,
        "num_tensors": num,
        "weight_bytes": wbytes,
        "keys": keys,
    }


def analyze_layout(base_info, residual_info):
    print(f"\n{'=' * 70}\nLAYOUT ANALYSIS (base vs residual split)\n{'=' * 70}")
    ok = True

    base_keys = set(base_info["keys"]) if base_info else set()
    res_keys = set(residual_info["keys"]) if residual_info else set()

    # --- base model checks -------------------------------------------------
    # Expect standard packed INT2 keys: .qweight / .scales / .qzeros, plus the
    # full non-quant model (embeddings, layernorms, final norm, ...).
    has_qweight = any(k.endswith(".qweight") for k in base_keys)
    has_scales = any(k.endswith(".scales") for k in base_keys)
    has_qzeros = any(k.endswith(".qzeros") for k in base_keys)
    # No residual-plane keys (qweight_1..3) should appear in the base.
    base_has_plane = any(
        any(k.endswith(f"{s}_{i}") for s in ("qweight", "scales", "qzeros") for i in (1, 2, 3))
        for k in base_keys
    )
    # Full model: embeddings present.
    has_embed = any("embed" in k.lower() for k in base_keys)
    has_norm = any("norm" in k.lower() for k in base_keys)

    print("  BASE model:")
    for name, cond in (
        (".qweight present", has_qweight),
        (".scales present", has_scales),
        (".qzeros present", has_qzeros),
        ("no residual-plane keys (.qweight_1..3)", not base_has_plane),
        ("embeddings present (full model)", has_embed),
        ("layernorms present (full model)", has_norm),
        (
            'quant_method is a base (non-rrq) method',
            base_info["qmethod"] is not None
            and "auto-round" in base_info["qmethod"]
            and base_info["qmethod"] != "auto-round-rrq"
            if base_info
            else False,
        ),
    ):
        print(f"    [{'ok' if cond else 'XX'}] {name}")
        ok = ok and cond

    # count base quant layers (distinct layer prefixes having .qweight)
    base_layers = {k[:- len(".qweight")] for k in base_keys if k.endswith(".qweight")}
    print(f"    base quantized layers: {len(base_layers)}")

    # --- residual model checks --------------------------------------------
    # Expect ONLY residual planes: .qweight_1/.qweight_2/.qweight_3 etc., with
    # matching .scales_k / .qzeros_k. No base .qweight, no non-RRQ weights.
    res_qweight_planes = {k for k in res_keys if k.endswith((".qweight_1", ".qweight_2", ".qweight_3"))}
    res_scales = [k for k in res_keys if k.endswith((".scales_1", ".scales_2", ".scales_3"))]
    res_qzeros = [k for k in res_keys if k.endswith((".qzeros_1", ".qzeros_2", ".qzeros_3"))]
    res_has_base_qweight = any(k.endswith(".qweight") for k in res_keys)
    # residual should have NO embeddings / layernorms / weight (non-RRQ).
    res_has_embed = any("embed" in k.lower() for k in res_keys)
    res_has_plain_weight = any(k.endswith(".weight") for k in res_keys)

    # every layer that has qweight_1 must also have qweight_2 and qweight_3
    layers_with_1 = {k[: -len(".qweight_1")] for k in res_qweight_planes if k.endswith(".qweight_1")}
    layers_with_3 = {k[: -len(".qweight_3")] for k in res_qweight_planes if k.endswith(".qweight_3")}

    print("  RESIDUAL model:")
    for name, cond in (
        (f".qweight_1/_2/_3 present ({len(res_qweight_planes)} tensors)", len(res_qweight_planes) > 0),
        (f".scales_1/_2/_3 present ({len(res_scales)} tensors)", len(res_scales) > 0),
        (f".qzeros_1/_2/_3 present ({len(res_qzeros)} tensors)", len(res_qzeros) > 0),
        ("3 planes per layer (qweight_1 set == qweight_3 set)", layers_with_1 == layers_with_3),
        ("no base .qweight", not res_has_base_qweight),
        ("no embeddings (residual-only)", not res_has_embed),
        ("no plain .weight (residual-only)", not res_has_plain_weight),
        ('quant_method == "auto-round-rrq"', residual_info["qmethod"] == "auto-round-rrq" if residual_info else False),
    ):
        print(f"    [{'ok' if cond else 'XX'}] {name}")
        ok = ok and cond

    # residual layer count should match base quantized layer count
    res_layers = {k[: -len(".qweight_1")] for k in res_qweight_planes if k.endswith(".qweight_1")}
    print(f"    residual layers (by qweight_1): {len(res_layers)}")
    if base_info and residual_info:
        match = len(res_layers) == len(base_layers)
        print(f"    [{'ok' if match else 'XX'}] residual layer count == base quant layer count")
        ok = ok and match

    # --- sample tensor shapes ----------------------------------------------
    if res_keys:
        sample = sorted(res_keys)[0]
        shape, dtype = residual_info["keys"][sample]
        print(f"  sample residual tensor: {sample} shape={shape} dtype={dtype}")
    if base_keys:
        base_qweight = sorted(k for k in base_keys if k.endswith(".qweight"))
        if base_qweight:
            sample = base_qweight[0]
            shape, dtype = base_info["keys"][sample]
            print(f"  sample base tensor:     {sample} shape={shape} dtype={dtype}")
        else:
            sample = sorted(base_keys)[0]
            shape, dtype = base_info["keys"][sample]
            print(f"  sample base tensor:     {sample} shape={shape} dtype={dtype}  (no .qweight found)")

    return ok, len(base_layers), len(res_layers)


def analyze_sizes(base_info, residual_info, orig_bytes: float):
    print(f"\n{'=' * 70}\nSIZE ANALYSIS\n{'=' * 70}")
    ok = True
    b = base_info["total"] if base_info else 0
    r = residual_info["total"] if residual_info else 0
    print(f"  original fp model : {_human(orig_bytes)}  (reference)")
    print(f"  base dir          : {_human(b)}")
    print(f"  residual dir      : {_human(r)}")
    if orig_bytes > 0:
        print(f"  base vs original    : {b / orig_bytes:.2%}")
        print(f"  residual vs original: {r / orig_bytes:.2%}")
    if b > 0 and r > 0:
        print(f"  residual / base       : {r / b:.2f}x")

    # Both outputs must be far smaller than the original fp model.
    cond1 = orig_bytes > 0 and b < 0.5 * orig_bytes
    print(f"    [{'ok' if cond1 else 'XX'}] base < 50% of original fp size")
    cond2 = orig_bytes > 0 and r < 0.9 * orig_bytes
    print(f"    [{'ok' if cond2 else 'XX'}] residual < 90% of original fp size")
    # The residual holds 3x the weight planes of the base (2-bit each), so it
    # should be a few times the base quant portion (allow a wide range since
    # the base also carries float embeddings/layernorms).
    if b > 0:
        ratio = r / b
        cond3 = 0.3 <= ratio <= 4.0
        print(f"    [{'ok' if cond3 else 'XX'}] residual/base ratio in [0.3, 4.0] (got {ratio:.2f}x)")
    else:
        cond3 = True
    ok = cond1 and cond2 and cond3
    return ok


def optional_load_verify(base_dir, residual_dir, device: str):
    print(f"\n{'=' * 70}\nOPTIONAL: load_rrq_model + forward smoke test\n{'=' * 70}")
    import torch
    import transformers

    from auto_round import load_rrq_model
    from auto_round.inference.rrq_linear import set_rrq_bits

    try:
        model = load_rrq_model(base_dir, residual_dir, active_bits=8, device=device)
        print("  load_rrq_model OK")

        n_rrq = sum(1 for m in model.modules() if type(m).__name__ == "RRQLinear")
        print(f"  RRQLinear layers: {n_rrq}")
        assert n_rrq > 0, "no RRQLinear layers found"

        tokenizer = transformers.AutoTokenizer.from_pretrained(base_dir)
        prompt = "The capital of France is"
        inputs = tokenizer(prompt, return_tensors="pt")
        if device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        model.eval()
        with torch.inference_mode():
            out8 = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        set_rrq_bits(model, 4)
        with torch.inference_mode():
            out4 = model.generate(**inputs, max_new_tokens=8, do_sample=False)

        print("  8-bit output:", tokenizer.decode(out8[0], skip_special_tokens=True))
        print("  4-bit output:", tokenizer.decode(out4[0], skip_special_tokens=True))
        print("  [ok] load + forward (8-bit and 4-bit) succeeded")
        return True
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"  [warn] load/forward check failed (non-fatal): {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description="RRQ quantize + verify Qwen3-0.6B")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--sym", action="store_true", help="symmetric (default: asymmetric)")
    ap.add_argument("--device", default="cpu", help="cpu / cuda / xpu")
    ap.add_argument("--out", default="./rrq_output")
    ap.add_argument(
        "--iters",
        type=int,
        default=0,
        help="Sign-SGD iterations per plane; 0 selects the RTN baseline, >0 enables Phase 3.",
    )
    ap.add_argument("--lr", type=float, default=None, help="Sign-SGD learning rate; default is auto.")
    ap.add_argument("--nsamples", type=int, default=128, help="Number of calibration samples for iters>0.")
    ap.add_argument("--seqlen", type=int, default=128, help="Calibration sequence length for iters>0.")
    ap.add_argument("--batch-size", type=int, default=1, help="Calibration batch size for iters>0.")
    ap.add_argument("--skip-quant", action="store_true", help="skip quantize; verify existing outputs")
    ap.add_argument("--verify-load", action="store_true", help="also load via load_rrq_model + forward")
    args = ap.parse_args()

    out_root = args.out
    base_dir = os.path.join(out_root, "base")
    residual_dir = os.path.join(out_root, "residual")

    if not args.skip_quant:
        if args.iters < 0:
            ap.error("--iters must be non-negative")
        base_dir, residual_dir = run_quantize(
            args.model,
            out_root,
            args.group_size,
            args.sym,
            args.device,
            args.iters,
            args.lr,
            args.nsamples,
            args.seqlen,
            args.batch_size,
        )
    else:
        print(f"\n=== Skipping quantize; verifying existing outputs in {out_root} ===")

    # Reference: original fp model size (best-effort, from HF config / cache).
    orig_bytes = _try_original_size(args.model)

    base_info = verify_dir(base_dir, "BASE MODEL")
    res_info = verify_dir(residual_dir, "RESIDUAL MODEL")

    if base_info is None or res_info is None:
        print("\n[FAIL] one of the output directories is missing")
        sys.exit(1)

    layout_ok, n_base_layers, n_res_layers = analyze_layout(base_info, res_info)
    size_ok = analyze_sizes(base_info, res_info, orig_bytes)

    load_ok = True
    if args.verify_load:
        load_ok = optional_load_verify(base_dir, residual_dir, args.device)

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(f"  base quant layers    : {n_base_layers}")
    print(f"  residual layers      : {n_res_layers}")
    print(f"  layout check         : {'PASS' if layout_ok else 'FAIL'}")
    print(f"  size check           : {'PASS' if size_ok else 'FAIL'}")
    if args.verify_load:
        print(f"  load/forward check   : {'PASS' if load_ok else 'FAIL'}")

    all_ok = layout_ok and size_ok and load_ok
    print(f"\n  OVERALL: {'PASS ✅' if all_ok else 'FAIL ❌'}")
    sys.exit(0 if all_ok else 1)


def _try_original_size(model_name: str) -> float:
    """Best-effort original fp model size in bytes (0 if unknown)."""
    try:
        import os

        from huggingface_hub import snapshot_download

        path = snapshot_download(model_name)
        total = 0
        for root, _d, files in os.walk(path):
            for f in files:
                if f.endswith((".safetensors", ".bin")):
                    total += os.path.getsize(os.path.join(root, f))
        print(f"\n  reference original fp model ({model_name}): {_human(total)}")
        return float(total)
    except Exception as e:  # noqa: BLE001
        print(f"\n  [warn] could not compute original model size ({e}); skipping size-vs-original checks")
        return 0.0


if __name__ == "__main__":
    main()
