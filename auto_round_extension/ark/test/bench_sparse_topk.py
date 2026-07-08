#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import importlib.util
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_ARK_PARENT = REPO_ROOT / "auto_round_extension" / "ark"
if str(LOCAL_ARK_PARENT) not in sys.path:
    sys.path.insert(0, str(LOCAL_ARK_PARENT))

import auto_round_kernel as ark


def ensure_sparse_binding() -> None:
    local_kernel_dir = REPO_ROOT / "auto_round_extension" / "ark" / "auto_round_kernel"
    current_file = getattr(getattr(ark, "xpu_lib", None), "__file__", None)
    if current_file is not None and hasattr(ark.xpu_lib, "sage_sparse"):
        try:
            if Path(current_file).resolve().is_relative_to(local_kernel_dir.resolve()):
                return
        except Exception:
            pass
    candidates = sorted((local_kernel_dir / "xbuild").glob("auto_round_kernel_xpu*.so"))
    if not candidates:
        raise RuntimeError("Unable to locate built XPU extension with sage_sparse in xbuild/")
    ext_path = candidates[-1]
    module_name = "auto_round_kernel._bench.auto_round_kernel_xpu"
    spec = importlib.util.spec_from_file_location(module_name, ext_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load extension spec from {ext_path}")
    sys.modules.pop(module_name, None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "sage_sparse"):
        raise RuntimeError(f"Loaded extension does not expose sage_sparse: {ext_path}")
    ark.xpu_lib = module


def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        out = fn()
        del out
    torch.xpu.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
        del out
    torch.xpu.synchronize()
    return (time.perf_counter() - start) * 1000.0 / float(iters)


def bench_with_output(fn, warmup: int, iters: int) -> tuple[object, float]:
    out = None
    for _ in range(warmup):
        out = fn()
        del out
    torch.xpu.synchronize()
    start = time.perf_counter()
    for i in range(iters):
        out = fn()
        if i != iters - 1:
            del out
    torch.xpu.synchronize()
    return out, (time.perf_counter() - start) * 1000.0 / float(iters)


def classify_exception(exc: Exception) -> tuple[str, str]:
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    if "out of memory" in lowered or "oom" in lowered:
        return "oom", text
    return "error", text


def empty_xpu_cache() -> None:
    if is_xpu_available():
        torch.xpu.synchronize()
        torch.xpu.empty_cache()


def build_inputs(
    batch: int,
    num_heads_q: int,
    num_heads_kv: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
):
    torch.manual_seed(20260611)
    q = torch.randn((batch, num_heads_q, seq_len, head_dim), dtype=dtype, device=device)
    k = torch.randn((batch, num_heads_kv, seq_len, head_dim), dtype=dtype, device=device)
    v = torch.randn((batch, num_heads_kv, seq_len, head_dim), dtype=dtype, device=device)
    return q, k, v


def attention_dense_flops(batch: int, num_heads_q: int, seq_len_q: int, seq_len_kv: int, head_dim: int) -> float:
    return float(4.0 * batch * num_heads_q * seq_len_q * seq_len_kv * head_dim)


def flops_to_tflops(flops: float, latency_ms: float) -> float:
    return flops / (latency_ms * 1.0e-3) / 1.0e12


def make_row(
    *,
    mode: str,
    batch: int,
    num_heads_q: int,
    num_heads_kv: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    is_causal: bool,
    warmup: int,
    iters: int,
    requested_topk: float | None,
    selected_ratio: float | None,
    selected_blocks_per_row: float | None,
    latency_ms: float | None,
    status: str,
    note: str = "",
) -> dict[str, object]:
    return {
        "mode": mode,
        "batch": batch,
        "num_heads_q": num_heads_q,
        "num_heads_kv": num_heads_kv,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "dtype": str(dtype).replace("torch.", ""),
        "causal": is_causal,
        "requested_topk": requested_topk,
        "selected_ratio": selected_ratio,
        "selected_blocks_per_row": selected_blocks_per_row,
        "warmup": warmup,
        "iters": iters,
        "latency_ms": latency_ms,
        "status": status,
        "note": note,
    }


def try_benchmark(
    mode: str,
    fn,
    *,
    batch: int,
    num_heads_q: int,
    num_heads_kv: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    is_causal: bool,
    warmup: int,
    iters: int,
    requested_topk: float | None = None,
    selected_ratio: float | None = None,
    selected_blocks_per_row: float | None = None,
) -> dict[str, object]:
    try:
        latency_ms = bench(fn, warmup, iters)
        return make_row(
            mode=mode,
            batch=batch,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            seq_len=seq_len,
            head_dim=head_dim,
            dtype=dtype,
            is_causal=is_causal,
            warmup=warmup,
            iters=iters,
            requested_topk=requested_topk,
            selected_ratio=selected_ratio,
            selected_blocks_per_row=selected_blocks_per_row,
            latency_ms=latency_ms,
            status="ok",
        )
    except Exception as exc:
        status, note = classify_exception(exc)
        empty_xpu_cache()
        return make_row(
            mode=mode,
            batch=batch,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            seq_len=seq_len,
            head_dim=head_dim,
            dtype=dtype,
            is_causal=is_causal,
            warmup=warmup,
            iters=iters,
            requested_topk=requested_topk,
            selected_ratio=selected_ratio,
            selected_blocks_per_row=selected_blocks_per_row,
            latency_ms=None,
            status=status,
            note=note,
        )


def benchmark_preprocess_stages(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    topk: float,
    is_causal: bool,
    quant_block_size: int,
    tensor_layout: str,
    warmup: int,
    iters: int,
) -> tuple[dict[str, float], dict[str, object]]:
    from auto_round_kernel.sparge_preprocess_triton import (
        _block_map_lut_triton,
        _fill_block_map_triton,
        _get_pool_sim_triton_simmean_fuse_quant,
    )
    from auto_round_kernel.sparge_preprocess_triton import _safe_softmax as _triton_safe_softmax

    stage_names = [
        "key_mean",
        "pool_q_block64",
        "pool_k_block64",
        "pool_q_tile",
        "pooled_score_softmax_sort",
        "fill_block_map",
        "tile_to_qblock_index",
        "block_map_to_lut",
    ]
    sums = {name: 0.0 for name in stage_names}
    last_meta = None

    for _ in range(warmup + iters):
        ctx = ark._build_sparge_preprocess_context(
            q,
            k,
            is_causal=is_causal,
            smooth_k=True,
            simthreshd1=-1.0,
            topk=topk,
            attention_sink=False,
            quant_block_size=quant_block_size,
            tensor_layout=tensor_layout,
        )
        stage_ms: dict[str, float] = {}

        key_mean, stage_ms["key_mean"] = bench_with_output(
            lambda: ark._sequence_mean_native_layout(ctx.key, ctx.tensor_layout),
            warmup=0,
            iters=1,
        )
        (pooled_q, sim_qblocks, q_int8_hnd, q_scale), stage_ms["pool_q_block64"] = bench_with_output(
            lambda: _get_pool_sim_triton_simmean_fuse_quant(
                ctx.query,
                None,
                ctx.quant_block_size,
                ctx.simthreshd1,
                ctx.tensor_layout,
            ),
            warmup=0,
            iters=1,
        )
        (pooled_k, sim_kblocks, k_int8_hnd, k_scale), stage_ms["pool_k_block64"] = bench_with_output(
            lambda: _get_pool_sim_triton_simmean_fuse_quant(
                ctx.key,
                key_mean,
                ctx.quant_block_size,
                ctx.simthreshd1[: ctx.num_heads_kv],
                ctx.tensor_layout,
            ),
            warmup=0,
            iters=1,
        )

        def qtile_stage():
            if ctx.blocks_per_qtile <= 1:
                return pooled_q, sim_qblocks
            tile_pooled_q = []
            tile_sim_q = []
            for qtile in range(ctx.num_q_tiles):
                qblk_start = qtile * ctx.blocks_per_qtile
                qblk_end = min(qblk_start + ctx.blocks_per_qtile, pooled_q.size(2))
                tile_tokens = ark._slice_sequence_native_layout(
                    ctx.query,
                    ctx.tensor_layout,
                    qblk_start * ctx.quant_block_size,
                    min((qblk_end * ctx.quant_block_size), ctx.seq_len_q),
                )
                pooled_tile, sim_tile, _, _ = _get_pool_sim_triton_simmean_fuse_quant(
                    tile_tokens,
                    None,
                    ctx.query_tile_tokens,
                    ctx.simthreshd1,
                    ctx.tensor_layout,
                )
                tile_pooled_q.append(pooled_tile[:, :, 0, :])
                tile_sim_q.append(sim_tile[:, :, 0])
            return torch.stack(tile_pooled_q, dim=2), torch.stack(tile_sim_q, dim=2)

        (pooled_q_for_routing, sim_q_for_routing), stage_ms["pool_q_tile"] = bench_with_output(
            qtile_stage,
            warmup=0,
            iters=1,
        )

        kv_head_index = torch.arange(ctx.num_heads_q, device=ctx.query.device, dtype=torch.int64) // (
            ctx.num_heads_q // ctx.num_heads_kv
        )

        def routing_stage():
            pooled_k_for_q = pooled_k[:, kv_head_index]
            sim_k_for_q = sim_kblocks[:, kv_head_index]
            sim_k_expand = sim_k_for_q.unsqueeze(-2).expand(-1, -1, ctx.num_q_tiles, -1)
            sim_q_expand = sim_q_for_routing.unsqueeze(-1).expand(-1, -1, -1, pooled_k.size(2))
            pooled_score = torch.matmul(
                pooled_q_for_routing.to(torch.float32),
                pooled_k_for_q.transpose(-1, -2).to(torch.float32),
            )
            pooled_score *= ctx.head_dim**-0.5
            pooled_score = pooled_score.masked_fill(~sim_k_expand, -torch.inf)
            if ctx.is_causal:
                causal_mask = ark._build_block_causal_mask(
                    ctx.num_q_tiles, pooled_k.size(2), ctx.blocks_per_qtile, ctx.query.device
                )
                pooled_score = pooled_score.masked_fill(
                    ~causal_mask.view(1, 1, ctx.num_q_tiles, pooled_k.size(2)),
                    -torch.inf,
                )
            else:
                causal_mask = None
            pooled_prob = _triton_safe_softmax(pooled_score)
            sorted_prob = torch.sort(pooled_prob, dim=-1, descending=True)
            return pooled_prob, sorted_prob, sim_k_expand, sim_q_expand, causal_mask

        (
            pooled_prob,
            sorted_prob,
            sim_k_expand,
            sim_q_expand,
            causal_mask,
        ), stage_ms[
            "pooled_score_softmax_sort"
        ] = bench_with_output(routing_stage, warmup=0, iters=1)

        _, _, _, num_k_blocks = pooled_prob.shape
        num_to_select = (
            (ctx.topk.view(1, ctx.num_heads_q, 1) * num_k_blocks)
            .to(torch.int64)
            .expand(ctx.batch, -1, ctx.num_q_tiles)
            .contiguous()
        )

        def fill_stage():
            final_tile_map = torch.zeros_like(pooled_prob, dtype=torch.bool)
            final_tile_map[~sim_k_expand] = True
            final_tile_map[~sim_q_expand] = True
            final_tile_map = _fill_block_map_triton(final_tile_map, num_to_select, sorted_prob.indices)
            if causal_mask is not None:
                final_tile_map &= causal_mask.view(1, 1, ctx.num_q_tiles, num_k_blocks)
            return final_tile_map

        final_tile_map, stage_ms["fill_block_map"] = bench_with_output(fill_stage, warmup=0, iters=1)

        q_block_to_tile = (
            torch.arange((ctx.seq_len_q + ctx.quant_block_size - 1) // ctx.quant_block_size, device=ctx.query.device)
            * ctx.quant_block_size
        ) // ctx.query_tile_tokens
        q_block_to_tile = q_block_to_tile.clamp_max(ctx.num_q_tiles - 1)
        raw_block_map, stage_ms["tile_to_qblock_index"] = bench_with_output(
            lambda: final_tile_map.index_select(2, q_block_to_tile).contiguous(),
            warmup=0,
            iters=1,
        )

        (lut, valid_block_num), stage_ms["block_map_to_lut"] = bench_with_output(
            lambda: _block_map_lut_triton(raw_block_map),
            warmup=0,
            iters=1,
        )

        if _ >= warmup:
            for name in stage_names:
                sums[name] += stage_ms[name]

        last_meta = ark._finalize_sparge_preprocess_outputs(
            ctx,
            {
                "query_i8": q_int8_hnd,
                "key_i8": k_int8_hnd,
                "qscale": q_scale,
                "kscale": k_scale,
                "raw_block_map": raw_block_map,
                "tile_block_map": final_tile_map.contiguous(),
                "sim_qblocks": sim_q_for_routing.contiguous(),
                "sim_kblocks": sim_kblocks.contiguous(),
                "backend": "torch",
            },
        )
        del key_mean, pooled_q, sim_qblocks, q_int8_hnd, q_scale
        del pooled_k, sim_kblocks, k_int8_hnd, k_scale
        del pooled_q_for_routing, sim_q_for_routing, pooled_prob, sorted_prob
        del sim_k_expand, sim_q_expand, final_tile_map, raw_block_map, lut, valid_block_num
        if causal_mask is not None:
            del causal_mask

    avg = {name: sums[name] / float(iters) for name in stage_names}
    avg["preprocess_total"] = sum(avg.values())
    return avg, last_meta


def nhd_to_hnd(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 1, 3).contiguous()


def hnd_to_nhd(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 1, 3).contiguous()


def summarize_speedups(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    torch_row = next((row for row in rows if row["mode"] == "dense_torch_sdpa" and row["status"] == "ok"), None)
    sage_row = next((row for row in rows if row["mode"] == "dense_sagev1" and row["status"] == "ok"), None)
    torch_ms = None if torch_row is None else float(torch_row["latency_ms"])
    sage_ms = None if sage_row is None else float(sage_row["latency_ms"])
    for row in rows:
        latency_ms = row["latency_ms"]
        if latency_ms is None:
            row["speedup_vs_torch"] = None
            row["speedup_vs_sagev1"] = None
            row["baseline_tflops"] = None
            row["effective_tflops"] = None
            continue
        row["speedup_vs_torch"] = (torch_ms / latency_ms) if torch_ms is not None else None
        row["speedup_vs_sagev1"] = (sage_ms / latency_ms) if sage_ms is not None else None
        batch = int(row["batch"])
        num_heads_q = int(row["num_heads_q"])
        seq_len = int(row["seq_len"])
        head_dim = int(row["head_dim"])
        dense_flops = attention_dense_flops(batch, num_heads_q, seq_len, seq_len, head_dim)
        mode = str(row["mode"])
        row["baseline_tflops"] = None
        row["effective_tflops"] = None
        if mode in {
            "dense_torch_sdpa",
            "dense_sagev1",
            "sparse_kernel_only",
            "sparse_e2e",
            "sparse_qtile256_row64k_kernel_only",
            "sparse_qtile256_row64k_e2e",
        }:
            row["baseline_tflops"] = flops_to_tflops(dense_flops, float(latency_ms))
            work_ratio = 1.0
            if mode.startswith("sparse_") and row["selected_ratio"] is not None:
                work_ratio = float(row["selected_ratio"])
            row["effective_tflops"] = flops_to_tflops(dense_flops * work_ratio, float(latency_ms))
    return rows


def print_summary(rows: list[dict[str, object]]) -> None:
    print(
        "| mode | topk | selected_ratio | blocks/row | latency (ms) | baseline_tflops | effective_tflops | status | speedup_vs_torch | speedup_vs_sagev1 |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        topk = "-" if row["requested_topk"] is None else f"{float(row['requested_topk']):.3f}"
        ratio = "-" if row["selected_ratio"] is None else f"{float(row['selected_ratio']):.6f}"
        blocks = "-" if row["selected_blocks_per_row"] is None else f"{float(row['selected_blocks_per_row']):.3f}"
        latency = "-" if row["latency_ms"] is None else f"{float(row['latency_ms']):.3f}"
        baseline_tflops = "-" if row.get("baseline_tflops") is None else f"{float(row['baseline_tflops']):.3f}"
        effective_tflops = "-" if row.get("effective_tflops") is None else f"{float(row['effective_tflops']):.3f}"
        sp_torch = "-" if row.get("speedup_vs_torch") is None else f"{float(row['speedup_vs_torch']):.3f}"
        sp_sage = "-" if row.get("speedup_vs_sagev1") is None else f"{float(row['speedup_vs_sagev1']):.3f}"
        print(
            f"| {row['mode']} | {topk} | {ratio} | {blocks} | {latency} | {baseline_tflops} | {effective_tflops} | {row['status']} | {sp_torch} | {sp_sage} |"
        )
        if row["note"]:
            print(f"note[{row['mode']}]: {row['note']}")


def write_csv(rows: list[dict[str, object]], output_csv: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(args: argparse.Namespace) -> list[dict[str, object]]:
    ensure_sparse_binding()
    if not is_xpu_available():
        raise RuntimeError("XPU device is required")

    device = torch.device("xpu")
    dtype = torch.float16
    scale = 1.0 / math.sqrt(args.head_dim)
    enable_gqa = args.num_heads_q // args.num_heads_kv > 1
    q_hnd_src, k_hnd_src, v_hnd_src = build_inputs(
        args.batch,
        args.num_heads_q,
        args.num_heads_kv,
        args.seq_len,
        args.head_dim,
        dtype,
        device,
    )
    q_nhd_src = hnd_to_nhd(q_hnd_src)
    k_nhd_src = hnd_to_nhd(k_hnd_src)
    v_nhd_src = hnd_to_nhd(v_hnd_src)
    if args.tensor_layout == "NHD":
        q, k, v = q_nhd_src, k_nhd_src, v_nhd_src
        q_hnd, k_hnd, v_hnd = q_hnd_src, k_hnd_src, v_hnd_src
    else:
        q, k, v = q_hnd_src, k_hnd_src, v_hnd_src
        q_hnd, k_hnd, v_hnd = q_hnd_src, k_hnd_src, v_hnd_src

    rows: list[dict[str, object]] = []
    rows.append(
        try_benchmark(
            "dense_torch_sdpa",
            lambda: (
                hnd_to_nhd(
                    F.scaled_dot_product_attention(
                        nhd_to_hnd(q_nhd_src),
                        nhd_to_hnd(k_nhd_src),
                        nhd_to_hnd(v_nhd_src),
                        dropout_p=0.0,
                        is_causal=args.causal,
                        scale=scale,
                        enable_gqa=enable_gqa,
                    )
                )
                if args.tensor_layout == "HND"
                else F.scaled_dot_product_attention(
                    q_hnd, k_hnd, v_hnd, dropout_p=0.0, is_causal=args.causal, scale=scale, enable_gqa=enable_gqa
                )
            ),
            batch=args.batch,
            num_heads_q=args.num_heads_q,
            num_heads_kv=args.num_heads_kv,
            seq_len=args.seq_len,
            head_dim=args.head_dim,
            dtype=dtype,
            is_causal=args.causal,
            warmup=args.warmup,
            iters=args.iters,
        )
    )
    rows.append(
        try_benchmark(
            "dense_sagev1",
            lambda: (
                hnd_to_nhd(
                    ark.sagev1(
                        nhd_to_hnd(q_nhd_src),
                        nhd_to_hnd(k_nhd_src),
                        nhd_to_hnd(v_nhd_src),
                        scale=scale,
                        is_causal=args.causal,
                        quant_block_size=args.quant_block_size,
                        tensor_layout="HND",
                    )
                )
                if args.tensor_layout == "HND"
                else ark.sagev1(
                    q,
                    k,
                    v,
                    scale=scale,
                    is_causal=args.causal,
                    quant_block_size=args.quant_block_size,
                    tensor_layout=args.tensor_layout,
                )
            ),
            batch=args.batch,
            num_heads_q=args.num_heads_q,
            num_heads_kv=args.num_heads_kv,
            seq_len=args.seq_len,
            head_dim=args.head_dim,
            dtype=dtype,
            is_causal=args.causal,
            warmup=args.warmup,
            iters=args.iters,
        )
    )

    for topk in args.topk:
        preprocess = None
        selected_ratio = None
        selected_blocks_per_row = None
        preprocess_note = ""
        try:
            preprocess = ark.sparge_preprocess_topk(
                q,
                k,
                is_causal=args.causal,
                smooth_k=True,
                simthreshd1=-1.0,
                topk=topk,
                attention_sink=False,
                quant_block_size=args.quant_block_size,
                tensor_layout=args.tensor_layout,
                query_tile_tokens=args.q_tile_override or None,
                sparse_q_block_tokens=args.sparse_q_block_tokens,
                sparse_k_block_tokens=args.sparse_k_block_tokens,
            )
            stats = preprocess.get("stats", {})
            selected_ratio = float(stats.get("selected_ratio", 0.0))
            selected_blocks_per_row = float(stats.get("selected_blocks_per_row", 0.0))
        except Exception as exc:
            status, preprocess_note = classify_exception(exc)
            empty_xpu_cache()
            rows.append(
                make_row(
                    mode="sparse_kernel_only",
                    batch=args.batch,
                    num_heads_q=args.num_heads_q,
                    num_heads_kv=args.num_heads_kv,
                    seq_len=args.seq_len,
                    head_dim=args.head_dim,
                    dtype=dtype,
                    is_causal=args.causal,
                    warmup=args.warmup,
                    iters=args.iters,
                    requested_topk=topk,
                    selected_ratio=None,
                    selected_blocks_per_row=None,
                    latency_ms=None,
                    status=status,
                    note=f"preprocess failed before kernel benchmark: {preprocess_note}",
                )
            )
            rows.append(
                make_row(
                    mode="sparse_e2e",
                    batch=args.batch,
                    num_heads_q=args.num_heads_q,
                    num_heads_kv=args.num_heads_kv,
                    seq_len=args.seq_len,
                    head_dim=args.head_dim,
                    dtype=dtype,
                    is_causal=args.causal,
                    warmup=args.warmup,
                    iters=args.iters,
                    requested_topk=topk,
                    selected_ratio=None,
                    selected_blocks_per_row=None,
                    latency_ms=None,
                    status=status,
                    note=f"preprocess failed before e2e benchmark: {preprocess_note}",
                )
            )
            continue

        # try:
        #     preprocess_stage_ms, preprocess_profile_meta = benchmark_preprocess_stages(
        #         q,
        #         k,
        #         topk=topk,
        #         is_causal=args.causal,
        #         quant_block_size=args.quant_block_size,
        #         tensor_layout=args.tensor_layout,
        #         warmup=args.warmup,
        #         iters=args.iters,
        #     )
        #     stage_selected_ratio = float(preprocess_profile_meta["stats"].get("selected_ratio", 0.0))
        #     stage_selected_blocks_per_row = float(preprocess_profile_meta["stats"].get("selected_blocks_per_row", 0.0))
        #     for stage_name, latency_ms in preprocess_stage_ms.items():
        #         mode = stage_name if stage_name == "preprocess_total" else f"preprocess_{stage_name}"
        #         rows.append(
        #             make_row(
        #                 mode=mode,
        #                 batch=args.batch,
        #                 num_heads_q=args.num_heads_q,
        #                 num_heads_kv=args.num_heads_kv,
        #                 seq_len=args.seq_len,
        #                 head_dim=args.head_dim,
        #                 dtype=dtype,
        #                 is_causal=args.causal,
        #                 warmup=args.warmup,
        #                 iters=args.iters,
        #                 requested_topk=topk,
        #                 selected_ratio=stage_selected_ratio,
        #                 selected_blocks_per_row=stage_selected_blocks_per_row,
        #                 latency_ms=latency_ms,
        #                 status="ok",
        #             )
        #         )
        # except Exception as exc:
        #     status, note = classify_exception(exc)
        #     rows.append(
        #         make_row(
        #             mode="preprocess_total",
        #             batch=args.batch,
        #             num_heads_q=args.num_heads_q,
        #             num_heads_kv=args.num_heads_kv,
        #             seq_len=args.seq_len,
        #             head_dim=args.head_dim,
        #             dtype=dtype,
        #             is_causal=args.causal,
        #             warmup=args.warmup,
        #             iters=args.iters,
        #             requested_topk=topk,
        #             selected_ratio=selected_ratio,
        #             selected_blocks_per_row=selected_blocks_per_row,
        #             latency_ms=None,
        #             status=status,
        #             note=f"preprocess stage profiling failed: {note}",
        #         )
        #     )

        sparse_kernel_mode = (
            "sparse_qtile256_row64k_kernel_only"
            if preprocess["sparse_q_block_tokens"] == 256
            and preprocess["sparse_k_block_tokens"] == 64
            and args.q_tile_override == 256
            else "sparse_kernel_only"
        )
        rows.append(
            try_benchmark(
                sparse_kernel_mode,
                lambda preprocess=preprocess: (
                    hnd_to_nhd(
                        ark.sage_sparse(
                            preprocess["query_i8"],
                            preprocess["key_i8"],
                            nhd_to_hnd(v_nhd_src),
                            preprocess["lut"],
                            preprocess["valid_block_num"],
                            is_causal=args.causal,
                            scale=scale,
                            quant_block_size=preprocess["quant_block_size"],
                            qscale=preprocess["qscale"],
                            kscale=preprocess["kscale"],
                            q_tile_override=args.q_tile_override,
                            sparse_q_block_tokens=preprocess["sparse_q_block_tokens"],
                            sparse_k_block_tokens=preprocess["sparse_k_block_tokens"],
                            tensor_layout="HND",
                        )
                    )
                    if args.tensor_layout == "HND"
                    else ark.sage_sparse(
                        preprocess["query_i8"],
                        preprocess["key_i8"],
                        v,
                        preprocess["lut"],
                        preprocess["valid_block_num"],
                        is_causal=args.causal,
                        scale=scale,
                        quant_block_size=preprocess["quant_block_size"],
                        qscale=preprocess["qscale"],
                        kscale=preprocess["kscale"],
                        q_tile_override=args.q_tile_override,
                        sparse_q_block_tokens=preprocess["sparse_q_block_tokens"],
                        sparse_k_block_tokens=preprocess["sparse_k_block_tokens"],
                        tensor_layout=args.tensor_layout,
                    )
                ),
                batch=args.batch,
                num_heads_q=args.num_heads_q,
                num_heads_kv=args.num_heads_kv,
                seq_len=args.seq_len,
                head_dim=args.head_dim,
                dtype=dtype,
                is_causal=args.causal,
                warmup=args.warmup,
                iters=args.iters,
                requested_topk=topk,
                selected_ratio=selected_ratio,
                selected_blocks_per_row=selected_blocks_per_row,
            )
        )
        sparse_e2e_mode = (
            "sparse_qtile256_row64k_e2e"
            if preprocess["sparse_q_block_tokens"] == 256
            and preprocess["sparse_k_block_tokens"] == 64
            and args.q_tile_override == 256
            else "sparse_e2e"
        )
        rows.append(
            try_benchmark(
                sparse_e2e_mode,
                lambda topk=topk: (
                    hnd_to_nhd(
                        ark.sparge_sage2_attn_meansim_topk_xpu(
                            nhd_to_hnd(q_nhd_src),
                            nhd_to_hnd(k_nhd_src),
                            nhd_to_hnd(v_nhd_src),
                            is_causal=args.causal,
                            scale=scale,
                            smooth_k=True,
                            simthreshd1=-1.0,
                            topk=topk,
                            attention_sink=False,
                            tensor_layout="HND",
                            q_tile_override=args.q_tile_override,
                            sparse_q_block_tokens=args.sparse_q_block_tokens,
                            sparse_k_block_tokens=args.sparse_k_block_tokens,
                        )
                    )
                    if args.tensor_layout == "HND"
                    else ark.sparge_sage2_attn_meansim_topk_xpu(
                        q,
                        k,
                        v,
                        is_causal=args.causal,
                        scale=scale,
                        smooth_k=True,
                        simthreshd1=-1.0,
                        topk=topk,
                        attention_sink=False,
                        tensor_layout=args.tensor_layout,
                        q_tile_override=args.q_tile_override,
                        sparse_q_block_tokens=args.sparse_q_block_tokens,
                        sparse_k_block_tokens=args.sparse_k_block_tokens,
                    )
                ),
                batch=args.batch,
                num_heads_q=args.num_heads_q,
                num_heads_kv=args.num_heads_kv,
                seq_len=args.seq_len,
                head_dim=args.head_dim,
                dtype=dtype,
                is_causal=args.causal,
                warmup=args.warmup,
                iters=args.iters,
                requested_topk=topk,
                selected_ratio=selected_ratio,
                selected_blocks_per_row=selected_blocks_per_row,
            )
        )
        del preprocess
        empty_xpu_cache()

    return summarize_speedups(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark torch SDPA, sagev1, and sparse attention across top-k.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num-heads-q", type=int, default=40)
    parser.add_argument("--num-heads-kv", type=int, default=40)
    parser.add_argument("--seq-len", type=int, default=75600)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--topk", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25, 0.125])
    parser.add_argument("--quant-block-size", type=int, default=64)
    parser.add_argument(
        "--q-tile-override",
        type=int,
        default=0,
        choices=(0, 64, 256),
        help="Sparse kernel q_tile override. 0 keeps the default kernel choice.",
    )
    parser.add_argument(
        "--sparse-q-block-tokens",
        type=int,
        default=None,
        help="Optional sparse Q-row granularity in tokens. Use 256 with --q-tile-override 256 for the decoupled path.",
    )
    parser.add_argument(
        "--sparse-k-block-tokens",
        type=int,
        default=None,
        help="Optional sparse K logical-block granularity in tokens. Use 64 for the decoupled qtile256 path.",
    )
    parser.add_argument("--tensor-layout", choices=("HND", "NHD"), default="HND")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument(
        "--causal", action="store_true", help="Run causal attention instead of the default non-causal mode."
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("bench_sparse_topk_results.csv"),
        help="Where to write the CSV result table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_benchmark(args)
    write_csv(rows, args.output_csv)
    print_summary(rows)
    print(f"wrote csv: {args.output_csv}")


if __name__ == "__main__":
    main()
