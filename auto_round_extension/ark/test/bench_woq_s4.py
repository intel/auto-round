# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

"""Replay the M=1 GEMV vs K64 split4 matrix using the test_woq_s4 fixture."""

import argparse
import hashlib
import importlib
import os
from contextlib import nullcontext
from itertools import product
from pathlib import Path
from time import perf_counter_ns

import test_woq_s4 as fixture
import torch

ark = fixture.ark
GROUP_SIZE = 128
OPTIONS = (fixture.COMPUTE_TYPE, fixture.WEIGHT_TYPE, fixture.SCALE_TYPE, fixture.ASYM)
M1_SHAPES = (
    "2048x2048",
    "4096x1024",
    "4096x4096",
    "4096x8192",
    "4096x12288",
    "4096x14336",
    f"{fixture.K}x{fixture.N}",
    "8192x4096",
    "8192x8192",
    "14336x4096",
)


def uses_ark_split4(inner, columns):
    return inner == fixture.K and 8192 <= columns <= fixture.N and columns % 128 == 0


def check_memory(*, timing):
    free, total = torch.xpu.mem_get_info()
    print(f"free_bytes={free}/{total} ({free / total:.1%})", flush=True)
    if timing and free <= 0.90 * total:
        raise SystemExit("Timing blocked: the selected XPU must have >90% free memory.")
    if free < 8 * 2**30:
        raise SystemExit("The selected XPU needs at least 8 GiB for this check.")


def make_case(group_size, *, with_bias, rows, inner, columns):
    return fixture._ark_case(rows, n=columns, k=inner, blocksize=group_size, with_bias=with_bias)


def profile_xpu_calls(call, count, warmup, *, record_events=True):
    for index in range(warmup):
        output = call(index)
    torch.xpu.synchronize()
    scope = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.XPU]) if record_events else nullcontext()
    with scope as trace:
        started = perf_counter_ns()
        for index in range(count):
            output = call(index)
        submitted = perf_counter_ns()
        torch.xpu.synchronize()
        finished = perf_counter_ns()
    events = (
        []
        if trace is None
        else [
            event
            for event in trace.events()
            if event.device_type == torch.autograd.DeviceType.XPU and not event.is_user_annotation
        ]
    )
    return output, events, ((submitted - started) / (1000 * count), (finished - started) / (1000 * count))


def stage_means(events, tags, count):
    stage_tags = {"main": tags[0], "cast": "CopyScalarFunc", "fill": "FillFunctor"}
    if len(tags) > 1:
        stage_tags["reduce"] = tags[1]
    means = dict.fromkeys(("main", "reduce", "cast", "fill"), 0.0)
    for event in events:
        stages = [stage for stage, tag in stage_tags.items() if tag in event.name]
        assert len(stages) == 1, event.name
        means[stages[0]] += event.time_range.elapsed_us() / count
    means["device"] = sum(means.values())
    return " ".join(f"{stage}_us={mean:.4f}" for stage, mean in means.items())


def timeline_means(events, count, host_times):
    intervals = sorted((event.time_range.start, event.time_range.end) for event in events)
    first_start, last_end = intervals[0]
    gaps = 0.0
    overlap = 0.0
    for start, end in intervals[1:]:
        gaps += max(0.0, start - last_end)
        overlap += max(0.0, min(end, last_end) - start)
        last_end = max(last_end, end)
    submit, wall = host_times
    return (
        f"span_us={(last_end - first_start) / count:.4f} gap_us={gaps / count:.4f} "
        f"overlap_us={overlap / count:.4f} prof_submit_us={submit:.4f} prof_wall_us={wall:.4f}"
    )


def compare_m1_matrix_group(
    probe,
    inner,
    columns,
    group_size,
    *,
    routes_only,
    profile_stages,
    bias_cast_ab,
    bf16_io_ab,
    io_buffer_ab,
):
    shape = f"{inner}x{columns}"
    activation, packed, bias, _ = make_case(group_size, with_bias=True, rows=1, inner=inner, columns=columns)
    activation_host = activation.cpu()
    bias_host = bias.cpu()
    weight = ark.unpack_weight(packed, fixture.DTYPE, columns, inner, group_size, *OPTIONS).cpu().float()
    queue_ptr = ark.get_stream(activation)
    device = activation.device
    native = ark.xpu_lib.woqgemm
    integrated = uses_ark_split4(inner, columns)
    dtype_code = ark.cvt_dtype(torch.float16)
    compute, weight_type, scale_type = (ark.cvtstr_dtype(dtype) for dtype in OPTIONS[:3])
    batch = 1 if routes_only else fixture._batch_for_m(1)
    runs = fixture._runs_for_m(1)
    warmup = fixture._warmup_for_runs(runs)
    weights = fixture._repeat_ark_blob(packed, batch)
    weight_ptrs = [view.data_ptr() for view in weights]
    blob_size = packed.numel()
    if not integrated:
        if probe is None:
            raise ValueError(f"{shape} requires the standalone split-K4 probe")
        split4 = probe.run_k64_split4
        scale_bytes = inner // group_size * columns * activation.element_size()
        scale_ptrs = [view[-scale_bytes:].data_ptr() for view in weights]
        scratch = torch.empty(4 * columns, dtype=torch.float32, device=device)
        scratch_ptr = scratch.data_ptr()
    orders = (("gemv", "k64-s4"),) if routes_only else (("gemv", "k64-s4"), ("k64-s4", "gemv"))
    bf16_only = bias_cast_ab or bf16_io_ab or io_buffer_ab == "split"
    dtypes = (torch.bfloat16,) if bf16_only else (torch.float16, torch.bfloat16)
    measure_host = bf16_io_ab or io_buffer_ab
    bias_cases = (False,) if measure_host else ((True,) if bias_cast_ab else (False, True))
    io_variants = ((True, True), (False, True), (True, False), (False, False)) if bf16_io_ab else ((True, True),)
    buffer_variants = (("0", False, False),)
    if io_buffer_ab == "split":
        buffer_variants += (("input", True, False), ("output", False, True))
    if io_buffer_ab:
        buffer_variants += (("1", True, True),)
    for dtype in dtypes:
        caller = activation_host.to(dtype)
        reference = caller.float() @ weight.t()
        activations = fixture._repeat_activation(caller.to(device), batch)
        prepared_inputs = activations.to(torch.float16) if bf16_io_ab else activations
        if io_buffer_ab:
            input_ring = torch.empty_like(activations, dtype=torch.float16)
            output_ring = torch.empty((batch, 1, columns), dtype=torch.float16, device=device)
            result_ring = torch.empty_like(output_ring, dtype=dtype)
        caller_bias = bias_host.to(dtype)
        for with_bias in bias_cases:
            bias_input = caller_bias.to(device) if with_bias else None
            bias_variants = (
                (("per-call", bias_input), ("precast", bias_input.to(torch.float16)))
                if bias_cast_ab
                else (("normal", bias_input),)
            )
            expected = reference + caller_bias.float() if with_bias else reference
            dtype_name = "fp16" if dtype == torch.float16 else "bf16"
            path = "ark" if integrated else "probe"
            label = f"{shape} M=1 group={group_size} bias={int(with_bias)} dtype={dtype_name} path={path}"
            cast_outputs = {}
            variants = tuple(product(bias_variants, io_variants, buffer_variants))
            check_memory(timing=not routes_only)
            for round_id, order in enumerate(orders, 1):
                round_variants = variants if round_id == 1 else variants[::-1]
                for ((bias_cast, bias_input), (cast_input, cast_output), buffers), mode in product(
                    round_variants, order
                ):
                    buffer_label, reuse_input, reuse_output = buffers
                    enabled = mode == "k64-s4"
                    os.environ[fixture.M1_SPLIT4_SWITCH] = "1" if integrated and enabled else "0"
                    use_probe = enabled and not integrated
                    case_label = f"{label} bias_cast={bias_cast}" if bias_cast_ab else label
                    if bf16_io_ab:
                        case_label += f" input_cast={int(cast_input)} output_cast={int(cast_output)}"
                    if io_buffer_ab:
                        case_label += f" reuse_io={buffer_label}"
                    active_inputs = activations if cast_input else prepared_inputs

                    def call(index):
                        slot = index % batch
                        source = active_inputs[slot]
                        input_fp16 = (
                            input_ring[slot].copy_(source)
                            if reuse_input and dtype == torch.bfloat16
                            else source.to(torch.float16)
                        )
                        bias_fp16 = (
                            torch.empty(0, dtype=torch.float16, device=device)
                            if bias_input is None
                            else bias_input.to(torch.float16)
                        )
                        output = (
                            output_ring[slot].zero_()
                            if reuse_output
                            else torch.zeros((1, columns), dtype=torch.float16, device=device)
                        )
                        weight_ptr = weight_ptrs[slot]
                        if use_probe:
                            split4(
                                queue_ptr,
                                input_fp16.data_ptr(),
                                weight_ptr,
                                scale_ptrs[slot],
                                bias_fp16.data_ptr(),
                                output.data_ptr(),
                                scratch_ptr,
                                1,
                                columns,
                                inner,
                                group_size,
                            )
                        else:
                            native(
                                queue_ptr,
                                1,
                                columns,
                                inner,
                                input_fp16.data_ptr(),
                                dtype_code,
                                weight_ptr,
                                output.data_ptr(),
                                bias_fp16.data_ptr(),
                                group_size,
                                compute,
                                weight_type,
                                scale_type,
                                OPTIONS[3],
                                blob_size,
                            )
                        if reuse_output and dtype == torch.bfloat16:
                            return result_ring[slot].copy_(output)
                        return output.to(dtype) if cast_output else output

                    if routes_only or profile_stages:
                        count = batch if profile_stages else 1
                        result, events, host_times = profile_xpu_calls(call, count, warmup if profile_stages else 1)
                        names = [event.name for event in events]
                        tags = (
                            ("::gemv",)
                            if not enabled
                            else (
                                ("SplitKS4ProbeName", "ReduceS4ProbeName")
                                if use_probe
                                else ("DenseWoqS4SplitK4", "DenseWoqS4ReduceK4")
                            )
                        )
                        casts = (
                            int(cast_input) + int(cast_output) + int(with_bias and bias_cast != "precast")
                            if dtype == torch.bfloat16
                            else 0
                        )
                        assert all(sum(tag in name for name in names) == count for tag in tags), (
                            case_label,
                            mode,
                            names,
                        )
                        assert len(names) == count * (len(tags) + casts + 1), (case_label, mode, names)
                        assert sum("CopyScalarFunc" in name for name in names) == count * casts, (case_label, names)
                    elif measure_host:
                        result, _, host_times = profile_xpu_calls(call, runs, warmup, record_events=False)
                        duration = host_times[1] / 1e6
                    else:
                        result, duration = fixture._benchmark_loop(call, runs, warmup)
                    actual = (result if cast_output else result.to(dtype)).cpu()
                    error = ((actual.float() - expected).norm() / expected.norm()).item()
                    assert error < 0.01, (case_label, mode, error)
                    if bias_cast_ab or measure_host:
                        bits = actual.view(torch.int16)
                        assert torch.equal(bits, cast_outputs.setdefault(mode, bits)), (case_label, mode)
                    if profile_stages:
                        print(
                            f"MATRIX_STAGE {case_label} R={round_id} {mode} calls={count} casts={casts} "
                            f"{stage_means(events, tags, count)} rel_l2={error:.7f} "
                            f"{timeline_means(events, count, host_times)}",
                            flush=True,
                        )
                    elif routes_only:
                        print(f"MATRIX_ROUTE {case_label} {mode} casts={casts} rel_l2={error:.7f}", flush=True)
                    else:
                        prefix = "MATRIX_IO_AB" if bf16_io_ab else ("MATRIX_BIAS_AB" if bias_cast_ab else "MATRIX")
                        if io_buffer_ab:
                            prefix = "MATRIX_BUFFER_AB"
                        host_split = ""
                        if measure_host:
                            submit, wall = host_times
                            host_split = f" submit_us={submit:.4f} sync_tail_us={wall - submit:.4f}"
                        print(
                            f"{prefix} {case_label} R={round_id} {mode} mean_us={duration * 1e6:.3f} "
                            f"rel_l2={error:.7f}{host_split}",
                            flush=True,
                        )
                check_memory(timing=not routes_only)


def compare_m1_matrix(probe, shapes, *, routes_only, profile_stages, bias_cast_ab, bf16_io_ab, io_buffer_ab):
    shape_labels = ",".join(shapes)
    if bf16_io_ab:
        print(f"M=1 group128 no-bias BF16: input/output casts inside/outside timing; {shape_labels}")
    elif io_buffer_ab:
        buffer_dtypes = "BF16" if io_buffer_ab == "split" else "FP16/BF16"
        print(f"M=1 group128 no-bias {buffer_dtypes}: per-call allocation vs reusable I/O buffers; {shape_labels}")
    else:
        print(f"M=1 group32/128 x bias/no-bias x FP16/BF16: {shape_labels}")
    print("Common caller includes casts and output allocation; split4 outside the wide-N rule uses the probe.")
    if bias_cast_ab:
        print(
            "BF16+bias only: per-call bias cast vs the same bias cast before the timing window; verify bitwise outputs."
        )
    if profile_stages:
        print("Device stages only, one 64-weight rotation per mode after warmup, two reversed rounds.")
    for shape in shapes:
        inner, columns = map(int, shape.split("x"))
        for group_size in ((GROUP_SIZE,) if (bf16_io_ab or io_buffer_ab) else (32, GROUP_SIZE)):
            compare_m1_matrix_group(
                probe,
                inner,
                columns,
                group_size,
                routes_only=routes_only,
                profile_stages=profile_stages,
                bias_cast_ab=bias_cast_ab,
                bf16_io_ab=bf16_io_ab,
                io_buffer_ab=io_buffer_ab,
            )


def verify_binary(module, expected):
    binary = Path(module.__file__).resolve()
    with binary.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if expected and digest != expected:
        raise SystemExit(f"Unexpected library SHA256: {binary}: {digest}")
    print(f"Loaded {binary} sha256={digest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    profile_options = parser.add_mutually_exclusive_group()
    profile_options.add_argument(
        "--routes-only", action="store_true", help="verify actual device kernels without timing"
    )
    profile_options.add_argument("--profile-stages", action="store_true", help="profile per-call XPU device stages")
    cast_options = parser.add_mutually_exclusive_group()
    cast_options.add_argument(
        "--bias-cast-ab", action="store_true", help="compare BF16 bias conversion inside/outside the call"
    )
    cast_options.add_argument(
        "--bf16-io-ab", action="store_true", help="isolate BF16 input and output casts on the integrated path"
    )
    cast_options.add_argument(
        "--io-buffer-ab",
        nargs="?",
        choices=("both", "split"),
        const="both",
        help="test reusable I/O buffers without removing casts; split tests each side in BF16",
    )
    parser.add_argument("--shape", choices=M1_SHAPES, action="append", help="restrict the shape sweep; may be repeated")
    parser.add_argument("--device-uuid", help="require this exact XPU device UUID")
    parser.add_argument("--expect-sha256", help="require this exact ARK XPU shared-library hash")
    parser.add_argument("--expect-probe-sha256", help="require this exact split4 probe shared-library hash")
    args = parser.parse_args()
    os.environ.pop(fixture.M1_SPLIT4_SWITCH, None)
    verify_binary(ark.xpu_lib, args.expect_sha256)
    print(f"Fixture: {Path(fixture.__file__).resolve()}")
    if torch.xpu.device_count() != 1:
        raise SystemExit("Select exactly one XPU before running the benchmark.")
    device = torch.xpu.get_device_properties(0)
    print(f"Torch {torch.__version__}; XPU {device.name}, ID {device.device_id:#06x}, UUID {device.uuid}")
    if args.device_uuid and str(device.uuid).lower() != args.device_uuid.lower():
        raise SystemExit(f"Unexpected XPU device: {device}")
    check_memory(timing=not args.routes_only)
    torch.set_num_threads(4)
    shapes = args.shape or M1_SHAPES
    probe = None
    if args.expect_probe_sha256 or any(not uses_ark_split4(*map(int, shape.split("x"))) for shape in shapes):
        probe = importlib.import_module("ark_2673_dpas_probe")
        verify_binary(probe, args.expect_probe_sha256)
    with torch.inference_mode():
        compare_m1_matrix(
            probe,
            shapes,
            routes_only=args.routes_only,
            profile_stages=args.profile_stages,
            bias_cast_ab=args.bias_cast_ab,
            bf16_io_ab=args.bf16_io_ab,
            io_buffer_ab=args.io_buffer_ab,
        )


if __name__ == "__main__":
    main()
