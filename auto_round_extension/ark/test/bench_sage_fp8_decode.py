# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import statistics

import auto_round_kernel as ark
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark the ARK Sage FP8 attention kernel")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seq-len-qo", type=int, default=1)
    parser.add_argument("--seq-len-kv", type=int, default=2048)
    parser.add_argument("--num-heads-q", type=int, default=8)
    parser.add_argument("--num-heads-kv", type=int, default=1)
    parser.add_argument("--head-dim", type=int, choices=(64, 96, 128, 192), default=128)
    parser.add_argument("--no-smooth-v", action="store_true", help="Disable V smoothing and fused mean restoration")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--causal", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.xpu.is_available():
        raise RuntimeError("An available XPU device is required")
    if ark.xpu_lib is None or not hasattr(ark.xpu_lib, "sage_fp8"):
        raise RuntimeError("The loaded ARK XPU extension does not export sage_fp8")
    if args.seq_len_qo <= 0 or args.seq_len_kv <= 0:
        raise ValueError("Sequence lengths must be positive")
    if args.iterations <= 0 or args.warmup < 0:
        raise ValueError("--iterations must be positive and --warmup must be non-negative")

    query = torch.randn(
        args.batch,
        args.num_heads_q,
        args.seq_len_qo,
        args.head_dim,
        device="xpu",
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        args.batch,
        args.num_heads_kv,
        args.seq_len_kv,
        args.head_dim,
        device="xpu",
        dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)

    softmax_scale = 1.0 / math.sqrt(args.head_dim)

    def run():
        return ark.sage_fp8(
            query,
            key,
            value,
            scale=softmax_scale,
            is_causal=args.causal,
            enable_gqa=args.num_heads_q != args.num_heads_kv,
            smooth_v=not args.no_smooth_v,
        )

    for _ in range(args.warmup):
        run()
    torch.xpu.synchronize()

    timings_us = []
    output = None
    for _ in range(args.iterations):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        output = run()
        end.record()
        end.synchronize()
        timings_us.append(start.elapsed_time(end) * 1000.0)

    phase = "decode" if args.seq_len_qo == 1 else "prefill"
    print(f"ARK module: {ark.xpu_lib.__file__}")
    print(f"phase: {phase}")
    print(
        f"shape: B={args.batch}, Sq={args.seq_len_qo}, Skv={args.seq_len_kv}, "
        f"Hq={args.num_heads_q}, Hkv={args.num_heads_kv}, D={args.head_dim}"
    )
    print(f"warmup={args.warmup}, iterations={args.iterations}, causal={args.causal}")
    print(f"median latency: {statistics.median(timings_us):.3f} us")
    print(f"samples: {[round(value, 3) for value in timings_us]} us")
    print(f"output: shape={tuple(output.shape)}, dtype={output.dtype}")


if __name__ == "__main__":
    main()