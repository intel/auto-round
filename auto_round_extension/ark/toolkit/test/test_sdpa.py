import math
from weakref import ref
import torch
import pandas as pd
from ut_utils import *
import time

ark = None


ARK_BENCHMARK_COLUMNS = [
    "#",
    "API Version",
    "Decode/Prefill",
    "data type",
    "Batch",
    "NumHeads_q",
    "NumHeads_kv",
    "Seq Length QO",
    "Seq Length KV",
    "Head Size QK",
    "Head Size VO",
    "Causal Mask",
    "Variable SeqLen",
    "CRI GB/s",
    "CRI TFlop/s",
    "CRI Time (ms)",
    "CRI flops/clk",
    "CRI % of peak",
    "Comments",
]


def dtype_to_name(dtype):
    dtype_names = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.float64: "float64",
    }
    return dtype_names.get(dtype, str(dtype).replace("torch.", ""))


def resolve_torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    dtype_aliases = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
        "fp64": torch.float64,
        "float64": torch.float64,
        "double": torch.float64,
    }
    normalized = str(dtype).strip().lower()
    if normalized not in dtype_aliases:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return dtype_aliases[normalized]


def save_ark_benchmark_results(df, output_file="benchmark_results.xlsx"):
    for column in ARK_BENCHMARK_COLUMNS:
        if column not in df.columns:
            df[column] = None
    df = df[ARK_BENCHMARK_COLUMNS]
    try:
        df.to_excel(output_file, index=False, engine="openpyxl")
    except Exception as exc:
        print(f"Error saving Excel file {output_file}: {exc}")


def build_attn_bias(seq, seq_kv, dtype, device, is_causal):
    attn_bias = torch.zeros(seq, seq_kv, dtype=dtype, device=device)
    if is_causal:
        diagonal = seq_kv - seq
        causal_mask = torch.ones(seq, seq_kv, dtype=torch.bool, device=device).tril(diagonal=diagonal)
        attn_bias.masked_fill_(causal_mask.logical_not(), float("-inf"))
    return attn_bias


def benchmark_ark_sdpa_case(
    batch,
    seq,
    seq_kv,
    h_q,
    h_kv,
    head_size_qk,
    head_size_vo=None,
    dt=torch.float16,
    dev="xpu",
    is_causal=False,
    warmup_runs=10,
    benchmark_runs=100,
):
    if head_size_vo is None:
        head_size_vo = head_size_qk

    group = h_q // h_kv
    q = torch.rand(batch, h_q, seq, head_size_qk, dtype=dt, device=dev)
    k = torch.rand(batch, h_kv, seq_kv, head_size_qk, dtype=dt, device=dev)
    v = torch.rand(batch, h_kv, seq_kv, head_size_vo, dtype=dt, device=dev)
    attn_bias = build_attn_bias(seq, seq_kv, dt, q.device, is_causal)
    scale = 1 / math.sqrt(head_size_qk)
    ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=scale, enable_gqa=group > 1, is_causal=True
        )
    ret = get_ark().sdpa(q, k, v, scale=scale, is_causal=is_causal)
    dff = abs(ref -ret)
    print_top_diffs(dff, ref, ret, topk=4, threshold=1)
    for _ in range(warmup_runs):
        get_ark().sdpa(q, k, v, scale=scale, is_causal=is_causal)
    if dev == "xpu":
        torch.xpu.synchronize()

    st = time.time()
    for _ in range(benchmark_runs):
        get_ark().sdpa(q, k, v, scale=scale, is_causal=is_causal)
    if dev == "xpu":
        torch.xpu.synchronize()
    et = time.time()
    torch.xpu.empty_cache()
    dur = (et - st) / benchmark_runs
    ops = group * h_kv * seq * head_size_qk * seq_kv * 2
    # ops += group * h_kv * seq * seq_kv * 2
    ops += group * h_kv * seq * head_size_vo * seq_kv * 2
    ops *= batch

    mem = batch * h_q * seq * head_size_qk * dt.itemsize
    mem += batch * h_kv * seq_kv * head_size_qk * dt.itemsize
    mem += batch * h_kv * seq_kv * head_size_vo * dt.itemsize
    mem += seq * seq_kv * dt.itemsize
    return {
        "CRI GB/s": mem / dur / 1e9,
        "CRI TFlop/s": ops / dur / 1e12,
        "CRI Time (ms)": dur * 1e3,
    }

TEST_CASES = [
    # #--- bfloat16 Cases ---
    ("prefill", "float16", 128, 1, 96, 8, 4096, 4096, True, False, "F16 Case 10 Prefill"),
    # ("prefill", "float16", 128, 1, 96, 8, 4096, 8192, True, False, "F16 Case 10 Prefill"),
    ("prefill", "float16", 128, 1, 96, 8, 8192, 8192, True, False, "F16 Case 10 Prefill"),
    ("prefill", "float16", 128, 1, 96, 8, 16384, 16384, True, False, "F16 Case 10 Prefill"),
    # ("prefill", "float16", 128, 1, 96, 8, 32768, 32768, True, False, "F16 Case 10 Prefill"),
# mode, dtype, hdim, batch, nh_q, nh_kv, seq_qo, seq_kv, is_causal, is_varlen, comment
    # ("decode", "bfloat16", 1, 32, 8, 1, 4096, False, False, "BF16 Case 10"),

    # # --- MXFP8 Cases ---
    # ("prefill", "mx_float_e4m3", 1, 4, 4, 512, 512, False, False, "MXFP8 e4m3 Case 1 Prefill"),
    # 
    # ("prefill", "mx_float_e5m2", 1, 4, 4, 512, 512, False, False, "MXFP8 e5m2 Case 5 Prefill"),
    # # --- FP8 Cases ---
    # ("prefill", "float_e4m3", 1, 4, 4, 512, 512, False, False, "FP8 e4m3 Case 1 Prefill"),
]

def run_ark_sdpa_to_excel(
    test_cases,
    output_file="benchmark_results.xlsx",
    api_version="ARK.sdpa",
    dev="xpu",
    warmup_runs=10,
    benchmark_runs=100,
):
    results = []
    print(f"Starting ARK SDPA benchmark run... ({len(test_cases)} cases)")

    for idx, case in enumerate(test_cases, start=1):
        mode, dtype, head_dim, batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, is_causal, is_varlen, comment = case
        torch_dtype = resolve_torch_dtype(dtype)
        row_data = {
            "#": idx,
            "API Version": api_version,
            "Decode/Prefill": mode.capitalize(),
            "data type": dtype_to_name(torch_dtype),
            "Batch": batch,
            "NumHeads_q": num_heads_q,
            "NumHeads_kv": num_heads_kv,
            "Seq Length QO": seq_len_qo,
            "Seq Length KV": seq_len_kv,
            "Head Size QK": head_dim,
            "Head Size VO": head_dim,
            "Causal Mask": "Yes" if is_causal else "No",
            "Variable SeqLen": "Yes" if is_varlen else "No",
            "CRI flops/clk": None,
            "CRI % of peak": None,
            "Comments": comment,
        }

        print(f"[{idx}/{len(test_cases)}] Running {comment}...")
        try:
            metrics = benchmark_ark_sdpa_case(
                batch=batch,
                seq=seq_len_qo,
                seq_kv=seq_len_kv,
                h_q=num_heads_q,
                h_kv=num_heads_kv,
                head_size_qk=head_dim,
                head_size_vo=head_dim,
                dt=torch_dtype,
                dev=dev,
                is_causal=is_causal,
                warmup_runs=warmup_runs,
                benchmark_runs=benchmark_runs,
            )
            row_data.update(metrics)
        except Exception as exc:
            print(f"Error running benchmark for case {idx} ({comment}): {exc}")
            row_data.update(
                {
                    "CRI GB/s": None,
                    "CRI TFlop/s": None,
                    "CRI Time (ms)": None,
                    "Comments": f"{comment} (Run Failed: {exc})",
                }
            )

        results.append(row_data)
        result_df = pd.DataFrame(results)
        print(result_df[["NumHeads_q", "NumHeads_kv", "Seq Length QO", "Seq Length KV", "CRI TFlop/s"]])
        save_ark_benchmark_results(result_df, output_file=output_file)

    print(f"Success! Results saved to {output_file}")
    return pd.DataFrame(results)

def is_xpu_available():
    return hasattr(torch, "xpu") and torch.xpu.is_available()

def get_ark():
    """Lazily initialize and return the ARK instance."""
    global ark
    if ark is None:
        if not is_xpu_available():
            raise RuntimeError("XPU is not available; cannot initialize auto_round_kernel.ARK()")
        ark = auto_round_kernel.ARK()
    return ark

def has_sdpa():
    """Check if Flash Attention kernel is available."""
    try:
        ark_instance = get_ark()
    except Exception:
        return False
    if ark_instance.xpu_lib is None:
        return False
    return hasattr(ark_instance.xpu_lib, "sdpa")

def build_mask(seq, seq_kv, dt=torch.float32, dev="xpu", const_count=0, const_value=0.0):
    mask = torch.rand(seq, seq_kv, dtype=dt, device=dev)
    if const_count <= 0:
        return mask
    flat_mask = mask.reshape(-1)
    const_count = min(const_count, flat_mask.numel())
    indices = torch.randperm(flat_mask.numel(), device=flat_mask.device)[:const_count]
    flat_mask[indices] = const_value
    return mask

def print_top_diffs(diff, ref, out, topk=10, threshold=0):
    flat_diff = diff.reshape(-1)
    topk = min(topk, flat_diff.numel())
    top_values, top_indices = torch.topk(flat_diff, k=topk)
    flat_ref = ref.reshape(-1)
    flat_out = out.reshape(-1)

    print(f"diff max={diff.max()} mean={diff.mean()}")
    print(f"Top {topk} diff entries:")
    if diff.max() > threshold:
        for rank, (value, flat_index) in enumerate(zip(top_values, top_indices), start=1):
            coord = tuple(int(index.item()) for index in torch.unravel_index(flat_index, diff.shape))
            ref_value = flat_ref[flat_index].item()
            out_value = flat_out[flat_index].item()
            print(
                f"#{rank}: index={coord}, diff={value.item()}, ref={ref_value}, out={out_value}"
            )


def sftm(seq, h, block=-1, dt=torch.float32, dev="xpu"):
    input = torch.rand(seq, h, dtype=dt, device=dev)
    ref = torch.softmax(input, dim=-1)
    print(ref)
    if block != -1:
        block_sum = []
        out = torch.zeros_like(ref)
        for i in range(0, h, block):
            inp = input[:, i : i + block]
            maxinp = inp.max(dim=-1, keepdim=True).values
            inexp = torch.exp(inp - maxinp)
            sum = torch.sum(inexp, dim=-1, keepdim=True)
            out[:, i : i + block] = inexp / sum
            block_sum.append(sum)
        block_sums = torch.concat(block_sum, dim=-1)
        all_sum = torch.sum(block_sums, dim=-1, keepdim=True)
        scale = block_sums / all_sum
        for i in range(len(block_sum)):
            out[:, i * block : (i + 1) * block] = out[:, i * block : (i + 1) * block] * scale[:, i : i + 1]
        print(out)
        diff = abs(ref - out)
        print_top_diffs(diff, ref, out)


def bench_sdpa(seq, seq_kv, h_q, h_kv, H, H_v, dt=torch.float32, dev="xpu", mask_const_count=0, mask_const_value=0.0):
    group = h_q // h_kv
    q = torch.rand(group, h_kv, seq, H, dtype=dt, device=dev)
    k = torch.rand(1, h_kv, H, seq_kv, dtype=dt, device=dev)
    v = torch.rand(1, h_kv, seq_kv, H_v, dtype=dt, device=dev)
    mask = build_mask(seq, seq_kv, dt=dt, dev=dev, const_count=mask_const_count, const_value=mask_const_value)
    scale = 1 / math.sqrt(H)
    n_runs = 100
    for i in range(n_runs):
        score = torch.matmul(q, k) * scale
        score = score + mask
        score = torch.softmax(score, dim=-1)
        out = torch.matmul(score, v)  # group, h_kv, seq, H_v
    if dev == "xpu":
        torch.xpu.synchronize()
    st = time.time()
    for i in range(n_runs):
        score = torch.matmul(q, k) * scale
        score = score + mask
        score = torch.softmax(score, dim=-1)
        out = torch.matmul(score, v)  # group, h_kv, seq, H_v
    if dev == "xpu":
        torch.xpu.synchronize()
    et = time.time()
    dur = (et - st) / n_runs
    OPS = group * h_kv * seq * H * seq_kv * 2  # q*k
    OPS += group * h_kv * seq * seq_kv * 2  # sfmx
    OPS += group * h_kv * seq * H_v * seq_kv * 2  # s*v
    MEM = group * h_kv * seq * H * dt.itemsize  # q
    MEM += h_kv * seq_kv * H * dt.itemsize  # q
    MEM += h_kv * seq_kv * H_v * dt.itemsize  # q
    print(f"Seq_Q:{seq}, Seq_KV:{seq_kv}, HeadNum_Q:{h_q}, HeadNum_KV:{h_kv}, HeadDim_QK:{H}, HeadDim_V:{H_v}")
    print(f"FLOPS:{OPS/dur/1e9:.2f} G, MEM:{MEM/dur/1e9:.2f} GB/s")
    # out = out.reshape(h_q, seq, H_v)
    # out = out.transpose(0, 1)
    # print(out)


def bench_torch(seq, seq_kv, h_q, h_kv, H, H_v, dt=torch.float32, dev="xpu", mask_const_count=0, mask_const_value=0.0):
    group = h_q // h_kv
    q = torch.rand(1, h_q, seq, H, dtype=dt, device=dev)
    k = torch.rand(1, h_kv, seq_kv, H, dtype=dt, device=dev)
    v = torch.rand(1, h_kv, seq_kv, H_v, dtype=dt, device=dev)
    mask = build_mask(seq, seq_kv, dt=dt, dev=dev, const_count=mask_const_count, const_value=mask_const_value)
    scale = 1 / math.sqrt(H)
    n_runs = 100
    for i in range(n_runs):
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask, scale=scale)
    if dev == "xpu":
        torch.xpu.synchronize()
    st = time.time()
    for i in range(n_runs):
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask, scale=scale)
    if dev == "xpu":
        torch.xpu.synchronize()
    et = time.time()
    dur = (et - st) / n_runs
    OPS = group * h_kv * seq * H * seq_kv * 2  # q*k
    OPS += group * h_kv * seq * seq_kv * 2  # sfmx
    OPS += group * h_kv * seq * H_v * seq_kv * 2  # s*v
    MEM = group * h_kv * seq * H * dt.itemsize  # q
    MEM += h_kv * seq_kv * H * dt.itemsize  # q
    MEM += h_kv * seq_kv * H_v * dt.itemsize  # q
    print(f"Seq_Q:{seq}, Seq_KV:{seq_kv}, HeadNum_Q:{h_q}, HeadNum_KV:{h_kv}, HeadDim_QK:{H}, HeadDim_V:{H_v}")
    print(f"FLOPS:{OPS/dur/1e9:.2f} G, MEM:{MEM/dur/1e9:.2f} GB/s")
    # out = out.reshape(h_q, seq, H_v)
    # out = out.transpose(0, 1)
    # print(out)


def bench_ark(batch, seq, seq_kv, h_q, h_kv, H, H_v, dt=torch.float32, dev="xpu", mask_const_count=0, mask_const_value=0.0, has_mask=False, is_causal=False):
    group = h_q // h_kv
    q = torch.rand(batch, h_q, seq, H, dtype=dt, device=dev)
    k = torch.rand(batch, h_kv, seq_kv, H, dtype=dt, device=dev)
    v = torch.rand(batch, h_kv, seq_kv, H_v, dtype=dt, device=dev)
    mask = build_mask(seq, seq_kv, dt=dt, dev=dev, const_count=mask_const_count, const_value=mask_const_value)
    scale = 1 / math.sqrt(H)
    n_runs = 20 if seq_kv > 8192 else 100
    for i in range(n_runs):
        ref = get_ark().sdpa(q, k, v, mask if has_mask else None, softmax_scale=scale, is_causal=is_causal)
    if dev == "xpu":
        torch.xpu.synchronize()
    st = time.time()
    for i in range(n_runs):
        ref = get_ark().sdpa(q, k, v, mask if has_mask else None, softmax_scale=scale, is_causal=is_causal)
    if dev == "xpu":
        torch.xpu.synchronize()
    et = time.time()
    dur = (et - st) / n_runs
    OPS = group * h_kv * seq * H * seq_kv * 2  # q*k
    OPS += group * h_kv * seq * seq_kv * 2  # sfmx
    OPS += group * h_kv * seq * H_v * seq_kv * 2  # s*v
    OPS *= batch
    MEM = group * h_kv * seq * H * dt.itemsize  # q
    MEM += h_kv * seq_kv * H * dt.itemsize  # q
    MEM += h_kv * seq_kv * H_v * dt.itemsize  # q
    MEM *= batch
    print(f"Batch:{batch} Seq_Q:{seq}, Seq_KV:{seq_kv}, HeadNum_Q:{h_q}, HeadNum_KV:{h_kv}, HeadDim_QK:{H}, HeadDim_V:{H_v} Causal:{is_causal}")
    print(f"Time:{dur*1e3:.3f} FLOPS:{OPS/dur/1e9:.2f} G, MEM:{MEM/dur/1e9:.2f} GB/s")
    # out = out.reshape(h_q, seq, H_v)
    # out = out.transpose(0, 1)
    # print(out)


def compare_sdpa(seq, seq_kv, h_q, h_kv, H, H_v, dt=torch.float32, dev="xpu", mask_const_count=0, mask_const_value=0.0):
    group = h_q // h_kv
    q = torch.rand(1, h_q, seq, H, dtype=dt, device=dev)
    k = torch.rand(1, h_kv, seq_kv, H, dtype=dt, device=dev)
    v = torch.rand(1, h_kv, seq_kv, H_v, dtype=dt, device=dev)
    mask = build_mask(seq, seq_kv, dt=dt, dev=dev, const_count=mask_const_count, const_value=mask_const_value)
    mask[:, :] = -100
    scale = 1 / math.sqrt(H)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask, scale=scale)

    q = q.view(group, h_kv, seq, H)
    k = k.transpose(2, 3).contiguous()
    score = torch.matmul(q, k) * scale
    score = score + mask
    score = torch.softmax(score, dim=-1)
    out = torch.matmul(score, v)  # group, h_kv, seq, H_v
    diff = abs(ref - out)
    print_top_diffs(diff, ref, out)


def compare_sdpa_ark(
    batch,
    seq,
    seq_kv,
    h_q,
    h_kv,
    H,
    H_v,
    dt=torch.float32,
    dev="xpu",
    qk_scale_factor=4.0,
    v_scale_factor=100.0,
):
    group = h_q // h_kv
    torch.manual_seed(1234)
    q = torch.randn(batch, h_q, seq, H, dtype=dt, device=dev) * qk_scale_factor
    k = torch.randn(batch, h_kv, seq_kv, H, dtype=dt, device=dev) * qk_scale_factor
    v = torch.randn(batch, h_kv, seq_kv, H_v, dtype=dt, device=dev) * v_scale_factor
    decode = seq == 1
    attn_bias = torch.zeros(seq, seq_kv, dtype=q.dtype, device=q.device)
    temp_mask = torch.ones(seq, seq_kv, dtype=torch.bool, device=q.device).tril(diagonal=seq_kv - seq)
    attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    scale = 1 / math.sqrt(H)
    ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias, scale=scale, enable_gqa=group > 1, is_causal=False
        )
    if not decode:
        out = get_ark().sdpa(q, k, v, attn_bias, softmax_scale=scale, is_causal=False)
        diff = abs(ref - out)
        print_top_diffs(diff, ref, out, topk=4, threshold=1)
        
        out2 = get_ark().sdpa(q, k, v, softmax_scale=scale, is_causal=True)
        diff = abs(out2 - out)
        print_top_diffs(diff, out2, out, topk=4)
    else:
        out = get_ark().sdpa(q, k, v, softmax_scale=scale, is_causal=True)
        diff = abs(ref - out)
        print_top_diffs(diff, ref, out, topk=4, threshold=1)
        
def compare_sdpa_sage(batch, seq, seq_kv, h_q, h_kv, H, H_v, dt=torch.float16, dev="xpu"):
    group = h_q // h_kv
    q = torch.randint(-4, 4, (batch, h_q, seq, H), dtype=torch.int8, device=dev)
    k = torch.randint(-4, 4, (batch, h_kv, seq_kv, H), dtype=torch.int8, device=dev)
    v = torch.rand(batch, h_kv, seq_kv, H_v, dtype=dt, device=dev)*100
    scale = 1 / math.sqrt(H)
    out = get_ark().sdpa(q, k, v, softmax_scale=scale, is_causal=False)
    
    n_runs = 100
    for i in range(n_runs):
        ref = get_ark().sdpa(q, k, v, softmax_scale=scale, is_causal=False)
    if dev == "xpu":
        torch.xpu.synchronize()
    st = time.time()
    for i in range(n_runs):
        ref = get_ark().sdpa(q, k, v, softmax_scale=scale, is_causal=False)
    if dev == "xpu":
        torch.xpu.synchronize()
    et = time.time()
    dur = (et - st) / n_runs
    
    q = q.view(batch, group, h_kv, seq, H).contiguous().to(torch.float32)
    k = k.view(batch, 1, h_kv, seq_kv, H).transpose(3, 4).contiguous().to(torch.float32)
    score = torch.matmul(q, k)
    score = score * scale
    score = torch.softmax(score, dim=-1)
    
    ref = torch.matmul(score.to(dt), v.view(batch, 1, h_kv, seq_kv, H_v)).view(batch, h_q, seq, H)  # group, h_kv, seq, H_v
    dff = abs(ref - out)
    print_top_diffs(dff, ref, out, topk=4, threshold=1)
    
    OPS = group * h_kv * seq * H * seq_kv * 2  # q*k
    OPS += group * h_kv * seq * seq_kv * 2  # sfmx
    OPS += group * h_kv * seq * H_v * seq_kv * 2  # s*v
    OPS *= batch
    MEM = group * h_kv * seq * H * dt.itemsize  # q
    MEM += h_kv * seq_kv * H * dt.itemsize  # q
    MEM += h_kv * seq_kv * H_v * dt.itemsize  # q
    MEM *= batch
    print(f"Batch:{batch} Seq_Q:{seq}, Seq_KV:{seq_kv}, HeadNum_Q:{h_q}, HeadNum_KV:{h_kv}, HeadDim_QK:{H}, HeadDim_V:{H_v}")
    print(f"Time:{dur*1e3:.3f} FLOPS:{OPS/dur/1e9:.2f} G, MEM:{MEM/dur/1e9:.2f} GB/s")

      
def compare_sdpa_sage_scale(batch, seq, seq_kv, h_q, h_kv, H, H_v, dt=torch.float16, is_causal=False, dev="xpu"):
    group = h_q // h_kv
    block_size = 64
    seq = seq//block_size*block_size
    seq_kv = seq_kv//block_size*block_size
    q = torch.randint(-128, 127, (batch, h_q, seq, H), dtype=torch.int8, device=dev)
    k = torch.randint(-128, 127, (batch, h_kv, seq_kv, H), dtype=torch.int8, device=dev)
    if block_size:
        q_scale =torch.randn(batch, h_q, seq//block_size, 1, dtype=torch.float32, device=dev)/100+0.001
        k_scale =torch.randn(batch, h_kv, seq_kv//block_size, 1, dtype=torch.float32, device=dev)/100+0.001
    else:
        q_scale = None
        k_scale = None
    v = torch.rand(batch, h_kv, seq_kv, H_v, dtype=dt, device=dev)*100
    scale = 1 / math.sqrt(H)
    # q_scale[...]=1
    # k_scale[...]=1
    attn_bias = torch.zeros(seq, seq_kv, dtype=v.dtype, device=q.device)
    temp_mask = torch.ones(seq, seq_kv, dtype=torch.bool, device=q.device).tril(diagonal=seq_kv - seq)
    attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    scale = 1 / math.sqrt(H)
    # attn_bias = None

    out = get_ark().sage(q, k, v, quant_block_size=block_size, qscale=q_scale, kscale=k_scale, scale=scale, is_causal=is_causal)

    n_runs = 10 if seq > 8192 else 100
    for i in range(n_runs):
        ref = get_ark().sage(q, k, v, quant_block_size=block_size, qscale=q_scale, kscale=k_scale, scale=scale, is_causal=is_causal)
    if dev == "xpu":
        torch.xpu.synchronize()
    st = time.time()
    for i in range(n_runs):
        ref = get_ark().sage(q, k, v, quant_block_size=block_size, qscale=q_scale, kscale=k_scale, scale=scale, is_causal=is_causal)
    if dev == "xpu":
        torch.xpu.synchronize()
    et = time.time()
    dur = (et - st) / n_runs
    if block_size:
        q_fp = (q.to(torch.float32) * q_scale.repeat_interleave(block_size, dim=2)).to(dt)
        k_fp = (k.to(torch.float32) * k_scale.repeat_interleave(block_size, dim=2)).to(dt)
    else:
        q_fp = q.to(dt)
        k_fp = k.to(dt)
    ref = torch.nn.functional.scaled_dot_product_attention(q_fp, k_fp, v, scale=scale, is_causal=is_causal, enable_gqa=group > 1)
    # out = get_ark().sdpa(q_fp, k_fp, v, softmax_scale=scale, is_causal=is_causal)
    # print(ref)
    dff = abs(ref - out)
    print_top_diffs(dff, ref, out, topk=4, threshold=1)
    
    OPS = group * h_kv * seq * H * seq_kv * 2  # q*k
    OPS += group * h_kv * seq * seq_kv * 2  # sfmx
    OPS += group * h_kv * seq * H_v * seq_kv * 2  # s*v
    OPS *= batch
    MEM = group * h_kv * seq * H * dt.itemsize  # q
    MEM += h_kv * seq_kv * H * dt.itemsize  # q
    MEM += h_kv * seq_kv * H_v * dt.itemsize  # q
    MEM *= batch
    print(f"Batch:{batch} Seq_Q:{seq}, Seq_KV:{seq_kv}, HeadNum_Q:{h_q}, HeadNum_KV:{h_kv}, HeadDim_QK:{H}, HeadDim_V:{H_v}")
    print(f"Time:{dur*1e3:.3f} FLOPS:{OPS/dur/1e9:.2f} G, MEM:{MEM/dur/1e9:.2f} GB/s")


def compare_sdpa_sage_dynquant(batch, seq, seq_kv, h_q, h_kv, H, H_v, dt=torch.float16, is_causal=False, dev="xpu", quant_block_size=64):
    """Test dynamic quantization SAGE attention.

    Q, K, V are FP16 inputs. The kernel dynamically quantizes Q, K to INT8
    """

    group = h_q // h_kv
    q = torch.randn(batch, h_q, seq, H, dtype=dt, device=dev)
    k = torch.randn(batch, h_kv, seq_kv, H, dtype=dt, device=dev)
    v = torch.randn(batch, h_kv, seq_kv, H_v, dtype=dt, device=dev) * 100
    scale = 1 / math.sqrt(H)
    if seq_kv > 4096 and is_causal:
        if seq_kv == seq:  
            ref = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, scale=scale, enable_gqa=group > 1, is_causal=is_causal
                )
        else:
            ref = torch.zeros_like(q)
    else:
        attn_bias = torch.zeros(seq, seq_kv, dtype=q.dtype, device=q.device)
        temp_mask = torch.ones(seq, seq_kv, dtype=torch.bool, device=q.device).tril(diagonal=seq_kv - seq)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        scale = 1 / math.sqrt(H)
        if is_causal:
            ref = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, scale=scale, enable_gqa=group > 1, is_causal=False, attn_mask=attn_bias
                )
        else:
            ref = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, scale=scale, enable_gqa=group > 1, is_causal=is_causal
                )
            

    # Dynamic quantization SAGE kernel
    out = get_ark().sagev1(q, k, v, scale=scale, is_causal=is_causal, quant_block_size=quant_block_size)


    dff = abs(ref - out)
    print(f"=== Dynamic Quantization SAGE Test (block_size={quant_block_size}) ===")
    print(f"Batch:{batch} Seq_Q:{seq}, Seq_KV:{seq_kv}, HeadNum_Q:{h_q}, HeadNum_KV:{h_kv}, HeadDim_QK:{H}, HeadDim_V:{H_v}, Causal:{is_causal}")
    print_top_diffs(dff, ref, out, topk=4, threshold=1)

    # Benchmark
    n_runs = 10 if seq > 8192 else 100
    for i in range(n_runs):
        _ = get_ark().sagev1(q, k, v, scale=scale, is_causal=is_causal, quant_block_size=quant_block_size)
    if dev == "xpu":
        torch.xpu.synchronize()
    st = time.time()
    for i in range(n_runs):
        _ = get_ark().sagev1(q, k, v, scale=scale, is_causal=is_causal, quant_block_size=quant_block_size)
    if dev == "xpu":
        torch.xpu.synchronize()
    et = time.time()
    dur = (et - st) / n_runs

    OPS = group * h_kv * seq * H * seq_kv * 2  # q*k
    OPS += group * h_kv * seq * seq_kv * 2  # sfmx
    OPS += group * h_kv * seq * H_v * seq_kv * 2  # s*v
    OPS *= batch
    MEM = group * h_kv * seq * H * dt.itemsize  # q
    MEM += h_kv * seq_kv * H * dt.itemsize  # k
    MEM += h_kv * seq_kv * H_v * dt.itemsize  # v
    MEM *= batch
    print(f"Time:{dur*1e3:.3f} FLOPS:{OPS/dur/1e9:.2f} G, MEM:{MEM/dur/1e9:.2f} GB/s")
    return out, ref


if __name__ == "__main__":
    # run_ark_sdpa_to_excel(TEST_CASES)
    # sftm(1, 1024, 512)
    # sftm(100, 1024, 512)
    # compare_sdpa(1024, 1024, 32, 32, 128, 128, dt=torch.float16)
    # compare_sdpa(1, 1024, 32, 32, 128, 128, dt=torch.float16)
    # compare_sdpa(
    #     512,
    #     512,
    #     32,
    #     32,
    #     128,
    #     128,
    #     dt=torch.float16
    # )
    # compare_sdpa_ark(
    #     1,
    #     1,
    #     4096,
    #     32,
    #     32,
    #     128,
    #     128,
    #     dt=torch.float16
    # )
    # compare_sdpa_ark(
    #     1,
    #     4096,
    #     8192,
    #     32,
    #     32,
    #     128,
    #     128,
    #     dt=torch.float16
    # )
    # compare_sdpa_ark(
    #     1, 512, 512, 32, 32, 128, 128,
    #     dt=torch.float16
    # )
    # compare_sdpa_ark(2, 1776, 1776, 30, 30, 128, 128, dt=torch.float16)
    # bench_ark(2, 17776, 17776, 30, 30, 64, 64, dt=torch.float16, is_causal=False)
    # bench_ark(1, 4096, 8192*1, 96, 8, 64, 64, dt=torch.float16)
    # bench_ark(1, 4096, 8192*1, 96, 8, 64, 64, dt=torch.float16, is_causal=True)
    # compare_sdpa_sage_scale(1, 8192, 8192, 96, 8, 128, 128, dt=torch.float16)
    # compare_sdpa_sage_scale(1, 8192, 8192, 96, 8, 128, 128, dt=torch.float16, is_causal=True)
    compare_sdpa_sage_dynquant(1, 8192, 8192, 96, 8, 128, 128, dt=torch.float16)
    compare_sdpa_sage_dynquant(1, 8192, 8192, 96, 8, 128, 128, dt=torch.float16, is_causal=True)
    # compare_sdpa_sage_scale(2, 4096, 4096, 96, 8, 128, 128, dt=torch.float16)
    # compare_sdpa_sage_scale(1, 4096, 4096, 96, 8, 128, 128, dt=torch.float16, is_causal=True)
    # compare_sdpa_sage(16, 512, 512, 32, 32, 128, 128, dt=torch.float16)
    # compare_sdpa_ark(16, 512, 512, 32, 32, 128, 128, dt=torch.float16)
    # compare_sdpa_ark(16, 256, 512, 32, 8, 128, 128, dt=torch.float16)
    # compare_sdpa_ark(16, 256, 512, 32, 8, 128, 128, dt=torch.bfloat16)
    # compare_sdpa_ark(16, 1, 512, 32, 8, 128, 128, dt=torch.float16)
    # compare_sdpa_ark(16, 1, 512, 32, 8, 128, 128, dt=torch.bfloat16)
    # # bench_sdpa(1024, 1024, 32, 32, 128, 128, dt=torch.float16)
    # # bench_torch(1024, 1024, 32, 32, 128, 128, dt=torch.float16)
    # bench_ark(2, 17776, 17776, 30, 30, 128, 128, dt=torch.float16, is_causal=False)
    # bench_ark(2, 17776, 17776, 30, 30, 64, 64, dt=torch.float16, is_causal=False)
    # --- Dynamic Quantization SAGE Tests ---
    # compare_sdpa_sage_dynquant(1, 4096, 4096, 32, 32, 128, 128, dt=torch.float16)
    # compare_sdpa_sage_dynquant(2, 4096, 4096, 96, 8, 128, 128, dt=torch.float16)
    # compare_sdpa_sage_dynquant(1, 4096, 4096, 96, 8, 128, 128, dt=torch.float16, is_causal=True)
    # compare_sdpa_sage_dynquant(2, 4096, 4096, 30, 30, 64, 64, dt=torch.float16)
    # bench_ark(1, 4096, 4096, 96, 8, 128, 128, dt=torch.float16, is_causal=True)
    # bench_ark(32, 512, 512, 32, 32, 128, 128, dt=torch.float16)
    # bench_ark(32, 512, 512, 16, 16, 128, 128, dt=torch.bfloat16)
    # bench_ark(32, 1, 512, 32, 8, 128, 128, dt=torch.float16)
    # bench_ark(32, 1, 512, 16, 16, 128, 128, dt=torch.bfloat16)
