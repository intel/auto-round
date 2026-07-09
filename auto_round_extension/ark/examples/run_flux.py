# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

import contextlib
import gzip
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from flux_sparse_patch import patch_flux_sparse_attention_from_env

dtype = torch.bfloat16
device = "xpu"


def env_flag(name, default="0"):
    return os.getenv(name, default).lower() not in {"0", "false", "no", "off"}


benchmark_enabled = env_flag("FLUX_BENCHMARK_ENABLE", "0")
profiler_enabled = env_flag("FLUX_PROFILER_ENABLE", "0")
benchmark_scope = os.getenv("FLUX_BENCHMARK_SCOPE", "full").strip().lower()

if benchmark_scope not in {"full", "denoising", "block"}:
    raise ValueError("FLUX_BENCHMARK_SCOPE must be one of: full, denoising, block")

benchmark_block_kind = os.getenv("FLUX_BENCHMARK_BLOCK_KIND", "joint").strip().lower()
if benchmark_block_kind not in {"joint", "single"}:
    raise ValueError("FLUX_BENCHMARK_BLOCK_KIND must be one of: joint, single")

benchmark_block_index = int(os.getenv("FLUX_BENCHMARK_BLOCK_INDEX", "0"))
benchmark_block_timestep_index = int(os.getenv("FLUX_BENCHMARK_BLOCK_TIMESTEP_INDEX", "0"))
cpu_offload_enabled = env_flag("FLUX_ENABLE_CPU_OFFLOAD", "0" if benchmark_scope == "block" else "1")

model_id = os.getenv("FLUX_MODEL", "black-forest-labs/FLUX.1-dev")
pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
if benchmark_scope != "block":
    pipe.to(device)
if cpu_offload_enabled:
    pipe.enable_model_cpu_offload()

height = int(os.getenv("FLUX_HEIGHT", "1024"))
width = int(os.getenv("FLUX_WIDTH", "1024"))
num_inference_steps = int(os.getenv("FLUX_STEPS", "50"))
guidance_scale = float(os.getenv("FLUX_GUIDANCE_SCALE", "3.5"))
max_sequence_length = int(os.getenv("FLUX_MAX_SEQUENCE_LENGTH", "512"))
seed = int(os.getenv("FLUX_SEED", "0"))
use_sparse = os.getenv("FLUX_USE_SPARSE", "1").lower() not in {"0", "false", "no", "off"}

output_file = (
    f"flux_output_{height}x{width}_{num_inference_steps}steps_"
    f"{guidance_scale}gs_sparse{os.getenv('FLUX_SPARSE_TOPK', '0.5')}.png"
)
output_path = os.getenv("FLUX_OUTPUT", output_file)

prompt = os.getenv("FLUX_PROMPT", "A cat holding a sign that says hello world")


def create_profiler(enabled, run_tag):
    if not enabled:
        return contextlib.nullcontext(), None, None, None

    xpu_activity = getattr(torch.profiler.ProfilerActivity, "XPU", None)
    if xpu_activity is None:
        raise RuntimeError("FLUX_PROFILER_ENABLE=1 requires torch.profiler.ProfilerActivity.XPU")

    profile_dir = Path(os.getenv("FLUX_PROFILER_DIR", "profiles"))
    profile_dir.mkdir(parents=True, exist_ok=True)
    trace_stem = os.getenv(
        "FLUX_PROFILER_TRACE_NAME",
        f"flux_{run_tag}_{height}x{width}_{num_inference_steps}steps_seed{seed}",
    )
    trace_path = profile_dir / f"{trace_stem}.json.gz"
    summary_path = profile_dir / f"{trace_stem}.txt"
    sort_key = "self_xpu_time_total"
    activities = [torch.profiler.ProfilerActivity.CPU, xpu_activity]
    profiler = torch.profiler.profile(
        activities=activities,
        record_shapes=env_flag("FLUX_PROFILER_RECORD_SHAPES", "1"),
        profile_memory=env_flag("FLUX_PROFILER_PROFILE_MEMORY", "0"),
        with_stack=env_flag("FLUX_PROFILER_WITH_STACK", "1"),
    )
    print(f"[flux_profile] capturing torch trace at {trace_path}")
    return profiler, trace_path, summary_path, sort_key


def save_profiler_results(profiler, trace_path, summary_path, sort_key):
    uncompressed_trace_path = trace_path.with_suffix("")
    profiler.export_chrome_trace(str(uncompressed_trace_path))
    with open(uncompressed_trace_path, "rb") as src, gzip.open(trace_path, "wb") as dst:
        dst.writelines(src)
    uncompressed_trace_path.unlink()
    print(f"[flux_profile] trace saved to {trace_path}")
    if env_flag("FLUX_PROFILER_WRITE_SUMMARY", "0"):
        try:
            summary = profiler.key_averages().table(
                sort_by=sort_key,
                row_limit=int(os.getenv("FLUX_PROFILER_ROW_LIMIT", "40")),
            )
            summary_path.write_text(summary)
            print(f"[flux_profile] summary saved to {summary_path}")
        except UnicodeDecodeError as exc:
            warning = (
                "Skipped profiler summary because PyTorch failed to decode an XPU event "
                f"name while parsing Kineto results: {exc}\n"
                "The compressed Chrome trace was still saved successfully.\n"
            )
            summary_path.write_text(warning)
            print(f"[flux_profile] warning: {warning.strip()}")


def maybe_synchronize_xpu():
    if device == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "synchronize"):
        torch.xpu.synchronize()


def log_sparse_stats(sparse_stats):
    print(
        "[flux_sparse] stats:"
        f" total_calls={sparse_stats.total_calls}"
        f" single_stream_calls={sparse_stats.single_stream_calls}"
        f" joint_stream_calls={sparse_stats.joint_stream_calls}"
        f" sparse_calls={sparse_stats.sparse_calls}"
        f" unsupported_fallbacks={sparse_stats.unsupported_fallbacks}"
        f" runtime_fallbacks={sparse_stats.runtime_fallbacks}"
        f" avg_sparsity={sparse_stats.avg_sparsity:.4f}"
    )
    if sparse_stats.timed_calls:
        print(
            "[flux_sparse][timing] summary:"
            f" timed_calls={sparse_stats.timed_calls}"
            f" avg_ms={sparse_stats.avg_processor_time_ms:.3f}"
            f" min_ms={sparse_stats.processor_time_ms_min:.3f}"
            f" max_ms={sparse_stats.processor_time_ms_max:.3f}"
        )


def common_generation_kwargs():
    return dict(
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        max_sequence_length=max_sequence_length,
        generator=torch.Generator("cpu").manual_seed(seed),
    )


@dataclass
class TransformerTimingStats:
    calls: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    @property
    def avg_ms(self) -> float:
        if self.calls == 0:
            return 0.0
        return self.total_ms / self.calls


@contextlib.contextmanager
def measure_transformer_forward_time(transformer):
    original_forward = transformer.forward
    stats = TransformerTimingStats()

    def wrapped_forward(*args, **kwargs):
        maybe_synchronize_xpu()
        start_time = time.perf_counter()
        output = original_forward(*args, **kwargs)
        maybe_synchronize_xpu()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        stats.calls += 1
        stats.total_ms += elapsed_ms
        stats.min_ms = min(stats.min_ms, elapsed_ms)
        stats.max_ms = max(stats.max_ms, elapsed_ms)
        return output

    transformer.forward = wrapped_forward
    try:
        yield stats
    finally:
        transformer.forward = original_forward


def sparse_patch_context():
    if not use_sparse:
        return contextlib.nullcontext(None)
    print(
        f"[flux_sparse] enabled attention sparse patch: topk={os.getenv('FLUX_SPARSE_TOPK', '0.5')} "
        f"smooth_k={os.getenv('FLUX_SPARSE_SMOOTH_K', '1')}"
        f" q_tile={os.getenv('FLUX_SPARSE_Q_TILE_OVERRIDE', '0')}"
        f" q_block={os.getenv('FLUX_SPARSE_Q_BLOCK_TOKENS', 'default')}"
        f" k_block={os.getenv('FLUX_SPARSE_K_BLOCK_TOKENS', 'default')}"
    )
    return patch_flux_sparse_attention_from_env(pipe.transformer)


def run_generation(output_type="pil"):
    common_kwargs = common_generation_kwargs()
    with sparse_patch_context() as sparse_stats:
        output = pipe(prompt, output_type=output_type, **common_kwargs).images
    if sparse_stats is not None:
        log_sparse_stats(sparse_stats)
    if output_type == "latent":
        return output
    return output[0]


def prepare_denoising_state():
    generation_kwargs = common_generation_kwargs()
    batch_size = 1
    num_images_per_prompt = 1
    execution_device = pipe._execution_device

    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        device=execution_device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=None,
    )

    num_channels_latents = pipe.transformer.config.in_channels // 4
    base_latents, latent_image_ids = pipe.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        execution_device,
        generation_kwargs["generator"],
        latents=None,
    )

    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    if hasattr(pipe.scheduler.config, "use_flow_sigmas") and pipe.scheduler.config.use_flow_sigmas:
        sigmas = None

    image_seq_len = base_latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, _ = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps,
        execution_device,
        sigmas=sigmas,
        mu=mu,
    )

    if pipe.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=execution_device, dtype=torch.float32)
        guidance = guidance.expand(base_latents.shape[0])
    else:
        guidance = None

    return {
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "text_ids": text_ids,
        "base_latents": base_latents,
        "latent_image_ids": latent_image_ids,
        "timesteps": timesteps,
        "guidance": guidance,
        "joint_attention_kwargs": {},
    }


def run_denoising_loop(prepared_state):
    latents = prepared_state["base_latents"].clone()
    prompt_embeds = prepared_state["prompt_embeds"]
    pooled_prompt_embeds = prepared_state["pooled_prompt_embeds"]
    text_ids = prepared_state["text_ids"]
    latent_image_ids = prepared_state["latent_image_ids"]
    timesteps = prepared_state["timesteps"]
    guidance = prepared_state["guidance"]
    joint_attention_kwargs = dict(prepared_state["joint_attention_kwargs"])

    pipe.scheduler.set_begin_index(0)
    for timestep_value in timesteps:
        expanded_timestep = timestep_value.expand(latents.shape[0]).to(latents.dtype)
        with pipe.transformer.cache_context("cond"):
            noise_pred = pipe.transformer(
                hidden_states=latents,
                timestep=expanded_timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=False,
            )[0]

        latents_dtype = latents.dtype
        latents = pipe.scheduler.step(noise_pred, timestep_value, latents, return_dict=False)[0]
        if latents.dtype != latents_dtype:
            latents = latents.to(latents_dtype)

    return latents


def decode_latents_to_image(latents):
    unpacked = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    unpacked = (unpacked / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(unpacked, return_dict=False)[0].detach()
    image = pipe.image_processor.postprocess(image, output_type="pil")
    pipe.maybe_free_model_hooks()
    return image[0]


def run_transformer_only_generation():
    latents = run_generation(output_type="latent")
    return decode_latents_to_image(latents)


def prepare_block_benchmark_state():
    prepared_state = prepare_denoising_state()
    transformer = pipe.transformer
    joint_attention_kwargs = dict(prepared_state["joint_attention_kwargs"])

    joint_blocks = transformer.transformer_blocks
    single_blocks = transformer.single_transformer_blocks
    if benchmark_block_kind == "joint":
        if not 0 <= benchmark_block_index < len(joint_blocks):
            raise ValueError(
                f"FLUX_BENCHMARK_BLOCK_INDEX={benchmark_block_index} is out of range for joint blocks "
                f"(0..{len(joint_blocks) - 1})"
            )
        target_block = joint_blocks[benchmark_block_index]
    else:
        if not 0 <= benchmark_block_index < len(single_blocks):
            raise ValueError(
                f"FLUX_BENCHMARK_BLOCK_INDEX={benchmark_block_index} is out of range for single blocks "
                f"(0..{len(single_blocks) - 1})"
            )
        target_block = single_blocks[benchmark_block_index]

    timesteps = prepared_state["timesteps"]
    if not 0 <= benchmark_block_timestep_index < len(timesteps):
        raise ValueError(
            f"FLUX_BENCHMARK_BLOCK_TIMESTEP_INDEX={benchmark_block_timestep_index} is out of range "
            f"for {len(timesteps)} denoising steps"
        )

    prompt_embeds = prepared_state["prompt_embeds"].to(device)
    pooled_prompt_embeds = prepared_state["pooled_prompt_embeds"].to(device)
    text_ids = prepared_state["text_ids"].to(device)
    latent_image_ids = prepared_state["latent_image_ids"].to(device)
    guidance = prepared_state["guidance"]
    if guidance is not None:
        guidance = guidance.to(device)

    modules_to_move = [
        transformer.x_embedder,
        transformer.context_embedder,
        transformer.time_text_embed,
    ]
    if benchmark_block_kind == "joint":
        modules_to_move.extend(joint_blocks[: benchmark_block_index + 1])
    else:
        modules_to_move.extend(joint_blocks)
        modules_to_move.extend(single_blocks[: benchmark_block_index + 1])
    for module in modules_to_move:
        module.to(device)

    latents = prepared_state["base_latents"].clone().to(device)
    timestep_value = timesteps[benchmark_block_timestep_index]
    timestep = timestep_value.expand(latents.shape[0]).to(device=device, dtype=latents.dtype) / 1000
    hidden_states = transformer.x_embedder(latents)
    timestep_embed = timestep.to(hidden_states.dtype) * 1000
    if guidance is None:
        temb = transformer.time_text_embed(timestep_embed, pooled_prompt_embeds)
    else:
        temb = transformer.time_text_embed(
            timestep_embed,
            guidance.to(hidden_states.dtype) * 1000,
            pooled_prompt_embeds,
        )

    encoder_hidden_states = transformer.context_embedder(prompt_embeds)
    img_ids = latent_image_ids
    if text_ids.ndim == 3:
        text_ids = text_ids[0]
    if img_ids.ndim == 3:
        img_ids = img_ids[0]
    image_rotary_emb = transformer.pos_embed(torch.cat((text_ids, img_ids), dim=0))

    with torch.no_grad():
        with transformer.cache_context("cond"):
            if benchmark_block_kind == "single":
                for block in joint_blocks:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )
            preceding_blocks = (
                joint_blocks[:benchmark_block_index]
                if benchmark_block_kind == "joint"
                else single_blocks[:benchmark_block_index]
            )
            for block in preceding_blocks:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

    return {
        "block": target_block,
        "block_kind": benchmark_block_kind,
        "block_index": benchmark_block_index,
        "timestep_index": benchmark_block_timestep_index,
        "call_kwargs": {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "temb": temb,
            "image_rotary_emb": image_rotary_emb,
            "joint_attention_kwargs": joint_attention_kwargs,
        },
        "input_shapes": {
            "hidden_states": tuple(hidden_states.shape),
            "encoder_hidden_states": tuple(encoder_hidden_states.shape),
            "temb": tuple(temb.shape),
            "text_ids": tuple(text_ids.shape),
            "img_ids": tuple(img_ids.shape),
        },
    }


def run_block_call(block_state):
    with pipe.transformer.cache_context("cond"):
        return block_state["block"](**block_state["call_kwargs"])


def run_block_benchmark(current_run_tag):
    warmup = int(os.getenv("FLUX_BENCHMARK_WARMUP", "2"))
    iters = int(os.getenv("FLUX_BENCHMARK_ITERS", "3"))
    if warmup < 0 or iters <= 0:
        raise ValueError("FLUX_BENCHMARK_WARMUP must be >= 0 and FLUX_BENCHMARK_ITERS must be > 0")

    block_state = prepare_block_benchmark_state()
    benchmark_path = Path(
        os.getenv(
            "FLUX_BENCHMARK_OUTPUT",
            (
                f"flux_benchmark_block_{block_state['block_kind']}{block_state['block_index']}_{current_run_tag}_"
                f"{height}x{width}_{num_inference_steps}steps_seed{seed}_t{block_state['timestep_index']}.json"
            ),
        )
    )
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[flux_bench] benchmarking single FLUX block: warmup={warmup} iters={iters}"
        f" mode={current_run_tag} kind={block_state['block_kind']} index={block_state['block_index']}"
        f" timestep_index={block_state['timestep_index']}"
    )
    print(
        "[flux_bench] block input shapes:"
        f" hidden_states={block_state['input_shapes']['hidden_states']}"
        f" encoder_hidden_states={block_state['input_shapes']['encoder_hidden_states']}"
        f" temb={block_state['input_shapes']['temb']}"
    )
    if cpu_offload_enabled:
        print("[flux_bench] warning: CPU offload is enabled; block benchmark isolation will be weaker")

    with sparse_patch_context() as sparse_stats:
        for idx in range(warmup):
            start_time = time.perf_counter()
            _ = run_block_call(block_state)
            maybe_synchronize_xpu()
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            print(f"[flux_bench] warmup {idx + 1}/{warmup} complete, latency: {latency_ms:.3f} ms")

        latencies_ms = []
        for idx in range(iters):
            start_time = time.perf_counter()
            _ = run_block_call(block_state)
            maybe_synchronize_xpu()
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            latencies_ms.append(latency_ms)
            print(f"[flux_bench] iter {idx + 1}/{iters}: {latency_ms:.3f} ms")

        if sparse_stats is not None:
            log_sparse_stats(sparse_stats)

    result = {
        "scope": "block",
        "mode": current_run_tag,
        "warmup": warmup,
        "iterations": iters,
        "latencies_ms": latencies_ms,
        "avg_ms": statistics.mean(latencies_ms),
        "median_ms": statistics.median(latencies_ms),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "config": {
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "max_sequence_length": max_sequence_length,
            "seed": seed,
            "use_sparse": use_sparse,
            "cpu_offload_enabled": cpu_offload_enabled,
            "block_kind": block_state["block_kind"],
            "block_index": block_state["block_index"],
            "block_timestep_index": block_state["timestep_index"],
            "input_shapes": block_state["input_shapes"],
            "sparse_topk": os.getenv("FLUX_SPARSE_TOPK", "0.5"),
            "sparse_smooth_k": os.getenv("FLUX_SPARSE_SMOOTH_K", "1"),
        },
    }
    benchmark_path.write_text(json.dumps(result, indent=2))
    print(
        f"[flux_bench] avg={result['avg_ms']:.3f} ms median={result['median_ms']:.3f} ms "
        f"min={result['min_ms']:.3f} ms max={result['max_ms']:.3f} ms"
    )
    print(f"[flux_bench] results saved to {benchmark_path}")
    pipe.maybe_free_model_hooks()
    return result


run_tag = "sparse" if use_sparse else "dense"


def maybe_run_with_profiler(run_callable, current_run_tag, record_label="flux_generate"):
    profiler_enabled = env_flag("FLUX_PROFILER_ENABLE", "0")
    profiler, trace_path, summary_path, sort_key = create_profiler(profiler_enabled, current_run_tag)
    with profiler:
        with torch.profiler.record_function(record_label):
            image = run_callable()
        maybe_synchronize_xpu()
    if profiler_enabled:
        save_profiler_results(profiler, trace_path, summary_path, sort_key)
    return image


def run_benchmark(run_callable, current_run_tag, benchmark_scope):
    warmup = int(os.getenv("FLUX_BENCHMARK_WARMUP", "2"))
    iters = int(os.getenv("FLUX_BENCHMARK_ITERS", "3"))
    if warmup < 0 or iters <= 0:
        raise ValueError("FLUX_BENCHMARK_WARMUP must be >= 0 and FLUX_BENCHMARK_ITERS must be > 0")

    benchmark_path = Path(
        os.getenv(
            "FLUX_BENCHMARK_OUTPUT",
            (
                f"flux_benchmark_{current_run_tag}_{height}x{width}_{num_inference_steps}steps_seed{seed}.json"
                if benchmark_scope == "full"
                else f"flux_benchmark_{benchmark_scope}_{current_run_tag}_{height}x{width}_{num_inference_steps}steps_seed{seed}.json"
            ),
        )
    )
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[flux_bench] benchmarking flux workload: warmup={warmup} iters={iters} "
        f"scope={benchmark_scope} mode={current_run_tag} size={height}x{width} steps={num_inference_steps}"
    )
    for idx in range(warmup):
        start_time = time.perf_counter()
        _ = run_callable()
        maybe_synchronize_xpu()
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        print(f"[flux_bench] warmup {idx + 1}/{warmup} complete, latency: {latency_ms:.3f} ms")

    latencies_ms = []
    image = None
    for idx in range(iters):
        start_time = time.perf_counter()
        image = run_callable()
        maybe_synchronize_xpu()
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        latencies_ms.append(latency_ms)
        print(f"[flux_bench] iter {idx + 1}/{iters}: {latency_ms:.3f} ms")

    result = {
        "scope": benchmark_scope,
        "mode": current_run_tag,
        "warmup": warmup,
        "iterations": iters,
        "latencies_ms": latencies_ms,
        "avg_ms": statistics.mean(latencies_ms),
        "median_ms": statistics.median(latencies_ms),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "config": {
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "max_sequence_length": max_sequence_length,
            "seed": seed,
            "use_sparse": use_sparse,
            "sparse_topk": os.getenv("FLUX_SPARSE_TOPK", "0.5"),
            "sparse_smooth_k": os.getenv("FLUX_SPARSE_SMOOTH_K", "1"),
            "output_path": output_path,
        },
    }
    benchmark_path.write_text(json.dumps(result, indent=2))
    print(
        f"[flux_bench] avg={result['avg_ms']:.3f} ms median={result['median_ms']:.3f} ms "
        f"min={result['min_ms']:.3f} ms max={result['max_ms']:.3f} ms"
    )
    print(f"[flux_bench] results saved to {benchmark_path}")
    return image


def run_denoising_benchmark(current_run_tag):
    warmup = int(os.getenv("FLUX_BENCHMARK_WARMUP", "2"))
    iters = int(os.getenv("FLUX_BENCHMARK_ITERS", "3"))
    if warmup < 0 or iters <= 0:
        raise ValueError("FLUX_BENCHMARK_WARMUP must be >= 0 and FLUX_BENCHMARK_ITERS must be > 0")

    benchmark_path = Path(
        os.getenv(
            "FLUX_BENCHMARK_OUTPUT",
            f"flux_benchmark_denoising_{current_run_tag}_{height}x{width}_{num_inference_steps}steps_seed{seed}.json",
        )
    )
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[flux_bench] benchmarking transformer forwards via pipeline latent output: warmup={warmup} iters={iters} "
        f"mode={current_run_tag} size={height}x{width} steps={num_inference_steps}"
    )

    common_kwargs = common_generation_kwargs()
    with sparse_patch_context() as sparse_stats:
        with measure_transformer_forward_time(pipe.transformer) as timing_stats:
            for idx in range(warmup):
                _ = pipe(prompt, output_type="latent", **common_kwargs).images
                print(
                    f"[flux_bench] warmup {idx + 1}/{warmup} complete,"
                    f" transformer_total_ms={timing_stats.total_ms:.3f}"
                )

            latencies_ms = []
            final_latents = None
            prev_total_ms = timing_stats.total_ms
            prev_calls = timing_stats.calls
            for idx in range(iters):
                final_latents = pipe(prompt, output_type="latent", **common_kwargs).images
                iter_total_ms = timing_stats.total_ms - prev_total_ms
                iter_calls = timing_stats.calls - prev_calls
                latencies_ms.append(iter_total_ms)
                prev_total_ms = timing_stats.total_ms
                prev_calls = timing_stats.calls
                print(
                    f"[flux_bench] iter {idx + 1}/{iters}: transformer_total_ms={iter_total_ms:.3f}"
                    f" transformer_calls={iter_calls}"
                )

        if sparse_stats is not None:
            log_sparse_stats(sparse_stats)

    if timing_stats.calls:
        print(
            "[flux_bench] transformer timing summary:"
            f" calls={timing_stats.calls}"
            f" avg_call_ms={timing_stats.avg_ms:.3f}"
            f" min_call_ms={timing_stats.min_ms:.3f}"
            f" max_call_ms={timing_stats.max_ms:.3f}"
        )

    result = {
        "scope": "denoising",
        "mode": current_run_tag,
        "warmup": warmup,
        "iterations": iters,
        "latencies_ms": latencies_ms,
        "avg_ms": statistics.mean(latencies_ms),
        "median_ms": statistics.median(latencies_ms),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "config": {
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "max_sequence_length": max_sequence_length,
            "seed": seed,
            "use_sparse": use_sparse,
            "sparse_topk": os.getenv("FLUX_SPARSE_TOPK", "0.5"),
            "sparse_smooth_k": os.getenv("FLUX_SPARSE_SMOOTH_K", "1"),
            "output_path": output_path,
        },
    }
    benchmark_path.write_text(json.dumps(result, indent=2))
    print(
        f"[flux_bench] avg={result['avg_ms']:.3f} ms median={result['median_ms']:.3f} ms "
        f"min={result['min_ms']:.3f} ms max={result['max_ms']:.3f} ms"
    )
    print(f"[flux_bench] results saved to {benchmark_path}")

    if final_latents is None:
        raise RuntimeError("Denoising benchmark did not produce final latents")
    pipe.maybe_free_model_hooks()
    return decode_latents_to_image(final_latents)


if benchmark_enabled:
    print(f"=======start benchmark for {run_tag} mode=======")
    if benchmark_scope == "full":
        image = run_benchmark(run_generation, run_tag, benchmark_scope)
    elif benchmark_scope == "denoising":
        image = run_denoising_benchmark(run_tag)
    else:
        image = None
        _ = run_block_benchmark(run_tag)
    if profiler_enabled:
        print("[flux_bench] capturing profiler trace in a separate run; benchmark latencies exclude profiler overhead")
        if benchmark_scope == "full":
            profile_callable = run_generation
            profile_label = "flux_full"
        elif benchmark_scope == "denoising":
            profile_callable = run_transformer_only_generation
            profile_label = "flux_denoising"
        else:
            profile_label = "flux_block"
            with sparse_patch_context() as sparse_stats:
                block_profile_state = prepare_block_benchmark_state()
                _ = maybe_run_with_profiler(
                    lambda: run_block_call(block_profile_state),
                    f"{run_tag}_{benchmark_scope}_bench_profile",
                    record_label=profile_label,
                )
            if sparse_stats is not None:
                log_sparse_stats(sparse_stats)
        if benchmark_scope != "block":
            _ = maybe_run_with_profiler(
                profile_callable,
                f"{run_tag}_{benchmark_scope}_bench_profile",
                record_label=profile_label,
            )
else:
    image = maybe_run_with_profiler(run_generation, run_tag)

if image is not None:
    print(f"Saving output to {output_path}")
    image.save(output_path)
