#!/usr/bin/env python3

import argparse
import json
import shutil
import tempfile
from pathlib import Path


DEFAULT_MODEL_ROOT = "/storage/yiliu7/nvidia/Cosmos3-Super"
DEFAULT_OUTPUT_DIR = "/storage/yiliu7/nvidia/Cosmos3-Super-W4A16-packed"
DEFAULT_IGNORE_LAYERS = ",".join(
    [
        "embed_tokens",
        "lm_head",
        "time_embedder",
        "proj_in",
        "proj_out",
        "action_proj_in",
        "action_proj_out",
        "audio_proj_in",
        "audio_proj_out",
    ]
)


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize Cosmos3-Super into a packed AutoGPTQ-style folder.")
    parser.add_argument("--model-root", default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-id", default="nvidia/Cosmos3-Super")
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--scheme", default="W4A16")
    parser.add_argument("--format", default="auto_gptq")
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--nsamples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--dataset", default=None, help="Optional local TSV/CSV calibration dataset.")
    parser.add_argument("--ignore-layers", default=DEFAULT_IGNORE_LAYERS)
    parser.add_argument(
        "--load-device-map",
        default=None,
        help="Optional device_map for diffusers loading, e.g. cuda or auto. Default loads on CPU first.",
    )
    parser.add_argument(
        "--quant-device-map",
        default=0,
        help="AutoRound device map. Use 0 for one GPU or auto/0,1,... for larger placements.",
    )
    parser.add_argument("--disable-safety-checker", action="store_true")
    parser.add_argument("--disable-opt-rtn", action="store_true")
    parser.add_argument("--disable-post-copy", action="store_true")
    parser.add_argument("--disable-hydrate-missing-components", action="store_true")
    return parser.parse_args()


def _mode_defaults(mode: str) -> tuple[int, int]:
    if mode == "smoke":
        return 1, 1
    return 50, 8


def _compact_json_string(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        return json.dumps(json.load(handle), separators=(",", ":"), ensure_ascii=False)


def _load_default_prompts(model_root: Path) -> tuple[list[str], str]:
    assets_dir = model_root / "assets"
    prompts = [
        _compact_json_string(assets_dir / "example_t2v_prompt.json"),
        _compact_json_string(assets_dir / "example_t2vs_prompt.json"),
    ]
    short_prompt_path = assets_dir / "example_t2v_prompt_short.txt"
    if short_prompt_path.exists():
        prompts.append(short_prompt_path.read_text(encoding="utf-8").strip())
    negative_prompt = _compact_json_string(assets_dir / "negative_prompt.json")
    return prompts, negative_prompt


def _write_local_tsv(model_root: Path, nsamples: int) -> tuple[str, str]:
    prompts, negative_prompt = _load_default_prompts(model_root)
    tmp_dir = tempfile.mkdtemp(prefix="cosmos3_quant_")
    dataset_path = Path(tmp_dir) / "calibration_prompts.tsv"
    with dataset_path.open("w", encoding="utf-8") as handle:
        handle.write("id\tcaption\n")
        for index in range(nsamples):
            prompt = prompts[index % len(prompts)]
            prompt = prompt.replace("\t", " ").replace("\n", " ")
            handle.write(f"{index}\t{prompt}\n")
    return str(dataset_path), negative_prompt


def _copy_non_transformer_artifacts(model_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in model_root.iterdir():
        if item.name in {"transformer", "model.safetensors.index.json"}:
            continue
        destination = output_dir / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)


def _verify_export(model_root: Path, output_dir: Path) -> None:
    required_paths = [
        output_dir / "model_index.json",
        output_dir / "transformer",
        output_dir / "transformer" / "config.json",
        output_dir / "scheduler",
        output_dir / "text_tokenizer",
        output_dir / "vae",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Packed export is missing required Omni paths: {missing}")

    quant_files = [
        output_dir / "transformer" / "quantize_config.json",
        output_dir / "transformer" / "quantization_config.json",
    ]
    if not any(path.exists() for path in quant_files):
        raise RuntimeError("Packed transformer export is missing quantization config metadata.")

    for item in model_root.iterdir():
        if item.name in {"transformer", "model.safetensors.index.json"}:
            continue
        if not (output_dir / item.name).exists():
            raise RuntimeError(f"Packed export is missing copied artifact '{item.name}'.")


def _build_transformer_quant_config(scheme: str) -> dict:
    from auto_round.schemes import parse_scheme
    from auto_round.version import __version__

    _, _, scheme_dict = parse_scheme(scheme, {})
    return {
        "bits": scheme_dict["bits"],
        "data_type": scheme_dict["data_type"],
        "group_size": scheme_dict["group_size"],
        "sym": scheme_dict["sym"],
        "enable_quanted_input": False,
        "autoround_version": __version__,
        "block_name_to_quantize": "layers",
        "quant_method": "auto-round",
        "packing_format": "auto_round:auto_gptq",
        "extra_config": {
            ".*time_embedder\\.linear_1\\..*": {"bits": 16, "data_type": "float"},
            ".*time_embedder\\.linear_2\\..*": {"bits": 16, "data_type": "float"},
        },
    }


def _normalize_transformer_artifacts(output_dir: Path) -> None:
    from auto_round.utils.model import rename_weights_files

    transformer_dir = output_dir / "transformer"
    if transformer_dir.exists():
        rename_weights_files(str(transformer_dir))

    root_model_index = output_dir / "model.safetensors.index.json"
    if root_model_index.exists():
        root_model_index.unlink()


def _ensure_transformer_metadata(model_root: Path, output_dir: Path, scheme: str) -> None:
    transformer_src = model_root / "transformer" / "config.json"
    transformer_dir = output_dir / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)

    quant_config = _build_transformer_quant_config(scheme)
    transformer_config_path = transformer_dir / "config.json"
    if transformer_config_path.exists():
        with transformer_config_path.open("r", encoding="utf-8") as handle:
            transformer_config = json.load(handle)
    else:
        with transformer_src.open("r", encoding="utf-8") as handle:
            transformer_config = json.load(handle)

    transformer_config.pop("quantization_config", None)
    with transformer_config_path.open("w", encoding="utf-8") as handle:
        json.dump(transformer_config, handle, indent=2)
        handle.write("\n")

    quant_config_path = transformer_dir / "quantization_config.json"
    with quant_config_path.open("w", encoding="utf-8") as handle:
        json.dump(quant_config, handle, indent=2)
        handle.write("\n")


def _hydrate_missing_components(model_root: Path, repo_id: str) -> None:
    from huggingface_hub import hf_hub_download, list_repo_files

    model_index_path = model_root / "model_index.json"
    with model_index_path.open("r", encoding="utf-8") as handle:
        model_index = json.load(handle)

    repo_files = set(list_repo_files(repo_id))
    for component_name, value in model_index.items():
        if component_name.startswith("_") or component_name == "transformer" or not isinstance(value, list):
            continue

        component_dir = model_root / component_name
        if not component_dir.exists():
            continue

        local_files = [path for path in component_dir.rglob("*") if path.is_file()]
        if any(path.name != "config.json" for path in local_files):
            continue

        remote_prefix = f"{component_name}/"
        remote_component_files = [path for path in repo_files if path.startswith(remote_prefix)]
        if not remote_component_files:
            continue

        for remote_path in remote_component_files:
            cached_path = Path(hf_hub_download(repo_id, remote_path))
            destination = model_root / remote_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(cached_path, destination)


def main() -> int:
    args = parse_args()
    model_root = Path(args.model_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not (model_root / "model_index.json").exists():
        raise FileNotFoundError(f"{model_root} does not look like a diffusers pipeline root.")
    if not args.disable_hydrate_missing_components:
        _hydrate_missing_components(model_root, args.repo_id)

    default_iters, default_nsamples = _mode_defaults(args.mode)
    iters = default_iters if args.iters is None else args.iters
    nsamples = default_nsamples if args.nsamples is None else args.nsamples
    disable_opt_rtn = args.disable_opt_rtn or iters == 0

    dataset_tmp_dir = None
    if args.dataset is None:
        dataset_path, negative_prompt = _write_local_tsv(model_root, nsamples)
        dataset_tmp_dir = str(Path(dataset_path).parent)
    else:
        dataset_path = args.dataset
        _, negative_prompt = _load_default_prompts(model_root)

    pipeline_call_kwargs = {
        "negative_prompt": negative_prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "max_sequence_length": args.max_sequence_length,
    }

    try:
        import torch
        from diffusers import Cosmos3OmniPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

        from auto_round import AutoRound
        from auto_round.utils.model import ensure_diffusion_pipeline_components_attr

        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "enable_safety_checker": not args.disable_safety_checker,
        }
        if args.load_device_map:
            load_kwargs["device_map"] = args.load_device_map

        pipe = Cosmos3OmniPipeline.from_pretrained(str(model_root), **load_kwargs)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=args.flow_shift)
        ensure_diffusion_pipeline_components_attr(pipe)

        autoround = AutoRound(
            model=pipe,
            tokenizer=None,
            scheme=args.scheme,
            dataset=dataset_path,
            iters=iters,
            nsamples=nsamples,
            batch_size=args.batch_size,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            pipeline_call_kwargs=pipeline_call_kwargs,
            ignore_layers=args.ignore_layers,
            device_map=args.quant_device_map,
            disable_model_free=True,
            disable_opt_rtn=disable_opt_rtn,
        )
        autoround.quantize_and_save(str(output_dir), format=args.format, inplace=True)

        if not args.disable_post_copy:
            _copy_non_transformer_artifacts(model_root, output_dir)
        _ensure_transformer_metadata(model_root, output_dir, args.scheme)
        _verify_export(model_root, output_dir)
    finally:
        if dataset_tmp_dir is not None:
            shutil.rmtree(dataset_tmp_dir, ignore_errors=True)

    print(f"Packed Cosmos3-Super model saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
