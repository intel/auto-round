import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import logging
from auto_round import AutoRound
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



topologies_config = {
    "ds_mxfp8": {
        "scheme": "MXFP8",
        "fp_layers": "lm_head",
        "iters": 0,
    },
    "ds_mxfp4": {
        "scheme": "MXFP4",
        "fp_layers": "lm_head,self_attn",
        "iters": 0,
    },
    "qwen_mxfp8": {
        "scheme": "MXFP8",
        "fp_layers": "lm_head,mlp.gate",
        "iters": 0,
    },
    "qwen_mxfp4": {
        "scheme": "MXFP4",
        "fp_layers": "lm_head,mlp.gate,self_attn",
        "iters": 0, # TODO: set to 200 before merge
    },
}


def quant_model(args):
    config = topologies_config[args.t]
    
    logger.info(f"Using fp_layers: {config['fp_layers']}")
    autoround = AutoRound(
        model=args.model,
        scheme=config["scheme"],
        enable_torch_compile=args.enable_torch_compile,
        iters=config['iters'],
        fp_layers=config["fp_layers"],
    )
    logger.info(f"Save quantized model to {args.output_dir}")
    format_type = "auto_round" if args.use_autoround_format else "llm_compressor"
    autoround.quantize_and_save(
        format=format_type,
        output_dir=f"quantized_model_{args.t}",
    )


if __name__ == "__main__":
    import argparse

    # import ar_schemes  # Assuming `ar_schemes` is a module in your project
    import auto_round.schemes as ar_schemes

    # Define available schemes
    AVAILABLE_SCHEMES = {
        "MXFP8": "MXFP8",
        "FP8_STATIC": ar_schemes.FP8_STATIC,
        "MXFP8_AR": ar_schemes.MXFP8,
        "MXFP4_AR": ar_schemes.MXFP4,
        "MXFP4": "MXFP4",
        "W4A16": "W4A16",
        "NVFP4": ar_schemes.NVFP4,
    }

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Select a quantization scheme.")
    parser.add_argument(
        "--model",
        type=str,
        help="Path to the pre-trained model or model identifier from Hugging Face Hub.",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        choices=AVAILABLE_SCHEMES.keys(),
        default="MXFP4",
        help="Quantization scheme to use. Available options: " + ", ".join(AVAILABLE_SCHEMES.keys()),
    )
    parser.add_argument(
        "-t",
        type=str,
        choices=topologies_config.keys(),
        default="qwen_mxfp4",
        help="Quantization scheme to use. Available options: " + ", ".join(topologies_config.keys()),
    )

    parser.add_argument(
        "--enable_torch_compile",
        action="store_true",
        help="Enable torch compile for the model.",
    )
    parser.add_argument(
        "--use_autoround_format",
        action="store_true",
        help="Use AutoRound format for saving the quantized model.",
    )

    parser.add_argument(
        "--skip_attn",
        action="store_true",
        help="Skip quantize attention layers.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=0,
        help="Number of iterations for quantization.",
    )
    # output_dir can also be added as an argument if needed
    parser.add_argument(
        "--output_dir",
        type=str,
        default="quantized_model",
        help="Directory to save the quantized model.",
    )

    args = parser.parse_args()

    # Set the scheme based on user input
    scheme = AVAILABLE_SCHEMES[args.scheme]

    # Print the selected scheme for confirmation
    logger.info(f"Selected quantization scheme: {args.scheme}")
    quant_model(args)
