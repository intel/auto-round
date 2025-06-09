import torch
import os
import sys
import argparse
import datetime
import logging
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from auto_round_diff.utils import (
    clear_memory,
    logger,
    set_cuda_visible_devices,
    get_device_and_parallelism
    )


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable_deterministic_algorithms", action='store_true',
                          help="disable torch deterministic algorithms.")
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--prompts_path",
        type=str,
        nargs="?",
        help="dir to write results to",
        # required=True
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    # linear quantization configs
    # parser.add_argument(
    #     "--ptq", action="store_true", help="apply post-training quantization"
    # )
    parser.add_argument(
        "--quant_act", action="store_true", 
        help="if to quantize activations when ptq==True"
    )
    parser.add_argument(
        "--weight_bits",
        type=int,
        default=8,
        help="int bit for weight quantization",
    )
    parser.add_argument(
        "--w_quant_granularity",
        type=str,
        default="channel_wise",
        help="weight quantization granularity",
    )
    parser.add_argument(
        "--w_group_size",
        type=int,
        default=128,
        help="group size for weight group quantization"
    )
    parser.add_argument(
        "--weight_asym",
        type=bool,
        default=True,
        help="weight quantization asymmetric"
    )
    parser.add_argument(
        "--w_data_type",
        type=str,
        default='int',
        help="data type for weight quantization"
    )
    parser.add_argument(
        "--w_scale_method",
        type=str,
        default='max',
        help='algorithm for initializing weight quant params'
    )
    parser.add_argument(
        "--act_quant_granularity",
        type=str,
        default="channel_wise",
        help="weight quantization granularity",
    )
    parser.add_argument(
        "--act_group_size",
        type=int,
        default=128,
        help="group size for weight group quantization"
    )
    parser.add_argument(
        "--act_bits",
        type=int,
        default=8,
        help="int bit for activation quantization",
    )
    parser.add_argument(
        "--act_asym",
        type=bool,
        default=True,
        help="activation quantization asymmetric"
    )
    parser.add_argument(
        "--act_dynamic",
        action="store_true",
        help="use dynamic quantization for activation"
    )
    parser.add_argument(
        "--act_data_type",
        type=str,
        default='int',
        help="data type for activation quantization"
    )
    parser.add_argument(
        "--act_scale_method",
        type=str,
        default='max',
        help='algorithm for initializing activation quant params '
    )
    parser.add_argument(
        "--enable_quanted_input",
        type=bool,
        default=True,
        help="enable quanted input for construction"
    )
    parser.add_argument(
        "--cali_st", type=int, default=1, 
        help="number of timesteps used for calibration"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, 
        help="batch size for qdiff reconstruction"
    )
    parser.add_argument(
        "--cali_n", type=int, default=1024, 
        help="number of samples for each timestep for qdiff reconstruction"
    )
    parser.add_argument(
        "--cali_iters_w", type=int, default=20000, 
        help="number of iterations for each qdiff reconstruction"
    )
    parser.add_argument('--cali_iters_a', default=5000, type=int, 
        help='number of iteration for LSQ')
    parser.add_argument('--cali_lr', default=4e-4, type=float, 
        help='learning rate for LSQ')
    parser.add_argument('--cali_p', default=2.4, type=float, 
        help='L_p norm minimization for LSQ')
    parser.add_argument(
        "--cali_ckpt", type=str,
        help="path for calibrated model ckpt"
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="tune weights and activation using the adaround/autoround algorithm"
    )
    parser.add_argument(
        "--w_lr",
        type=float,
        default=4e-5,
        help="learning rate for weight tuning"
    )
    parser.add_argument(
        "--a_lr",
        type=float,
        default=4e-4,
        help="learning rate for activation tuning"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="resume the calibrated qdiff model"
    )
    parser.add_argument(
        "--resume_w", action="store_true",
        help="resume the calibrated qdiff model weights only"
    )
    parser.add_argument(
        "--cond", action="store_true",
        help="whether to use conditional guidance"
    )
    parser.add_argument(
        "--no_grad_ckpt", action="store_true",
        help="disable gradient checkpointing"
    )
    parser.add_argument(
        "--split", action="store_true",
        help="use split strategy in skip connection"
    )
    parser.add_argument(
        "--cali_data_path", 
        type=str,
        required=True,
        help="cali data for quant"
    )
    parser.add_argument(
        "--running_stat", action="store_true",
        help="use running statistics for act quantizers"
    )
    parser.add_argument(
        "--rs_sm_only", action="store_true",
        help="use running statistics only for softmax act quantizers"
    )
    parser.add_argument(
        "--sm_abit",type=int, default=8,
        help="attn softmax activation bit"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="print out info like quantized model arch"
    )
    parser.add_argument(
        "--round_type",
        type=str,
        default="adaround",
        help="algorithm for round operation"
    )
    parser.add_argument(
            "--device",
            "--devices",
            default="0",
            type=str,
            help="the device to be used for tuning. "
                 "Currently, device settings support CPU, GPU, and HPU."
                 "The default is set to cuda:0,"
                 "allowing for automatic detection and switch to HPU or CPU."
                 "set --device 0,1,2 to use multiple cards.")

    args = parser.parse_args()
    return args

def set_logger(args):
    
    os.makedirs(args.outdir, exist_ok=True)
    log_path = os.path.join(args.outdir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(log_path)

    log_path = os.path.join(log_path, "run.log")
    sh = logging.FileHandler(log_path)

    from auto_round_diff.utils import AutoRoundFormatter
    sh.setFormatter(AutoRoundFormatter())
    logger.addHandler(sh)

def load_model_from_config(config, ckpt, verbose=False):
    logging.info(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        logging.info(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        logging.info("missing keys:")
        logging.info(m)
    if len(u) > 0 and verbose:
        logging.info("unexpected keys:")
        logging.info(u)

    model.cuda()
    model.eval()
    return model

def tune(args):

    ##must set this before import torch
    set_cuda_visible_devices(args.device)
    device_str, use_auto_mapping = get_device_and_parallelism(args.device)

    if not args.disable_deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("'torch.use_deterministic_algorithms' is turned on by default for reproducibility, " \
              "and can be turned off by setting the '--disable_deterministic_algorithms' parameter.")

    # load model
    config = OmegaConf.load(f"{args.config}")
    model = load_model_from_config(config, f"{args.ckpt}")

    from auto_round_diff.diffusion.autoround_diffusion import AdaRoundUnetDiffusion
    model = model.eval()

    if args.round_type == 'autoround':
        # round = AutoRoundDiffusion(
        #     model,
        #     prompts_path=args.prompts_path, # cali prompts
        #     weight_bit=args.weight_bit,
        #     quant_granularity=args.quant_granularity,
        #     weight_sym=not args.weight_asym,
        #     batch_size=args.batch_size,
        #     cali_iters_w=args.cali_iters_w,
        #     lr=args.lr,
        #     amp=not args.disable_amp,
        #     enable_quanted_input=not args.disable_quanted_input,
        #     truncation=args.truncation,
        #     nsamples=args.nsamples,
        #     low_gpu_mem_usage=args.low_gpu_mem_usage,
        #     device=device_str,
        #     seed=args.seed,
        #     gradient_accumulate_steps=args.gradient_accumulate_steps,
        #     scale_dtype=args.scale_dtype,
        #     layer_config=layer_config,
        #     template=args.template,
        #     enable_minmax_tuning=not args.disable_minmax_tuning,
        #     act_bits=args.act_bits,
        #     quant_nontext_module=args.quant_nontext_module,
        #     not_use_best_mse=args.not_use_best_mse,
        #     to_quant_block_names=args.to_quant_block_names,
        #     enable_torch_compile=enable_torch_compile,
        #     device_map=args.device_map,
        #     model_kwargs=model_kwargs
        # )
        # model, _ = autoround.quantize()
        # round = AutoRoundDiffusion()
        pass
    elif args.round_type == 'adaround':
        round = AdaRoundUnetDiffusion(
            model,
            prompts_path=args.prompts_path, # cali prompts
            weight_bits=args.weight_bits,
            w_quant_granularity=args.w_quant_granularity,
            w_group_size=args.w_group_size,
            weight_sym=not args.weight_asym,
            w_data_type=args.w_data_type,
            w_scale_method=args.w_scale_method,
            tune=args.tune,
            batch_size=args.batch_size,
            cali_iters_w=args.cali_iters_w,
            quant_act=args.quant_act,
            act_bits=args.act_bits,
            act_quant_granularity=args.w_quant_granularity,
            act_group_size=args.w_group_size,
            act_sym=not args.act_asym,
            act_dynamic=args.act_dynamic,
            act_data_type=args.act_data_type,
            act_scale_method=args.act_scale_method,
            running_stat=args.running_stat,
            sm_abit=args.sm_abit,
            cali_iters_a=args.cali_iters_a,
            cali_n=args.cali_n,
            cali_data_path=args.cali_data_path,
            a_lr=args.a_lr,
            rs_sm_only=args.rs_sm_only,
            w_lr=args.w_lr,
            enable_quanted_input=args.enable_quanted_input,
            device=device_str,
            seed=args.seed,
            split=args.split,
            resume_w=args.resume_w
        )
    else:
        raise NotImplementedError("This round algorithm has not been implemented yet.")

    model, _ = round.quantize()

    model.eval()
    clear_memory()

    