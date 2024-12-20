import os

from dataclasses import dataclass


@dataclass
class GlobalConfig:
    FP8_INPUT_BACKOFF: float = 1.0
    FP8_WEIGHT_BACKOFF: float = 1.0


# https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_FP8.html?highlight=fp8#configuring-backoff-factors
# The default values are input_backoff=0.25 and weight_backoff=0.5
global_config = GlobalConfig()
global_config.FP8_INPUT_BACKOFF = float(os.environ.get("AR_FP8_INPUT_BACKOFF", 1.0))
global_config.FP8_WEIGHT_BACKOFF = float(os.environ.get("AR_FP8_WEIGHT_BACKOFF", 1.0))

from loguru import logger

logger.info(f"Global config: {global_config}")

inc_default_config = GlobalConfig(FP8_INPUT_BACKOFF=0.25, FP8_WEIGHT_BACKOFF=0.5)
config4_in_result_table = inc_default_config
config5_in_result_table = GlobalConfig(FP8_INPUT_BACKOFF=0.25, FP8_WEIGHT_BACKOFF=1.0)
