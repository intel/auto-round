import os

from dataclasses import dataclass


@dataclass
class GlobalConfig:
    FP8_INPUT_BACKOFF: float = 1.0
    FP8_WEIGHT_BACKOFF: float = 1.0
    # enbale weight_fp8_max_scale
    ENABLE_WEIGHT_FP8_MAX_SCALE: bool = False
    W4A8_PC: bool = False
    W4A8_DYNAMIC: bool = False
    
    def __repr__(self):
        return (
            f"GlobalConfig(FP8_INPUT_BACKOFF={self.FP8_INPUT_BACKOFF}, "
            f"FP8_WEIGHT_BACKOFF={self.FP8_WEIGHT_BACKOFF}, "
            f"ENABLE_WEIGHT_FP8_MAX_SCALE={self.ENABLE_WEIGHT_FP8_MAX_SCALE}, "
            f"W4A8_PC={self.W4A8_PC}, "
            f"W4A8_DYNAMIC={self.W4A8_DYNAMIC}"
            f")"
            
        )


# https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_FP8.html?highlight=fp8#configuring-backoff-factors
# The default values are input_backoff=0.25 and weight_backoff=0.5
global_config = GlobalConfig()
global_config.FP8_INPUT_BACKOFF = float(os.environ.get("AR_FP8_INPUT_BACKOFF", 1.0))
global_config.FP8_WEIGHT_BACKOFF = float(os.environ.get("AR_FP8_WEIGHT_BACKOFF", 1.0))
global_config.ENABLE_WEIGHT_FP8_MAX_SCALE = os.environ.get("AR_ENABLE_WEIGHT_FP8_MAX_SCALE", "0") == "1"
global_config.W4A8_PC = os.environ.get("W4A8_PC", "0") == "1"
global_config.W4A8_DYNAMIC = os.environ.get("W4A8_DYNAMIC", "0") == "1"

print(global_config)
