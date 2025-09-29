from typing import Union, Iterable

import torch

from auto_round import AutoScheme


class GenScheme:
    def __init__(self,
                 auto_scheme: AutoScheme,
                 model: torch.nn.Module,
                 quant_layer_names: Iterable[str],
                 fixed_layer_scheme:dict[str, dict],
                 scale_dtype: str = "fp16",
                 dataset="pile-10k"
                 ):
        pass


