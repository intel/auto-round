from auto_round_extension.ipex.qlinear_ipex_awq import QuantLinear as IpexAWQQuantLinear
from auto_round_extension.ipex.qlinear_ipex_gptq import (
    QuantLinear as IpexGPTQQuantLinear,
)

ipex_qlinear_classes = (IpexAWQQuantLinear, IpexGPTQQuantLinear)
