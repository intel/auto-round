from auto_round_extension.qbits.qlinear_qbits import QuantLinear as QBitsQuantLinear
from auto_round_extension.qbits.qlinear_qbits_gptq import (
    QuantLinear as QBitsGPTQQuantLinear,
)
from auto_round_extension.qbits.qbits_awq import QuantLinear as QBitsAWQQuantLinear

qbits_qlinear_classes = (QBitsQuantLinear, QBitsGPTQQuantLinear)

qbits_awq_classes = (QBitsAWQQuantLinear,)
