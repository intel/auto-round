from auto_round.quantizers.base import *
from auto_round.quantizers.mode import *
from auto_round.quantizers.model_type import *
from auto_round.quantizers.data_type import *

def create_quantizers():
    # example
    quantizers = {
        QuantizerType.DATA_TYPE: GGUFQuantizer,
        QuantizerType.MODEL_TYPE: LLMQuantizer,
        QuantizerType.MODE: TuneQuantizer,
    }

    dynamic_quantizer = type("AutoRoundQuantizer", tuple(quantizers.values()), {})
    return dynamic_quantizer
    