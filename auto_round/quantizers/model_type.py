from auto_round.quantizers.base import BaseQuantizer, QuantizerType

class ModeQuantizer(BaseQuantizer):
    quantizer_type = QuantizerType.MODEL_TYPE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
@BaseQuantizer.register("llm")
class LLMQuantizer(ModeQuantizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@BaseQuantizer.register("vlm")
class VLMQuantizer(ModeQuantizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)