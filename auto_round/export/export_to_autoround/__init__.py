import transformers
from .autogptq_backend import AutoHfQuantizer

transformers.quantizers.auto.AutoHfQuantizer = AutoHfQuantizer
transformers.quantizers.auto.AutoQuantizationConfig = AutoHfQuantizer
transformers.modeling_utils.AutoHfQuantizer = AutoHfQuantizer
from transformers import AutoModelForCausalLM as AutoRoundModelForCausalLM
