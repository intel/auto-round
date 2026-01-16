

import torch
import torch.nn as nn
from transformers.models.deepseek_v2.modeling_deepseek_v2 import ACT2FN
from transformers.models.deepseek_v2.modular_deepseek_v2 import  DeepseekV2Config, DeepseekV2DecoderLayer, DeepseekV2Attention, DeepseekV2PreTrainedModel
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import auto_docstring
from transformers.utils.import_utils import is_grouped_mm_available
from transformers import PreTrainedModel
from transformers.quantizers import quantizer_finegrained_fp8
from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer
# from transformers.quantizers.finegrained_fp8

class OOTFineGrainedFP8HfQuantizer(FineGrainedFP8HfQuantizer):
    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        **kwargs,
    ):
        return 
    def get_weight_conversions(self):
        return

quantizer_finegrained_fp8.FineGrainedFP8HfQuantizer = OOTFineGrainedFP8HfQuantizer


from auto_round.utils.model import dequant_block_fp8_weight


class FP8Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype=None,
        block_size: tuple[int, int] | None = (128, 128),
        activation_scheme="dynamic",
    ):
        super().__init__(in_features, out_features)

        # If block size is None, it means that we are doing per-tensor quantization
        self.block_size = block_size
        self.activation_scheme = activation_scheme

        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn))

        if self.block_size is None:
            self.weight_scale_inv = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            scale_out_features = (out_features + self.block_size[0] - 1) // self.block_size[0]
            scale_in_features = (in_features + self.block_size[1] - 1) // self.block_size[1]
            self.weight_scale_inv = nn.Parameter(
                torch.empty(scale_out_features, scale_in_features, dtype=torch.float32)
            )

        if self.activation_scheme == "static":
            self.activation_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)

    def qdq_input(self, bf16_input: torch.Tensor):
        input_scale, input_fp8 = quant_tensor(bf16_input)
        qdq_input_bf16 = input_fp8.to(bf16_input.dtype) * input_scale
        return qdq_input_bf16
    

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        dequant_weight = dequant_block_fp8_weight(
            self.weight,
            self.weight_scale_inv,
            block_size=self.block_size,
        )
        dequant_weight = dequant_weight.to(input.dtype)
        # input = self.qdq_input(input)
        out = torch.nn.functional.linear(input, dequant_weight, self.bias)
        return out.to(input.dtype)


from torch import nn




# from _fp8_quant/_core/fp_utils.py
def calc_maxabs_scale(xmaxabs, fullscale, backoff=1):
    scale = xmaxabs / (fullscale * backoff)
    return scale

SCALE_DTYPE = torch.float32
WEIGHT_BACKOFF = 1.0
FULL_RANGE =  torch.finfo(torch.float8_e4m3fn).max
def quant_tensor(tensor):
    # Note:
    #  1. Check the scale dtype
    #  2. Check the scale shape
    amax = tensor.abs().max()
    scale = calc_maxabs_scale(amax, FULL_RANGE, WEIGHT_BACKOFF)
    scale = scale.to(SCALE_DTYPE)
    qtensor = tensor / scale
    cliped_qtensor = torch.clamp(qtensor, -FULL_RANGE, FULL_RANGE)
    cliped_qtensor_fp8 = cliped_qtensor.to(torch.float8_e4m3fn)
    return scale, cliped_qtensor_fp8


# def _maybe_create_dir(qmodel_path):
#     if not os.path.exists(qmodel_path):
#         os.makedirs(qmodel_path)

# Adapted from https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/1d044fd82b15f1cedb197a288e50cc96a2c27205/inference/model.py#L91-L108
class FP8QDQLinear(torch.nn.Linear):
    dtype = torch.bfloat16
    fp8_dtype = torch.float8_e4m3fn

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None):
        super().__init__(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=FP8QDQLinear.fp8_dtype), requires_grad=True
        )
        self.weight_scale_inv = nn.Parameter(torch.tensor(0, dtype=FP8QDQLinear.dtype), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def dequant_weight_online(self):
        fp8_weight = self.weight
        qdq_weight = fp8_weight.to(FP8QDQLinear.dtype) #* self.weight_scale_inv
        return qdq_weight

    def qdq_input(self, bf16_input: torch.Tensor):
        input_scale, input_fp8 = quant_tensor(bf16_input)
        qdq_input_bf16 = input_fp8.to(FP8QDQLinear.dtype) * input_scale
        return qdq_input_bf16

    @classmethod
    def create_from_linear(cls, linear: nn.Linear):
        qdq_linear = cls(linear.in_features, linear.out_features)
        qdq_linear.weight.data = linear.weight.data
        if linear.bias is not None:
            qdq_linear.bias = linear.bias
        return qdq_linear

    def forward(self, bf16_input: torch.Tensor) -> torch.Tensor:
        qdq_input = self.qdq_input(bf16_input)
        qdq_weight = self.dequant_weight_online()
        out = torch.nn.functional.linear(qdq_input, qdq_weight, self.bias)
        return out



# @use_experts_implementation
class DeepseekV2ExpertsFix(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()


        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue

            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            if top_k_weights.shape[1] == self.num_experts:
                weights = top_k_weights[token_idx, expert_idx]
            else:
                weights = top_k_weights[token_idx, top_k_pos]
            current_hidden_states = current_hidden_states * weights[:, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states
        

class _DeepseekV2ExpertMLPBF16(nn.Module):
    """Single expert GLU-style MLP: down_proj(act(gate_proj(x)) * up_proj(x))."""

    def __init__(self, config: DeepseekV2Config):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.moe_intermediate_size
        # Bias disabled to mirror packed-weights behavior
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _DeepseekV2ExpertMLPFP8(_DeepseekV2ExpertMLPBF16):
    def __init__(self, config: DeepseekV2Config):
        super().__init__(config)
        hidden_size = config.hidden_size
        intermediate_size = config.moe_intermediate_size
        block_size = (128, 128)
        self.gate_proj = FP8Linear(hidden_size, intermediate_size, bias=False, block_size=block_size)
        self.up_proj = FP8Linear(hidden_size, intermediate_size, bias=False, block_size=block_size)
        self.down_proj = FP8Linear(intermediate_size, hidden_size, bias=False, block_size=block_size)

# @use_experts_implementation
class _DeepseekV2ExpertsMLP(nn.ModuleList):
    """MoE experts implemented directly as a ModuleList of per-expert MLPs.

    Functional equivalent to `DeepseekV2Experts` (packed weights) but implemented
    with explicit Linear layers per expert. Forward signature is identical.
    """

    def __init__(self, config: DeepseekV2Config):
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        experts = [_DeepseekV2ExpertMLP(config) for _ in range(self.num_experts)]
        super().__init__(experts)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)

        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            # Run expert MLP
            current_hidden_states = self[expert_idx](current_state)
            # Weight by router scores (supports [N, top_k] or [N, num_experts])
            if top_k_weights.shape[1] == self.num_experts:
                weights = top_k_weights[token_idx, expert_idx]
            else:
                weights = top_k_weights[token_idx, top_k_pos]
            current_hidden_states = current_hidden_states * weights[:, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

from loguru import logger

# implement warning_once
from functools import lru_cache
@lru_cache(maxsize=None)
def warning_once(message: str):
    logger.warning(message)


class DeepseekV3MLPBF16(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        warning_once("DeepseekV3MLP forward pass is called.")
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
    
class DeepseekV3MLPFP8(DeepseekV3MLPBF16):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__(config, hidden_size, intermediate_size)
        block_size = (128, 128)
        self.gate_proj = FP8Linear(self.hidden_size, self.intermediate_size, bias=False, block_size=block_size)
        self.up_proj = FP8Linear(self.hidden_size, self.intermediate_size, bias=False, block_size=block_size)
        self.down_proj = FP8Linear(self.intermediate_size, self.hidden_size, bias=False, block_size=block_size)

# DeepseekV3MLP = 
DeepseekV3MLP = DeepseekV3MLPBF16
torch.nn.Linear = FP8Linear
# torch.nn.Linear = FP8QDQLinear


class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(
                torch.empty((self.n_routed_experts))
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        return 
        import torch.nn.init as init
        
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = torch.nn.functional.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )
        if self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        ### select top-k experts
        if self.topk_method == "noaux_tc":
            assert not self.training
            scores_for_choice = scores.view(bsz * seq_len, -1) + self.e_score_correction_bias.unsqueeze(0)
            group_scores = (
                scores_for_choice.view(bsz * seq_len, self.n_group, -1).topk(2, dim=-1)[0].sum(dim = -1)
            )  # [n, n_group]
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[
                1
            ]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group
                )
                .reshape(bsz * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]
            _, topk_idx = torch.topk(
                tmp_scores, k=self.top_k, dim=-1, sorted=False
            )
            topk_weight = scores.gather(1, topk_idx)
        else:
            raise NotImplementedError(
                f"insupportable TopK function for MoE gating: {self.topk_method}"
            )

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor # must multiply the scaling factor

        return topk_idx, topk_weight


class DeepseekV3MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        if hasattr(config, "ep_size") and config.ep_size > 1:
            assert config.ep_size == dist.get_world_size()
            self.ep_size = config.ep_size
            self.experts_per_rank = config.n_routed_experts // config.ep_size
            self.ep_rank = dist.get_rank()
            self.experts = nn.ModuleList(
                [
                    (
                        DeepseekV3MLP(
                            config, intermediate_size=config.moe_intermediate_size
                        )
                        if i >= self.ep_rank * self.experts_per_rank
                        and i < (self.ep_rank + 1) * self.experts_per_rank
                        else None
                    )
                    for i in range(config.n_routed_experts)
                ]
            )
        else:
            self.ep_size = 1
            self.experts_per_rank = config.n_routed_experts
            self.ep_rank = 0
            self.experts = nn.ModuleList(
                [
                    DeepseekV3MLP(
                        config, intermediate_size=config.moe_intermediate_size
                    )
                    for i in range(config.n_routed_experts)
                ]
            )
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV3MLP(
                config=config, intermediate_size=intermediate_size
            )

    def forward(self, hidden_states):
        warning_once("DeepseekV3MoE is experimental and the implementation may change in the future.")
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if not self.training:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        sorted_tokens_shape = sorted_tokens.shape
        if self.ep_size > 1:
            tokens_per_ep_rank = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)
            tokens_per_expert_group = tokens_per_expert.new_empty(
                tokens_per_expert.shape[0]
            )
            dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert)
            output_splits = (
                tokens_per_expert_group.view(self.ep_size, -1)
                .sum(1)
                .cpu()
                .numpy()
                .tolist()
            )
            gathered_tokens = sorted_tokens.new_empty(
                tokens_per_expert_group.sum(dim=0).cpu().item(), sorted_tokens.shape[1]
            )
            input_split_sizes = tokens_per_ep_rank.cpu().numpy().tolist()
            dist.all_to_all(
                list(gathered_tokens.split(output_splits)),
                list(sorted_tokens.split(input_split_sizes)),
            )
            tokens_per_expert_post_gather = tokens_per_expert_group.view(
                self.ep_size, self.experts_per_rank
            ).sum(dim=0)
            gatherd_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int32)
            s = 0
            for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
                gatherd_idxs[s : s + k] = i % self.experts_per_rank
                s += k
            gatherd_idxs = gatherd_idxs.argsort()
            sorted_tokens = gathered_tokens[gatherd_idxs]
            tokens_per_expert = tokens_per_expert_post_gather
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        if self.ep_size > 1:
            new_x = torch.empty_like(outs)
            new_x[gatherd_idxs] = outs
            gathered_tokens = new_x.new_empty(*sorted_tokens_shape)
            dist.all_to_all(
                list(gathered_tokens.split(input_split_sizes)),
                list(new_x.split(output_splits)),
            )
            outs = gathered_tokens

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out




class _DeepseekV2PreTrainedModel(DeepseekV2PreTrainedModel):

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        # if isinstance(module, DeepseekV2Experts):
        #     init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
        #     init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)



from transformers.models.deepseek_v3.modular_deepseek_v3 import DeepseekV3PreTrainedModel, DeepseekV3TopkRouter, DeepseekV3NaiveMoe
class _DeepseekV3PreTrainedModel(DeepseekV3PreTrainedModel):


    @torch.no_grad()
    def _init_weights(self, module):
        # breakpoint()
        pass
        # if isinstance(module, DeepseekV3TopkRouter):
        #     # init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        #     init.zeros_(module.e_score_correction_bias)
        # elif isinstance(module, DeepseekV3NaiveMoe):
        #     pass
        #     # init.zeros_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
        #     # init.zeros_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        # else:
        #     super()._init_weights(module)



def apply_ds_v2_fixes():
    # monkey patching
    import transformers.models.deepseek_v2.modular_deepseek_v2 as modular_deepseek_v2
    modular_deepseek_v2.DeepseekV2PreTrainedModel = _DeepseekV2PreTrainedModel
    # modeling_deepseek_v2.DeepseekV2Experts = DeepseekV2ExpertsFix
    modular_deepseek_v2.DeepseekV2Experts = _DeepseekV2ExpertsMLP


# # def apply_ds_v3_fixes():
# # monkey patching

# import transformers.models.deepseek_v3.modular_deepseek_v3 as modular_deepseek_v3
# import transformers.models.deepseek_v3.modeling_deepseek_v3 as modeling_deepseek_v3
# # modular_deepseek_v3.DeepseekV3NaiveMoe = _MixtralExpertsMLP
# # modeling_deepseek_v3.DeepseekV3NaiveMoe = _MixtralExpertsMLP
# # from ds_v2 import _DeepseekV2ExpertsMLP
# # modular_deepseek_v3.DeepseekV3NaiveMoe = _DeepseekV2ExpertsMLP
# # modeling_deepseek_v3.DeepseekV3NaiveMoe = _DeepseekV2ExpertsMLP
# modeling_deepseek_v3.DeepseekV3MoE = DeepseekV3MoE
# modular_deepseek_v3.DeepseekV3PreTrainedModel = _DeepseekV3PreTrainedModel
# modeling_deepseek_v3.DeepseekV3PreTrainedModel = _DeepseekV3PreTrainedModel
# print("Applied DeepseekV3NaiveMoe monkey patching.")
# # MixtralExperts = _MixtralExpertsMLP  # Switch between implementations here