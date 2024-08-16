import torch
import transformers


@torch.no_grad()
def calib_forward(model, tokenizer, nsamples=512):
    """Perform calibration for quantization.

    This method calibrates the model for quantization by processing a specified
    number of samples from the calibration dataset. It ensures that the data is
    properly formatted and feeds it to the model. If the number of samples processed
    is less than the specified number, it logs a warning. If no samples are processed,
    it logs an error and exits.
    Args:
        nsamples (int): The number of samples to use for calibration.
        bs (int): The number of samples to use for calibration
    """
    from auto_round.calib_dataset import get_dataloader
    seqlen = 2048
    device = "cuda"
    dataloader = get_dataloader(
        tokenizer,
        seqlen,
    )

    total_cnt = 0
    for data in dataloader:
        if data is None:
            continue
        if isinstance(data, torch.Tensor):
            input_ids = data.to("cuda")
            data_new = input_ids

        elif isinstance(data, str):
            data = tokenizer(data, truncation=True, max_length=seqlen, return_tensors="pt").data
            data_new = {}
            for key in data.keys():
                data_new[key] = data[key].to(device)
            input_ids = data_new["input_ids"]
        else:
            data_new = {}
            for key in data.keys():
                data_new[key] = data[key].to(device)
            input_ids = data_new["input_ids"]
        if input_ids.shape[-1] < seqlen:
            continue

        try:
            if isinstance(data_new, torch.Tensor):
                model(data_new)
            else:
                model(**data_new)
        except NotImplementedError:
            pass
        except Exception as error:
            pass

        total_cnt += input_ids.shape[0]
        if total_cnt >= nsamples:
            break
    if total_cnt == 0:
        print(
            f"no data has been cached, please provide more data with sequence length >={seqlen} in the "
            f"dataset or decease the sequence length"
        )
        exit()
    elif total_cnt < nsamples:
        print(
            f"Insufficient number of samples collected may affect the quantification. "
            f"Valid samples size:{total_cnt}, Target sample size:{nsamples}"
        )


@torch.no_grad()
def get_act_minmax_observer(act_group_size=-1):
    """cache group_wise or channel wise minmax
    :param act_group_size: act group size
    :return: A hook function."""

    def cache_input_hook(module, inputs, outputs):
        input = inputs
        if isinstance(inputs, tuple) or isinstance(input, list):
            input = inputs[0]
        if act_group_size != -1:
            input = input.reshape(-1, act_group_size)
        else:
            input = input.reshape(-1, input.shape[-1])
        current_min = torch.min(input, dim=0)[0]
        current_max = torch.max(input, dim=0)[0]
        if hasattr(module, "act_min"):
            module.act_min = torch.min(current_min, module.act_min) if module.act_min is not None else current_min
            module.act_max = torch.max(current_max, module.act_max) if module.act_max is not None else current_max
        else:
            module.act_min = current_min
            module.act_max = current_max

    return cache_input_hook


def calib(model, tokenizer):
    hook_handels = []
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            hook_func = get_act_minmax_observer()
            hook_handel = m.register_forward_hook(hook_func)
            hook_handels.append(hook_handel)
    calib_forward(model, tokenizer)
    return hook_handels


def get_llama_head_for_scaling(model):
    lm_head = dict(prev_op=model.model.norm, layers=[model.lm_head])
    return lm_head

def get_llama_layers_for_scaling(module):
    layers = []

    # attention input
    layers.append(
        dict(
            prev_op=module.input_layernorm,
            layers=[
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ],

        )
    )

    # attention out
    # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
    if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
        layers.append(
            dict(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.o_proj],
            )
        )

    # linear 1
    layers.append(
        dict(
            prev_op=module.post_attention_layernorm,
            layers=[module.mlp.gate_proj, module.mlp.up_proj],
        )
    )

    # linear 2
    layers.append(
        dict(
            prev_op=module.mlp.up_proj,
            layers=[module.mlp.down_proj],
        )
    )


    return layers


def get_mistral_layers_for_scaling(
        module
):
    layers = []

    # attention input
    layers.append(
        dict(
            prev_op=module.input_layernorm,
            layers=[
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ],
        )
    )

    # attention out
    # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
    if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
        layers.append(
            dict(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.o_proj],
            )
        )

    # linear 1
    layers.append(
        dict(
            prev_op=module.post_attention_layernorm,
            layers=[module.mlp.gate_proj, module.mlp.up_proj],
        )
    )

    # linear 2
    layers.append(
        dict(
            prev_op=module.mlp.up_proj,
            layers=[module.mlp.down_proj],
        )
    )

    return layers


def get_phi3_layers_for_scaling(module):
    layers = []

    # attention input
    layers.append(
        dict(
            prev_op=module.input_layernorm,
            layers=[module.self_attn.qkv_proj],
        )
    )

    # attention out
    layers.append(
        dict(
            prev_op=module.self_attn.qkv_proj,
            layers=[module.self_attn.o_proj],
        )
    )

    # linear 1
    layers.append(
        dict(
            prev_op=module.post_attention_layernorm,
            layers=[module.mlp.gate_up_proj],
        )
    )

    # linear 2
    layers.append(
        dict(
            prev_op=module.mlp.gate_up_proj,
            layers=[module.mlp.down_proj],
        )
    )

    return layers

def get_qwen2_head_for_scaling(model):
    lm_head = dict(prev_op=model.model.norm, layers=[model.lm_head])
    return lm_head

@staticmethod
def get_qwen2_layers_for_scaling(module):
    layers = []

    # attention input
    layers.append(
        dict(
            prev_op=module.input_layernorm,
            layers=[
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ],

        )
    )

    # attention out
    # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
    if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
        layers.append(
            dict(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.o_proj],
            )
        )

    # linear 1
    layers.append(
        dict(
            prev_op=module.post_attention_layernorm,
            layers=[module.mlp.gate_proj, module.mlp.up_proj],
        )
    )

    # linear 2
    layers.append(
        dict(
            prev_op=module.mlp.up_proj,
            layers=[module.mlp.down_proj],
        )
    )

    return layers


def fuse_norm(model):
    model = model.eval()
    if isinstance(model, transformers.models.llama.LlamaForCausalLM):
        func = get_llama_layers_for_scaling
        lm_head_func = get_llama_head_for_scaling
    elif isinstance(model, transformers.models.mistral.MistralForCausalLM):
        func = get_mistral_layers_for_scaling
    # elif isinstance(model, transformers.models.phi3.Phi3ForCausalLM):
    #     func = get_phi3_layers_for_scaling
    elif isinstance(model, transformers.models.qwen2.Qwen2ForCausalLM):
        func = get_qwen2_layers_for_scaling
        lm_head_func = get_qwen2_head_for_scaling
    # elif "Phi3ForCausalLM" in model.__class__.__name__:
    #     absorb_dict = get_phi3_layers_for_scaling
    else:
        assert False, "not supported architecture"

    from auto_round.utils import get_block_names
    from auto_round.utils import get_module
    block_names = get_block_names(model)[0]
    for block_name in block_names:
        module = get_module(model, block_name)
        abosrb_pairs = func(module)
        for pair in abosrb_pairs:
            prev_layer = pair['prev_op']
            layers = pair["layers"]
            if "norm" in prev_layer.__class__.__name__ or "Norm" in prev_layer.__class__.__name__ or "NORM" in prev_layer.__class__.__name__:
                scale = prev_layer.weight
                for layer in layers:
                    layer.weight.data.copy_(layer.weight.data * scale.view(1, -1))
                prev_layer.weight.data.copy_(
                    torch.ones(prev_layer.weight.shape, dtype=prev_layer.weight.dtype, device=prev_layer.weight.device))
    lm_head_paris = lm_head_func(model)
    layers = lm_head_paris["layers"]
    prev_layer = lm_head_paris['prev_op']
    if "norm" in prev_layer.__class__.__name__ or "Norm" in prev_layer.__class__.__name__ or "NORM" in prev_layer.__class__.__name__:
        scale = prev_layer.weight
        for layer in layers:
            layer.weight.data.copy_(layer.weight.data * scale.view(1, -1))
        prev_layer.weight.data.copy_(
            torch.ones(prev_layer.weight.shape, dtype=prev_layer.weight.dtype, device=prev_layer.weight.device))




def convert_sq_model(model, tokenizer, ratio=0.05):
    model = model.to("cuda")
    model = model.eval()
    if isinstance(model, transformers.models.llama.LlamaForCausalLM):
        func = get_llama_layers_for_scaling
    elif isinstance(model, transformers.models.mistral.MistralForCausalLM):
        func = get_mistral_layers_for_scaling
    elif isinstance(model, transformers.models.phi3.Phi3ForCausalLM):
        func = get_phi3_layers_for_scaling
    elif isinstance(model, transformers.models.qwen2.Qwen2ForCausalLM):
        func = get_qwen2_layers_for_scaling
    elif "Phi3ForCausalLM" in model.__class__.__name__:
        func = get_phi3_layers_for_scaling
    else:
        assert False, "not supported architecture"

    hook_handles = calib(model, tokenizer)
    for handle in hook_handles:
        handle.remove()

    from auto_round.utils import get_block_names
    from auto_round.utils import get_module
    block_names = get_block_names(model)[0]
    for block_name in block_names:
        module = get_module(model, block_name)
        absorb_pairs = func(module)
        for pair in absorb_pairs:
            prev_layer = pair['prev_op']
            layers = pair["layers"]
            abs_max = torch.max(torch.abs(layers[0].act_min), torch.abs(layers[0].act_max))
            abs_max_group = abs_max.reshape(-1, 32)
            mean_value = torch.mean(abs_max_group, dim=1, keepdim=True)
            mean_value = torch.repeat_interleave(mean_value, 32, dim=1)
            mean_value = mean_value.reshape(-1)
            abs_max[abs_max == 0] = 1e-5
            scale = 1.0 / torch.sqrt(abs_max / mean_value)
            # import numpy
            # abs_max_np = abs_max.to(torch.float32).cpu().numpy()
            # abs_max_np
            # t = numpy.percentile(abs_max_np, ratio)
            # if t == 0:
            #     t = 1e-5
            # conf = (t / abs_max).to(torch.float32).to("cuda")
            # scale = scale.to("cuda")
            # scale[abs_max > t] = conf[abs_max > t]
            if (
                    "norm" in prev_layer.__class__.__name__ or "Norm" in prev_layer.__class__.__name__ or "NORM" in prev_layer.__class__.__name__):
                prev_layer.weight.data.copy_(
                    prev_layer.weight.data * scale)
            else:
                prev_layer.weight.data.copy_(prev_layer.weight.data * scale.view(-1, 1))

            for layer in layers:
                layer.weight.data.copy_(layer.weight.data / scale.view(1, -1))

    for n, m in model.named_modules():
        if hasattr(m, "act_min"):
            delattr(m, "act_min")
        if hasattr(m, "act_max"):
            delattr(m, "act_max")

    model = model.to("cpu")
