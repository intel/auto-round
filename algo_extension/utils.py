import torch


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