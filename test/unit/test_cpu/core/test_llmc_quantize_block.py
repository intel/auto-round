from types import SimpleNamespace

import torch

from auto_round.compressors.data_driven import DataDrivenCompressor


class _QuantizerStub:
    batch_size = 1
    infer_bs_coeff = 1
    enable_quanted_input = False
    act_data_type = ""

    def register_calibration_hooks(self, block, *, act_max=True, imatrix=True):
        return []

    def _get_block_outputs(self, block, input_ids, input_others, bs, save_output=True):
        return torch.ones(1, 1)

    def quantize_block(
        self, block, input_ids, input_others, reference_output, loss_device=None, mid_iter_mem_check=False
    ):
        return None


def test_llmc_quantize_block_allows_mllm(monkeypatch):
    compressor = object.__new__(DataDrivenCompressor)
    compressor.model_context = SimpleNamespace(is_mllm=True, is_diffusion=False, amp_dtype=torch.float32)
    compressor._post_init_done = True
    compressor._calibration_state = SimpleNamespace(inputs={})
    compressor.compress_context = SimpleNamespace(
        device_map="cpu",
        device_list=["cpu"],
        low_gpu_mem_usage=False,
    )
    compressor.quantizer = _QuantizerStub()
    compressor.quant_block_list = [["decoder.layers.0"]]

    def _normalize(_inputs):
        compressor.inputs = {"decoder.layers.0": [(torch.ones(1, 1), {})]}

    def _preprocess(_block_inputs, _input_name="hidden_states"):
        assert compressor.model_context.is_mllm is False
        return torch.ones(1, 1), {}

    compressor.normalize_decoding_layer_inputs_ = _normalize
    compressor._preprocess_block_inputs = _preprocess

    monkeypatch.setattr("auto_round.compressors.data_driven.materialize_model_", lambda block: None)
    monkeypatch.setattr(
        "auto_round.compressors.data_driven.convert_module_to_hp_if_necessary",
        lambda block, amp_dtype, device: None,
    )
    monkeypatch.setattr("auto_round.compressors.data_driven.mv_module_from_gpu", lambda block: None)
    monkeypatch.setattr("auto_round.compressors.data_driven.clear_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr("auto_round.compressors.data_driven.is_nv_fp", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("auto_round.compressors.data_driven.is_static_wfp8afp8", lambda *_args, **_kwargs: False)

    block = torch.nn.Linear(1, 1)
    q_outputs, reference_output = compressor.quantize_block(block, inputs=[((torch.ones(1, 1), {}),)])

    assert q_outputs is None
    assert torch.equal(reference_output, torch.ones(1, 1))
    assert compressor.model_context.is_mllm is True
