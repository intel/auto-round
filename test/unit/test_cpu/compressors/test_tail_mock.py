# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the tail-mock primitives (passthrough stubs, tail injector, capture head)."""

import torch
import torch.nn as nn

from auto_round.compressors.tail_mock import (
    CaptureHead,
    PassthroughStub,
    TailInjector,
    install_block_stubs_,
    restore_blocks_,
)


class _FakeLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, hidden_states, *args, **kwargs):
        return self.proj(hidden_states)


def _fake_model(hidden=8, vocab=32, n_layers=3):
    class Body(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_FakeLayer(hidden) for _ in range(n_layers)])
            self.norm = nn.LayerNorm(hidden)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Body()
            self.lm_head = nn.Linear(hidden, vocab)

        def forward(self, input_ids, attention_mask=None, **kwargs):
            emb = input_ids.float().unsqueeze(-1).expand(-1, -1, self.model.layers[0].proj.in_features)
            hidden_states = emb.clone()
            for layer in self.model.layers:
                out = layer(hidden_states)
                hidden_states = out[0] if isinstance(out, tuple) else out
            return self.lm_head(self.model.norm(hidden_states))

    return Model()


class TestPassthroughStub:
    def test_bare_arity_returns_input_unchanged(self):
        stub = PassthroughStub(arity=0)
        x = torch.randn(2, 3, 4)
        out = stub(x)
        assert out is x

    def test_tuple_arity_returns_single_element_tuple(self):
        stub = PassthroughStub(arity=1)
        x = torch.randn(2, 3, 4)
        out = stub(x, attention_mask=None)
        assert isinstance(out, tuple) and len(out) == 1 and out[0] is x

    def test_ignores_extra_positional_and_kwargs(self):
        stub = PassthroughStub(arity=0)
        x = torch.randn(1, 2, 4)
        out = stub(x, "mask-object", position_ids=torch.zeros(1, 2), past_key_value=None)
        assert out is x


class TestTailInjector:
    def test_returns_cached_tail_regardless_of_input(self):
        tail = torch.randn(1, 5, 8)
        inj = TailInjector(tail, arity=0)
        out = inj(torch.randn(1, 5, 8))
        assert torch.equal(out, tail)

    def test_dtype_and_device_preserved(self):
        tail = torch.randn(1, 3, 8, dtype=torch.float64)
        inj = TailInjector(tail, arity=0)
        assert inj(torch.randn(1, 3, 8)).dtype is torch.float64

    def test_set_tail_swaps_payload(self):
        inj = TailInjector(torch.randn(1, 2, 8), arity=0)
        new_tail = torch.randn(1, 2, 8)
        inj.tail = new_tail
        assert torch.equal(inj(torch.zeros(1, 2, 8)), new_tail)

    def test_tuple_arity_matches_stub_convention(self):
        tail = torch.randn(1, 2, 8)
        inj = TailInjector(tail, arity=1)
        out = inj(torch.randn(1, 2, 8))
        assert isinstance(out, tuple) and torch.equal(out[0], tail)


class TestCaptureHead:
    def test_records_full_sequence_input_on_cpu(self):
        head = CaptureHead(out_features=32)
        x = torch.randn(2, 5, 8)
        head(x)
        assert len(head.records) == 1
        rec = head.records[0]
        assert rec.shape == x.shape and rec.device.type == "cpu"
        assert torch.equal(rec, x)

    def test_returns_tiny_dummy_logits(self):
        head = CaptureHead(out_features=32)
        dummy = head(torch.randn(2, 5, 8))
        assert dummy.shape == (2, 1, 32)
        assert torch.count_nonzero(dummy) == 0

    def test_delegates_head_attributes_to_original(self):
        original = nn.Linear(8, 32, bias=False)
        head = CaptureHead(out_features=32, original=original)
        assert head.weight is original.weight  # mamba-style dtype probe path
        assert head.in_features == 8
        head(torch.randn(1, 2, 8))  # records + dummy still work
        assert len(head.records) == 1

    def test_missing_attribute_without_original_raises(self):
        head = CaptureHead(out_features=16)
        try:
            _ = head.weight
        except AttributeError:
            pass
        else:
            raise AssertionError("expected AttributeError")

    def test_multiple_calls_append_records(self):
        head = CaptureHead(out_features=16)
        head(torch.randn(1, 3, 8))
        head(torch.randn(1, 4, 8))
        assert [r.shape[1] for r in head.records] == [3, 4]

    def test_input_tensor_not_mutated_or_retained_grad(self):
        head = CaptureHead(out_features=16)
        x = torch.randn(1, 2, 8, requires_grad=True)
        dummy = head(x)
        assert not dummy.requires_grad
        assert len(head.records) == 1


class TestInstallRestore:
    def test_install_replaces_all_block_slots_with_stubs(self):
        model = _fake_model()
        install_block_stubs_(model, block_names=["model.layers.0", "model.layers.1", "model.layers.2"], arity=1)
        for slot in model.model.layers:
            assert isinstance(slot, PassthroughStub)

    def test_restore_returns_original_modules_bit_identical(self):
        model = _fake_model()
        originals = {i: model.model.layers[i] for i in range(3)}
        restore_info = install_block_stubs_(
            model, block_names=["model.layers.0", "model.layers.1", "model.layers.2"], arity=1
        )
        restore_blocks_(model, restore_info)
        for i in range(3):
            assert model.model.layers[i] is originals[i]

    def test_restore_survives_intermediate_slot_changes(self):
        model = _fake_model()
        restore_info = install_block_stubs_(model, block_names=["model.layers.0"])
        # the capture pass swaps the last slot for an injector afterwards
        model.model.layers[0] = TailInjector(torch.randn(1, 2, 8))
        restore_blocks_(model, restore_info)
        assert isinstance(model.model.layers[0], _FakeLayer)

    def test_partial_install_failure_rolls_back(self):
        model = _fake_model()
        originals = list(model.model.layers)
        try:
            install_block_stubs_(model, block_names=["model.layers.0", "model.does_not_exist"])
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
        for before, after in zip(originals, model.model.layers):
            assert before is after

    def test_missing_block_name_raises(self):
        model = _fake_model()
        try:
            install_block_stubs_(model, block_names=["model.does_not_exist"])
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for an unknown block name")


class TestTailSmokeCheck:
    """The two-token init-time smoke: stubs + real head + the model's own code."""

    @staticmethod
    def _tuple_loop_model(hidden=8, vocab=32, n_layers=3):
        class Body(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([_FakeLayer(hidden) for _ in range(n_layers)])
                self.norm = nn.LayerNorm(hidden)

        class Model(nn.Module):
            def forward(self, input_ids, attention_mask=None, **kwargs):
                hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, self.lm_head.in_features)
                hidden = hidden.clone()
                for layer in self.model.layers:
                    out = layer(hidden)
                    hidden = out[0] if isinstance(out, tuple) else out
                return self.lm_head(self.model.norm(hidden))

        m = Model()
        m.model = Body()
        m.lm_head = nn.Linear(hidden, vocab)
        return m

    @staticmethod
    def _bare_loop_model(hidden=8, vocab=32):
        class Body(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([_FakeLayer(hidden) for _ in range(2)])
                self.norm = nn.LayerNorm(hidden)

        class Model(nn.Module):
            def forward(self, input_ids, attention_mask=None, **kwargs):
                hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, self.lm_head.in_features).clone()
                for layer in self.model.layers:
                    hidden = layer(hidden)  # consumes bare tensors only
                return self.lm_head(self.model.norm(hidden))

        m = Model()
        m.model = Body()
        m.lm_head = nn.Linear(hidden, vocab)
        return m

    def _run(self, model):
        from auto_round.compressors.tail_mock import tail_smoke_check

        blocks = [f"model.layers.{i}" for i in range(len(model.model.layers))]
        return tail_smoke_check(model, "lm_head", blocks)

    def test_tuple_loop_passes_with_tuple_arity(self):
        ok, arity = self._run(self._tuple_loop_model())
        assert ok is True and arity == 1

    def test_bare_loop_passes_with_bare_arity(self):
        ok, arity = self._run(self._bare_loop_model())
        assert ok is True and arity == 0

    def test_layers_restored_after_pass(self):
        model = self._tuple_loop_model()
        originals = list(model.model.layers)
        self._run(model)
        for before, after in zip(originals, model.model.layers):
            assert before is after

    def test_residual_pair_loop_passes_with_arity_two(self):
        """zaya-style layers return (hidden, residual) pairs."""

        class Body(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([_FakeLayer(8) for _ in range(2)])
                self.norm = nn.LayerNorm(8)

        class Model(nn.Module):
            def forward(self, input_ids, attention_mask=None, **kwargs):
                hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, self.hidden_size).clone()
                residual = hidden
                for layer in self.model.layers:
                    hidden, residual = layer(hidden, residual)
                return self.lm_head(self.model.norm(hidden))

        m = Model()
        m.hidden_size = 8
        m.model = Body()
        m.lm_head = nn.Linear(8, 32)
        from auto_round.compressors.tail_mock import tail_smoke_check

        ok, arity = tail_smoke_check(m, "lm_head", ["model.layers.0", "model.layers.1"])
        assert ok is True and arity == 2

    def test_broken_wrapper_fails_and_restores(self):
        model = self._tuple_loop_model()

        def broken_forward(*args, **kwargs):
            raise RuntimeError("text-only call unsupported")

        model.forward = broken_forward
        originals = list(model.model.layers)
        ok, arity = self._run(model)
        assert ok is False and arity is None
        for before, after in zip(originals, model.model.layers):
            assert before is after

    def test_wrong_shape_logits_fail(self):
        model = self._tuple_loop_model()

        def flat_forward(input_ids, attention_mask=None, **kwargs):
            return torch.zeros(1, 5)  # 2-D output regardless of stubs

        model.forward = flat_forward
        ok, arity = self._run(model)
        assert ok is False and arity is None

    def test_missing_head_fails(self):
        from auto_round.compressors.tail_mock import tail_smoke_check

        model = self._tuple_loop_model()
        ok, arity = tail_smoke_check(model, "does_not_exist", ["model.layers.0"])
        assert ok is False and arity is None

    def test_training_mode_preserved(self):
        model = self._tuple_loop_model()
        model.train()
        self._run(model)
        assert model.training is True
