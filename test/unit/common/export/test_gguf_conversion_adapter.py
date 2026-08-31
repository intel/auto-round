import importlib
import json
import shutil
import textwrap

import pytest

from auto_round.export.export_to_gguf.config import ModelType


def _reset_adapter(adapter):
    adapter._clear_loaded_conversion_modules()
    adapter._ACTIVE_CONVERSION_ROOT = None
    adapter._CONVERSION_MODULE = None
    adapter._CONVERSION_SOURCE = ""


def _write_conversion(root, arch="ToyForCausalLM"):
    conversion = root / "conversion"
    conversion.mkdir(parents=True)
    (conversion / "__init__.py").write_text(
        textwrap.dedent(f"""
            from enum import IntEnum

            TEXT_MODEL_MAP = {{"{arch}": "toy"}}
            MMPROJ_MODEL_MAP = {{}}

            class ModelType(IntEnum):
                TEXT = 1
                MMPROJ = 2

            class ModelBase:
                _model_classes = {{ModelType.TEXT: {{}}, ModelType.MMPROJ: {{}}}}

                @staticmethod
                def load_hparams(dir_model, is_mistral_format=False):
                    import json
                    with open(dir_model / "config.json", encoding="utf-8") as f:
                        return json.load(f)

                @classmethod
                def from_model_architecture(cls, model_architecture, model_type=ModelType.TEXT):
                    try:
                        return cls._model_classes[model_type][model_architecture]
                    except KeyError:
                        raise NotImplementedError(model_architecture)

            def get_model_architecture(hparams, model_type=ModelType.TEXT):
                return hparams["architectures"][0]

            def get_model_class(name, mmproj=False):
                __import__("conversion.toy")
                return ModelBase.from_model_architecture(name, ModelType.MMPROJ if mmproj else ModelType.TEXT)
            """),
        encoding="utf-8",
    )
    (conversion / "toy.py").write_text(
        textwrap.dedent(f"""
            from . import ModelBase, ModelType

            class ToyModel:
                pass

            ModelBase._model_classes[ModelType.TEXT]["{arch}"] = ToyModel
            """),
        encoding="utf-8",
    )


def test_llama_cpp_root_conversion_preferred(tmp_path, monkeypatch):
    from auto_round.export.export_to_gguf import llama_cpp_conversion as adapter

    _reset_adapter(adapter)
    root = tmp_path / "llama.cpp"
    _write_conversion(root)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"architectures": ["ToyForCausalLM"]}), encoding="utf-8")
    monkeypatch.setenv("LLAMA_CPP_ROOT", str(root))

    context = adapter.get_conversion(model_dir, model_type=ModelType.TEXT)

    assert context.is_supported("ToyForCausalLM", ModelType.TEXT)
    assert "LLAMA_CPP_ROOT" in context.source
    _reset_adapter(adapter)


def test_missing_conversion_reports_auto_update_hint(tmp_path, monkeypatch):
    from auto_round.export.export_to_gguf import llama_cpp_conversion as adapter

    _reset_adapter(adapter)
    monkeypatch.delenv("LLAMA_CPP_ROOT", raising=False)
    monkeypatch.delenv("AUTO_ROUND_GGUF_AUTO_UPDATE", raising=False)
    monkeypatch.setattr(adapter, "_bundled_root", lambda: tmp_path / "missing")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"architectures": ["NewForCausalLM"]}), encoding="utf-8")

    with pytest.raises(NotImplementedError, match="AUTO_ROUND_GGUF_AUTO_UPDATE=1"):
        adapter.get_conversion(model_dir, model_type=ModelType.TEXT)
    _reset_adapter(adapter)


def test_auto_update_downloads_minimal_conversion(tmp_path, monkeypatch):
    from auto_round.export.export_to_gguf import llama_cpp_conversion as adapter

    _reset_adapter(adapter)
    monkeypatch.delenv("LLAMA_CPP_ROOT", raising=False)
    monkeypatch.setenv("AUTO_ROUND_GGUF_AUTO_UPDATE", "1")
    monkeypatch.setattr(adapter, "_bundled_root", lambda: tmp_path / "missing")
    monkeypatch.setattr(adapter, "_cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(adapter, "_latest_commit", lambda: "abc123")

    files = {
        "conversion/__init__.py": textwrap.dedent("""
            from enum import IntEnum
            TEXT_MODEL_MAP = {"NewForCausalLM": "new_model"}
            MMPROJ_MODEL_MAP = {}
            class ModelType(IntEnum):
                TEXT = 1
                MMPROJ = 2
            class ModelBase:
                _model_classes = {ModelType.TEXT: {}, ModelType.MMPROJ: {}}
                @classmethod
                def from_model_architecture(cls, arch, model_type=ModelType.TEXT):
                    try:
                        return cls._model_classes[model_type][arch]
                    except KeyError:
                        raise NotImplementedError(arch)
            def get_model_architecture(hparams, model_type=ModelType.TEXT):
                return hparams["architectures"][0]
            def get_model_class(name, mmproj=False):
                __import__("conversion.new_model")
                return ModelBase.from_model_architecture(name, ModelType.MMPROJ if mmproj else ModelType.TEXT)
            """),
        "conversion/base.py": "from . import ModelBase, ModelType\n",
        "conversion/new_model.py": textwrap.dedent("""
            from . import ModelBase, ModelType
            class NewModel:
                pass
            ModelBase._model_classes[ModelType.TEXT]["NewForCausalLM"] = NewModel
            """),
    }

    def fake_fetch(url):
        path = url.split("/abc123/", 1)[1]
        return files[path]

    monkeypatch.setattr(adapter, "_fetch_text", fake_fetch)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"architectures": ["NewForCausalLM"]}), encoding="utf-8")

    context = adapter.get_conversion(model_dir, model_type=ModelType.TEXT)

    assert context.is_supported("NewForCausalLM", ModelType.TEXT)
    assert (tmp_path / "cache" / "abc123" / "conversion" / "new_model.py").is_file()
    shutil.rmtree(tmp_path / "cache", ignore_errors=True)
    _reset_adapter(adapter)


def test_live_model_tensor_names_use_checkpoint_conversion_mapping():
    import gguf

    from auto_round.export.export_to_gguf.convert import _special_name_handle
    from auto_round.utils.common import revert_checkpoint_conversion_mapping

    class WrappedGemma4Model:
        model_arch = gguf.MODEL_ARCH.GEMMA4

    mapping = {r"^model\.language_model\.layers": "model.layers"}
    checkpoint_name = revert_checkpoint_conversion_mapping(
        "model.language_model.layers.0.self_attn.q_proj.weight", mapping
    )

    assert checkpoint_name == "model.layers.0.self_attn.q_proj.weight"
    assert _special_name_handle(WrappedGemma4Model(), checkpoint_name) == "model.layers.0.self_attn.q_proj.weight"


def test_gguf_tensor_names_are_split_between_text_and_mmproj():
    from auto_round.export.export_to_gguf.convert import is_mmproj_tensor_name
    from auto_round.utils.common import MM_MODULE_KEYS

    assert not is_mmproj_tensor_name("model.language_model.layers.0.self_attn.q_proj.weight")
    assert is_mmproj_tensor_name("model.vision_tower.patch_embedder.position_embedding_table")
    assert is_mmproj_tensor_name("model.audio_tower.layers.0.self_attn.q_proj.weight")
    assert is_mmproj_tensor_name("model.waveform_encoder.layers.0.weight")
    assert not any(key in "model.language_model.layers.0.self_attn.q_proj" for key in MM_MODULE_KEYS)
