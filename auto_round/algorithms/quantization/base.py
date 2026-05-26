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
import importlib
import traceback
from contextlib import contextmanager

import torch

from auto_round.algorithms.quantization.config import QuantizationConfig
from auto_round.compressors.utils import (
    block_forward,
    check_need_act_calibration,
    immediate_pack,
)
from auto_round.data_type import QUANT_FUNC_WITH_DTYPE
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size
from auto_round.logger import logger
from auto_round.utils import (
    INNER_SUPPORTED_LAYER_TYPES,
    SUPPORTED_LAYER_TYPES,
    check_to_quantized,
    clear_memory,
    compile_func,
    convert_module_to_hp_if_necessary,
    get_module,
    set_module,
)
from auto_round.wrapper import WrapperLinear


class BasePipelineMember:
    """Shared interface for all members of a :class:`QuantizationPipeline`.

    Both :class:`BaseWeightTransformer` (pre-processing algorithms that modify
    FP weights in-place) and :class:`BaseQuantizer` (terminal compression
    algorithms) inherit from this class.

    This base class owns the lifecycle hooks and binding logic that are common
    to every pipeline member, so that :class:`QuantizationPipeline` can iterate
    over ``self.all()`` with a single uniform interface.

    Common lifecycle (called by the Compressor / Pipeline in order):
        1. :meth:`bind`               – wire shared compressor state (once at init)
        2. :meth:`prepare_run`        – model-level setup (once before block loop)
        3. :meth:`block_forward_hooks`– register forward hooks (context manager, per block)
        4. :meth:`finalize_run`       – model-level teardown (once after block loop)
    """

    model_context = None
    compress_context = None

    def __init__(self, config=None):
        self.config = config

    @classmethod
    def from_config(cls, config):
        """Instantiate the correct subclass from a config object.

        Fast path: if this class IS the target declared by ``config._alg_cls``,
        construct directly without a registry round-trip.  This handles the common
        case where ``from_config`` is called on the concrete class itself.
        """
        module = importlib.import_module("auto_round.algorithms.quantization")
        alg_cls = getattr(module, config._alg_cls)
        if cls is alg_cls:
            return cls(config)
        return alg_cls(config)

    def bind(self, compressor) -> None:
        """Wire shared context from the owning compressor."""
        self.model_context = compressor.model_context
        self.compress_context = compressor.compress_context

    def prepare_run(self, run_ctx) -> None:
        """Model-level preparation (called once before block iteration starts).

        Override to validate compatibility and resolve model-wide data structures.
        """
        return

    def get_act_calib_policy(self, ctx):
        """Return the activation calibration policy for this block.

        Default: ``ActCalibPolicy(when=SKIP)``.  Subclasses that need activation
        calibration should override and return an appropriate policy.
        """
        from auto_round.algorithms.quantization.pipeline import ActCalibPolicy, CalibTiming, InputSource

        return ActCalibPolicy(when=CalibTiming.SKIP, source=InputSource.FP_CACHE)

    @contextmanager
    def block_forward_hooks(self, ctx):
        """Register algorithm-specific forward hooks for the reference forward.

        Hook handles *must* be removed before this context manager exits.
        Yields a list of hook handles (empty list by default = no hooks registered).

        Both :class:`BaseWeightTransformer` subclasses (e.g. AWQ stats collection)
        and :class:`BaseQuantizer` subclasses (act-calib hooks) implement this
        interface, allowing :class:`QuantizationPipeline` to manage all hooks
        through a single uniform call.
        """
        yield []

    def finalize_run(self, run_ctx) -> None:
        """Model-level teardown (called once after all blocks are processed).

        Must be idempotent – the Compressor calls this inside a ``try/finally``.
        """
        return


class BaseWeightTransformer(BasePipelineMember):
    """Base class for weight-transformation algorithms in a QuantizationPipeline.

    Weight transformers modify FP weights in-place (smooth, scale, rotate) *before*
    the quantizer compresses them.  They do NOT compress weights to low-bit.

    Developers adding a new pre-processing algorithm should inherit from this
    class and override only the lifecycle hooks that are relevant.

    Lifecycle hooks (in call order per block):
        1. :meth:`prepare_run`          – model-level setup (once before all blocks)
        2. :meth:`block_forward_hooks`  – register forward stats hooks (context manager)
        3. :meth:`pre_quantize_block`   – consolidate stats + apply in-place weight transforms
        4. :meth:`post_quantize_block`  – release per-block caches
        5. :meth:`finalize_run`         – model-level teardown (once after all blocks)
    """

    def pre_quantize_block(self, ctx) -> None:
        """Called after the reference forward, before block quantization.

        Override to consolidate stats collected during the reference forward
        and apply in-place weight transforms (e.g. AWQ smoothing).
        """
        return

    def post_quantize_block(self, ctx) -> None:
        """Called after the block quantizer completes.

        Override to release per-block caches (activation stats, etc.).
        """
        return


class RTNLayerFallbackMixin:
    """Default outside-block/layer quantization via RTN.

    Algorithms that want RTN fallback for embeddings, lm_head, or layers outside
    transformer blocks should inherit this mixin explicitly.
    """

    @torch.no_grad()
    def quantize_layer_via_rtn(self, layer_name: str, disable_opt_rtn: bool | None = None) -> None:
        """Quantize one layer with RTN and handle optional immediate pack/save."""
        layer = get_module(self.model, layer_name)
        layer = convert_module_to_hp_if_necessary(layer, self.model_context.amp_dtype, self.compress_context.device)
        set_module(self.model, layer_name, layer)
        tuning_device = layer.tuning_device if hasattr(layer, "tuning_device") else self.compress_context.device
        if (
            self.compress_context.is_immediate_packing
            and self.compress_context.formats[0].is_gguf()
            and not getattr(self.config, "disable_opt_rtn", False)
        ):
            layer = layer.to(tuning_device)
            layer.scale = None
            layer.zp = None
        else:
            try:
                if disable_opt_rtn is None:
                    disable_opt_rtn = bool(getattr(self.config, "disable_opt_rtn", False))
                if (
                    not disable_opt_rtn
                    and getattr(self.config, "orig_disable_opt_rtn", None) is None
                    and self.model_context.is_moe_model
                    and "expert" in layer.global_name
                    and "shared_expert" not in layer.global_name
                    and self.config.super_bits is None
                ):
                    disable_opt_rtn = True
                    logger.warning_once(
                        "MoE layer detected: optimized RTN is disabled for efficiency. "
                        "Use `--enable_opt_rtn` to force-enable it for MoE layers."
                    )
                layer = layer.to(tuning_device)
                layer = WrapperLinear(
                    layer,
                    device=tuning_device,
                    enable_minmax_tuning=False,
                    enable_norm_bias_tuning=False,
                    enable_round_tuning=False,
                    enable_torch_compile=self.compress_context.enable_torch_compile,
                    disable_opt_rtn=disable_opt_rtn,
                    iters=0,
                )
                layer = layer.unwrapper({})
            except torch.OutOfMemoryError:
                cuda_error_msg = traceback.format_exc()
                layer = layer.orig_layer if hasattr(layer, "orig_layer") else layer
                try:
                    logger.error(cuda_error_msg)
                    logger.warning("falling back to CPU.")
                    layer.to("cpu")
                    layer = WrapperLinear(
                        layer,
                        enable_minmax_tuning=False,
                        enable_norm_bias_tuning=False,
                        enable_round_tuning=False,
                        enable_torch_compile=self.compress_context.enable_torch_compile,
                        iters=0,
                    )
                    layer = layer.unwrapper({})
                except Exception:
                    raise

        set_module(self.model, layer_name, layer)
        self._immediate_pack_and_save_module(layer_name)

    def _immediate_pack_and_save_module(self, module_name):
        from auto_round.compressors.shard_writer import ShardWriter

        shard_writer = ShardWriter.get_shard_writer()
        to_cpu = self.compress_context.low_gpu_mem_usage
        module = get_module(self.model, module_name)
        if self.compress_context.is_immediate_packing:
            immediate_pack(module_name, self.layer_config)
            if to_cpu:
                module = module.to("cpu")
                packed_module = get_module(self.model, module_name)
                set_module(self.model, module_name, packed_module.to("cpu"))
        else:
            if to_cpu:
                module = module.to("cpu")
            set_module(self.model, module_name, module)
        if self.compress_context.is_immediate_saving:
            module = get_module(self.model, module_name)
            module.to("cpu")
            shard_writer.write(module, module_name, False)
            module.to("meta")

    def quantize_layer_outside_block(self, layer_name: str, input_ids=None, **kwargs):
        dtype = kwargs.pop("dtype", None)
        if dtype is not None:
            layer = get_module(self.model, layer_name)
            set_module(self.model, layer_name, layer.to(dtype))
        self.quantize_layer_via_rtn(layer_name, **kwargs)


class DiffusionMixin:
    """Mixin that adds diffusion-model support to a :class:`BaseQuantizer` subclass.

    Attach to any :class:`BaseQuantizer` subclass that needs to quantize
    diffusion models (e.g. Flux, StableAudio, Wan):

        class SignRoundQuantizer(DiffusionMixin, BaseQuantizer): ...

    The mixin overrides :meth:`create_block_io` so diffusion-specific input
    slicing, output mapping, and hidden_states extraction live in
    :class:`DiffusionBlockIO` rather than individual quantizers.

    ``DIFFUSION_OUTPUT_CONFIGS`` maps block class name → ordered list of output
    tensor keys.  Extend in subclasses to register new diffusion architectures
    without overriding any method.
    """

    # Map block class name → list of output tensor keys returned by that block.
    # The order of keys must match the order of tensors in the block's return tuple.
    # Subclasses can extend this dict to register new architectures without
    # overriding any method.
    DIFFUSION_OUTPUT_CONFIGS: dict = {
        "FluxTransformerBlock": ["encoder_hidden_states", "hidden_states"],
        "FluxSingleTransformerBlock": ["encoder_hidden_states", "hidden_states"],
        "OvisImageTransformerBlock": ["encoder_hidden_states", "hidden_states"],
        "OvisImageSingleTransformerBlock": ["encoder_hidden_states", "hidden_states"],
        "StableAudioDiTBlock": ["hidden_states"],
        "WanTransformerBlock": ["hidden_states"],
    }

    def _get_output_config(self, block: torch.nn.Module) -> list:
        """Return the output key list for *block* from ``DIFFUSION_OUTPUT_CONFIGS``."""
        return self.DIFFUSION_OUTPUT_CONFIGS.get(block.__class__.__name__, ["hidden_states"])

    def create_block_io(self, input_ids, input_others, quantized_input=None, block=None):
        from auto_round.algorithms.quantization.pipeline import DiffusionBlockIO, InputSource

        active_source = (
            InputSource.QUANTIZED_INPUT
            if quantized_input is not None and self.enable_quanted_input
            else InputSource.FP_CACHE
        )
        io = DiffusionBlockIO(
            fp_inputs=input_ids,
            input_others=input_others,
            quantized_inputs=quantized_input,
            active_source=active_source,
            batch_dim=self.batch_dim,
            seqlen=self.seqlen,
            shared_cache_keys=self.model_context.shared_cache_keys,
            output_config=self._get_output_config(block),
        )
        io.fp_inputs, io.input_others = io.preprocess_block_inputs(io.fp_inputs, io.input_others, block)
        return io


class BaseQuantizer(BasePipelineMember):
    """Base class for terminal weight-compression algorithms in a QuantizationPipeline.

    Developers adding a new quantization algorithm should inherit from this
    class and override at minimum :meth:`quantize_block`.

    For diffusion model support, also inherit :class:`DiffusionMixin`:
        ``class MyQuantizer(DiffusionMixin, BaseQuantizer): ...``

    Lifecycle hooks to override as needed:
        - :meth:`prepare_run`                – model-level setup (once before all blocks)
        - :meth:`get_act_calib_policy`       – activation calibration policy
        - :meth:`block_forward_hooks`        – register act-calib hooks (context manager)
        - :meth:`quantize_block`             – **must override**: quantize a single block
        - :meth:`quantize_layer_outside_block` – quantize layers outside blocks
        - :meth:`finalize_run`               – model-level teardown (once after all blocks)
    """

    # Class-level attribute declarations for convenient access in quantization methods.
    # Scheme-related attrs (layer_config, scale_dtype, has_qlayer_outside_block, etc.)
    # are resolved by SchemeMixin in BaseCompressor and synced here after post_init().
    dataset = None
    supported_types = SUPPORTED_LAYER_TYPES
    inner_supported_types = INNER_SUPPORTED_LAYER_TYPES
    enable_alg_ext = False

    def __init__(self, config: QuantizationConfig):
        super().__init__(config)
        self.layer_config = None
        self.bits = config.bits
        self.group_size = config.group_size
        self.sym = config.sym
        self.data_type = config.data_type
        self.act_bits = config.act_bits
        self.act_group_size = config.act_group_size
        self.act_sym = config.act_sym
        self.act_data_type = config.act_data_type
        self.act_dynamic = config.act_dynamic
        self.super_bits = config.super_bits
        self.super_group_size = config.super_group_size
        self.scale_dtype = config.scale_dtype
        self.ignore_layers = config.ignore_layers
        self.quant_lm_head = config.quant_lm_head
        self.to_quant_block_names = config.to_quant_block_names
        # Calibration-time state lives on a shared
        # :class:`~auto_round.calibration.state.CalibrationState` instance.
        # The compressor wires its own instance here in ``_resolve_scheme``;
        # until then we own a private placeholder so property-based reads /
        # writes during construction don't blow up.
        from auto_round.calibration.state import CalibrationState

        self._calibration_state = CalibrationState()
        self.infer_bs_coeff = getattr(config, "infer_bs_coeff", 1)
        # Whether to feed quantized-block outputs as inputs to the next block.
        # Subclasses that support cascaded quantized-input (e.g. SignRoundQuantizer)
        # override this from their config.  Defaults to False for zero-shot algorithms
        # (RTN) where activations are not used during weight optimization.
        self.enable_quanted_input = getattr(config, "enable_quanted_input", False)

    # ── Shared CalibrationState forwarders ───────────────────────────────────────
    @property
    def calibration_state(self):
        return self._calibration_state

    @calibration_state.setter
    def calibration_state(self, new_state) -> None:
        # Compressor-supplied shared instance; just rebind.
        self._calibration_state = new_state

    @property
    def attention_mask(self) -> list:
        return self._calibration_state.attention_mask

    @attention_mask.setter
    def attention_mask(self, value) -> None:
        self._calibration_state.attention_mask = value if value is not None else []

    @property
    def batch_dim(self):
        return self._calibration_state.batch_dim

    @batch_dim.setter
    def batch_dim(self, value) -> None:
        self._calibration_state.batch_dim = value

    @property
    def batch_size(self) -> int:
        return self._calibration_state.batch_size

    @batch_size.setter
    def batch_size(self, value) -> None:
        self._calibration_state.batch_size = value

    @property
    def nsamples(self) -> int:
        return self._calibration_state.nsamples

    @property
    def seqlen(self) -> int:
        return self._calibration_state.seqlen

    def bind(self, compressor) -> None:
        """Wire shared state from the owning compressor.

        The compressor owns the authoritative ``model_context`` /
        ``compress_context`` / ``CalibrationState`` and resolves
        ``scale_dtype`` (string → torch dtype).  All quantizer fields that
        merely mirror the compressor are pulled from here in one place.
        """
        self.model_context = compressor.model_context
        self.compress_context = compressor.compress_context
        self.scale_dtype = compressor.scale_dtype
        # Share the compressor's CalibrationState instance.
        self._calibration_state = compressor._calibration_state

    @property
    def model(self):
        return self.model_context.model if self.model_context is not None else None

    @property
    def formats(self):
        return getattr(self.compress_context, "formats", None)

    @property
    def amp(self):
        return getattr(self.model_context, "amp", False)

    @property
    def amp_dtype(self):
        return getattr(self.model_context, "amp_dtype", torch.float32)

    # ── Activation-calibration hook infrastructure ───────────────────────────────

    def _register_act_max_hooks(self, model):
        """Register per-module act_max tracking hooks for static activation quantization.

        Internal implementation called by :meth:`block_forward_hooks`.
        Returns a list of hook handles that the caller must remove when done.
        """

        def collect_act_max(module, input, output):
            input = input[0] if isinstance(input, (tuple, list)) else input
            if input.numel() == 0:
                return
            input, _, _ = reshape_pad_tensor_by_group_size(input, self.act_group_size)
            act_max = torch.max(torch.abs(input), dim=-1).values
            if not hasattr(module, "act_max") or module.act_max.numel() == 0:
                module.act_max = act_max
                if self.config.is_act_nv_fp:
                    max_val = act_max.max()
                    module.act_max = max_val.unsqueeze(0) if max_val.dim() == 0 else max_val
                return

            act_max = act_max.to(module.act_max.device)
            if self.config.is_act_nv_fp:
                max_val = torch.max(act_max.max(), module.act_max.max())
                module.act_max = max_val.unsqueeze(0) if max_val.dim() == 0 else max_val
            else:
                module.act_max = torch.max(act_max, module.act_max)

        def should_collect(name, module):
            if isinstance(module, SUPPORTED_LAYER_TYPES):
                return (
                    hasattr(module, "act_dynamic")
                    and check_need_act_calibration(module.act_dynamic, module.act_data_type, module.act_bits)
                    and check_to_quantized(module)
                )
            if name in self.layer_config:
                config = self.layer_config[name]
                act_dynamic = config.get("act_dynamic", True)
                act_data_type = config.get("act_data_type", None)
                act_bits = config.get("act_bits", 16)
                return (
                    config["bits"] <= 8
                    and check_need_act_calibration(act_dynamic, act_data_type, act_bits)
                    and check_to_quantized(config)
                )
            return False

        handles = []
        if should_collect("", model):
            handles.append(model.register_forward_hook(collect_act_max))
            return handles
        for name, module in model.named_modules():
            if name and should_collect(name, module):
                handles.append(module.register_forward_hook(collect_act_max))
        return handles

    @contextmanager
    def block_forward_hooks(self, ctx):
        """Register act-calib forward hooks for the reference forward.

        Implements the :meth:`BasePipelineMember.block_forward_hooks` interface for
        terminal quantizers.  Yields the list of hook handles so the caller can
        determine whether any act-calib hooks were registered (used to decide
        whether a second forward with quantized inputs is needed).
        """
        from auto_round.algorithms.quantization.pipeline import CalibTiming

        policy = self.get_act_calib_policy(ctx)
        if policy.when == CalibTiming.SKIP:
            yield []
            return
        handles = self._register_act_max_hooks(ctx.block)
        try:
            yield handles
        finally:
            for h in handles:
                h.remove()

    def get_act_calib_policy(self, ctx):
        """Return the activation calibration policy for this block.

        Default: ``WITH_REFERENCE + FP_CACHE``, or ``QUANTIZED_INPUT`` when
        ``enable_quanted_input=True`` and a quantized previous-block output is available.
        """
        from auto_round.algorithms.quantization.pipeline import ActCalibPolicy, CalibTiming, InputSource

        quantized_input = getattr(ctx, "quantized_input", None)
        if quantized_input is not None and self.enable_quanted_input:
            return ActCalibPolicy(when=CalibTiming.WITH_REFERENCE, source=InputSource.QUANTIZED_INPUT)
        return ActCalibPolicy(when=CalibTiming.WITH_REFERENCE, source=InputSource.FP_CACHE)

    def create_block_io(self, input_ids, input_others, quantized_input=None, block=None):
        from auto_round.algorithms.quantization.pipeline import BlockIO, InputSource

        active_source = (
            InputSource.QUANTIZED_INPUT
            if quantized_input is not None and self.enable_quanted_input
            else InputSource.FP_CACHE
        )
        return BlockIO(
            fp_inputs=input_ids,
            input_others=input_others,
            quantized_inputs=quantized_input,
            active_source=active_source,
            batch_dim=self.batch_dim,
            seqlen=self.seqlen,
            shared_cache_keys=self.model_context.shared_cache_keys,
        )

    # ── Embedding quantization ────────────────────────────────────────────────────

    @torch.inference_mode()
    def _quantize_embedding_layer(self):
        """Quantizes embedding layers in the model according to the configuration.

        This method iterates through all modules in the model, identifies embedding
        layers specified in `self.layer_config`, and applies the appropriate quantization
        function based on bit precision, grouping strategy, and dtype.

        Returns:
            bool: True if any embedding layer was quantized.
        """
        is_quantized = False
        for name, module in self.model_context.model.named_modules():
            # Skip non-Embedding modules or layers not in config
            if not isinstance(module, torch.nn.Embedding) or name not in self.layer_config:
                continue

            config = self.layer_config[name]

            # Skip layers that are not marked for quantization
            if not check_to_quantized(config):
                continue
            is_quantized = True
            config["scale_dtype"] = self.scale_dtype
            dtype = config["data_type"]

            # Determine quantization function key with symmetry/asymmetry
            if dtype not in QUANT_FUNC_WITH_DTYPE:
                dtype = f"{dtype}_{'sym' if config['sym'] else 'asym'}"

            quant_func = QUANT_FUNC_WITH_DTYPE[dtype]
            dtype = module.weight.dtype
            # As typically float32 are used in RTN to search scale zp,
            # to avoid cache a bf16 copy we'd better use float32
            if config.get("super_group_size", None) is not None:
                dtype = torch.float32

            # Attempt quantization on GPU, fall back to CPU if OOM
            try:
                weight, scale, zp = quant_func(
                    module.weight.to(dtype=dtype, device=self.compress_context.device),
                    **{
                        k: config.get(k, None)
                        for k in ["bits", "group_size", "super_bits", "super_group_size", "scale_dtype"]
                    },
                )
            except torch.OutOfMemoryError:
                cuda_error_msg = traceback.format_exc()
                try:
                    logger.error(cuda_error_msg)
                    logger.warning("falling back to CPU")
                    weight, scale, zp = quant_func(
                        module.weight.to("cpu"),
                        **{
                            k: config.get(k, None)
                            for k in ["bits", "group_size", "super_bits", "super_group_size", "scale_dtype"]
                        },
                    )
                except Exception as e:
                    raise

            # Overwrite the module's weights with the quantized version
            module.weight.data.copy_(weight.cpu())

            # Attach scale and zero point (zp) to the module
            for param_name, value in zip(["scale", "zp"], [scale, zp]):
                if isinstance(value, dict):
                    for k, v in value.items():
                        setattr(module, k if k == "scale" else f"w_{k}", v.cpu())
                elif isinstance(value, torch.Tensor):
                    setattr(module, param_name, value.cpu())
                else:
                    setattr(module, param_name, value)

            # Update config
            self.layer_config.setdefault(name, {}).update(config)
            del weight
            del scale
            del zp
            clear_memory(device_list=self.compress_context.device_list)

        return is_quantized

    # ── Abstract quantization interface ──────────────────────────────────────────

    def quantize_block(self, ctx) -> dict:
        """Apply the quantization algorithm to a prepared block.

        This is the **pure-algorithm** entry point called by the Compressor after
        all infrastructure work (device placement, data collection, act-max hook
        registration, DDP setup) has been completed.

        Implementations should:
        - Perform the algorithm-specific weight/activation quantization on ``ctx.block``.
        - Return a dict of best parameters (may be empty for zero-shot algorithms).

        Args:
            ctx: Per-block pipeline context. ``ctx.io`` owns calibration inputs,
                quantized inputs, and reference outputs.

        Returns:
            dict: Best quantization parameters found, or ``{}`` if not applicable.
        """
        raise NotImplementedError("quantize_block must be implemented in subclasses of BaseQuantizer")

    def quantize_layer(self, layer_name: str, **kwargs):
        """Quantizes a single layer of the model.

        Args:
            layer_name (str): The name of the layer to quantize. The layer module is
                retrieved internally via get_module(model, layer_name).
        """
        raise NotImplementedError("quantize_layer must be implemented in subclasses of BaseQuantizer")

    def quantize_layer_outside_block(self, layer_name: str, input_ids=None, **kwargs):
        """Quantizes a single layer of the model outside of a block.

        Args:
            layer_name (str): The name of the layer to quantize. The layer module is
                retrieved internally via get_module(model, layer_name).
            input_ids: Optional calibration inputs for data-driven outside-layer quantization.
        """
        raise NotImplementedError("quantize_layer_outside_block must be implemented in subclasses or mixins")

    def _resolve_block_forward(self):
        """Resolve and cache the block forward function once.

        This avoids repeated attribute checks in the hot training loop
        (called thousands of times per block).

        Mirrors old-arch behaviour: act-quant hooks, alg-ext, and optimized RTN
        use the plain ``block_forward`` instead of ``torch.compile``.
        """
        cached = self.__dict__.get("_resolved_block_forward")
        if cached is not None:
            return cached
        if (
            (self.config.is_act_quantize and (not self.config.act_dynamic or self.config.is_act_nv_fp))
            or self.enable_alg_ext
            or not getattr(self.config, "disable_opt_rtn", True)
        ):
            self._resolved_block_forward = block_forward
        elif self.compress_context.enable_torch_compile:
            compiled = self.__dict__.get("_compiled_block_forward")
            if compiled is None:
                compiled = compile_func(block_forward, self.compress_context.device)
                self._compiled_block_forward = compiled
            self._resolved_block_forward = compiled
        else:
            self._resolved_block_forward = block_forward
        return self._resolved_block_forward

    def _invalidate_block_forward_cache(self):
        """Clear the cached block forward function (call when block changes)."""
        self.__dict__.pop("_resolved_block_forward", None)
        self.__dict__.pop("_compiled_block_forward", None)

    def prepare_run(self, run_ctx) -> None:
        """Model-level preparation (called once before block iteration starts)."""
        return

    def finalize_run(self, run_ctx) -> None:
        """Model-level teardown (called once after all blocks are processed).

        Must be idempotent – the Compressor calls this inside a ``try/finally``.
        """
        return
