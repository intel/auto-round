import torch
################ Check available sys.module to decide behavior #################
def is_package_available(package_name: str) -> bool:
    """Check if the package exists in the environment without importing.

    Args:
        package_name (str): package name
    """
    from importlib.util import find_spec

    package_spec = find_spec(package_name)
    return package_spec is not None


def is_hpu_lazy_mode():
    return os.getenv("PT_HPU_LAZY_MODE") != "0"


def _use_hpu_compile_mode():
    from auto_round.utils.common import TORCH_VERSION_AT_LEAST_2_4

    return TORCH_VERSION_AT_LEAST_2_4 and not is_hpu_lazy_mode()


def _bump_dynamo_cache_limit(min_size: Optional[int] = None):
    """Raise torch._dynamo cache/recompile limits.

    The same quant function (e.g. ``quant_tensor_sym``) is reused across
    every linear layer in a transformer block (q/k/v/o_proj, gate/up/
    down_proj, ...), each with a different weight shape. Because dynamo's
    compile cache is keyed by the function's code object (shared across
    all WrapperLinear instances), per-layer static recompiles quickly
    exceed the default ``recompile_limit`` (8) and trigger a fallback to
    eager with a noisy warning. We keep static-shape compilation (best
    perf) and just allow more cache entries.

    The threshold can be overridden via the ``AR_DYNAMO_CACHE_SIZE_LIMIT``
    environment variable (default: 16).
    """
    try:
        if min_size is None:
            from auto_round import envs

            min_size = envs.AR_DYNAMO_CACHE_SIZE_LIMIT
        from torch._dynamo import config as _dynamo_config

        for attr in ("cache_size_limit", "accumulated_cache_size_limit", "recompile_limit"):
            if hasattr(_dynamo_config, attr) and getattr(_dynamo_config, attr) < min_size:
                setattr(_dynamo_config, attr, min_size)
    except Exception:  # pragma: no cover - best effort
        pass


def compile_func(
    fun: Union[torch.nn.Module, Callable], device: Union[str, torch.device, int]
) -> Union[torch.nn.Module, Callable]:
    """Compile a function on the specified device.

    The shared dynamo cache-limit bump lives in :func:`_bump_dynamo_cache_limit`;
    the per-device ``torch.compile`` customization (whether to compile and which
    backend to use) is delegated to the corresponding :class:`Device`, keeping
    this entry point device-agnostic.
    """
    return get_ar_device(device).compile_func(fun)


def is_numba_available():  # pragma: no cover
    """Check if Numba is available."""
    try:
        import numba

        return True
    except ImportError:
        return False


def _is_tbb_installed():  # pragma: no cover
    import importlib.metadata

    try:
        importlib.metadata.version("tbb")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def _is_tbb_configured():  # pragma: no cover
    try:
        from numba.np.ufunc.parallel import _check_tbb_version_compatible

        # check if TBB is present and compatible
        _check_tbb_version_compatible()

        return True
    except ImportError as e:
        logger.warning_once(f"TBB not available: {e}")
        return False


def is_tbb_available():  # pragma: no cover
    """Check if TBB is available."""
    if not _is_tbb_installed():
        logger.warning_once("TBB is not installed, please install it with `pip install tbb`.")
        return False
    if not _is_tbb_configured():
        logger.warning_once(
            (
                "TBB is installed but not configured correctly. \n"
                "Please add the TBB library path to `LD_LIBRARY_PATH`, "
                "for example: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/`."
            )
        )
        return False
    return True


def can_pack_with_numba():  # pragma: no cover
    """Check if Numba and TBB are available for packing.

    To pack tensor with Numba, both Numba and TBB are required, and TBB should be configured correctly.
    """
    if not is_numba_available():
        logger.warning_once("Numba is not installed, please install it with `pip install numba`.")
        return False
    if not is_tbb_available():
        return False
    return True


## check hpex
if is_package_available("habana_frameworks"):
    _hpex_available = True
else:
    _hpex_available = False


@torch._dynamo.disable()
@lru_cache(None)
def is_hpex_available():
    return _hpex_available


_xpu_sdpa_patched = False


# TODO: This is a workaround for the XPU SDPA memory blow-up issue.
# We should remove this patch after the issue is fixed in XPU side.
# https://github.com/intel/auto-round/issues/990
def patch_xpu_sdpa_drop_causal_mask():
    """Workaround for XPU peak-VRAM blow-up in SDPA.

    On Intel XPU, ``torch.nn.functional.scaled_dot_product_attention`` falls back
    to the MATH backend whenever ``attn_mask`` is non-None (FLASH on XPU does
    not support explicit attn_mask, EFFICIENT/CUDNN are not implemented).
    The MATH backend materializes the full ``(B, H, S, S)`` probability matrix
    in both forward and backward, costing several GB at typical
    ``batch_size=8, seqlen=2048``.

    HuggingFace transformers happily passes a *pure causal* 4D mask, even
    though the same effect is achievable via ``is_causal=True`` (which the
    FLASH backend supports and which uses ~12x less memory).

    This monkey-patch detects a pure causal mask, drops it, and forwards
    ``is_causal=True`` instead -- only on XPU and only when no real mask info
    would be lost. CPU/CUDA/HPU paths are left untouched.

    Idempotent. Safe to call multiple times.
    """
    global _xpu_sdpa_patched
    if _xpu_sdpa_patched:
        return
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return
    torch.use_deterministic_algorithms(False)

    _orig_sdpa = torch.nn.functional.scaled_dot_product_attention

    def _is_pure_causal_mask(mask: torch.Tensor, s: int) -> bool:
        # Cheap shape check first
        if mask.shape[-1] != s or mask.shape[-2] != s:
            return False
        if mask.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False
        # Pick the first (B,H) slice; HF mask is broadcast across batch/heads.
        m2d = mask.reshape(-1, s, s)[0]
        tri_up = torch.triu(torch.ones(s, s, dtype=torch.bool, device=mask.device), 1)
        return bool(torch.isinf(m2d[tri_up]).all().item()) and bool((m2d[~tri_up] == 0).all().item())

    def _patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        if (
            query.device.type == "xpu"
            and attn_mask is not None
            and not is_causal
            and query.shape[-2] == key.shape[-2]  # square QK length (no kv-cache)
            and _is_pure_causal_mask(attn_mask, query.shape[-2])
        ):
            attn_mask = None
            is_causal = True
        return _orig_sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            **kwargs,
        )

    torch.nn.functional.scaled_dot_product_attention = _patched_sdpa
    _xpu_sdpa_patched = True
    logger.info("torch.use_deterministic_algorithms(False) is set for XPU.")
    logger.info(
        "Patched torch SDPA on XPU to use is_causal=True for pure causal masks "
        "(avoids ~10x peak-VRAM blow-up from MATH backend)."
    )


def check_is_cpu(device):
    """Check if the device is a CPU.

    Args:
        device: The device to be checked.

    Returns:
        bool: True if the device is a CPU, False otherwise.
    """
    return device == torch.device("cpu") or device == "cpu"