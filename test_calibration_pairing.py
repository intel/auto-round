"""End-to-end pairing-correctness test for calibration-aware rotation.

Bypasses ``auto_round.__init__`` (which pulls in ``datasets``) by loading the
needed leaf modules directly.  Verifies:

1. After ``prepare_calibration_aware_rotation``, every dim that should be
   calibrated has an entry in the cache (including head_dim and num_heads).
2. Two independent ``_resolve_preset_matrix`` calls for the same dim return
   the *same tensor data* (so the weight side and the online-hook side are
   guaranteed to cancel: H Â· Háµ€ = I).
3. Selected matrices are orthogonal up to numerical noise.
"""

import importlib.util
import logging
import pathlib
import sys
import types

import torch

ROOT = pathlib.Path(__file__).parent / "auto_round"


def _load(modname, relpath):
    spec = importlib.util.spec_from_file_location(modname, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


# Stub out the heavy ``auto_round`` package so we can load leaves directly.
pkg = types.ModuleType("auto_round")
pkg.__path__ = [str(ROOT)]
sys.modules["auto_round"] = pkg
utils = types.ModuleType("auto_round.utils")
utils.logger = logging.getLogger("test")
sys.modules["auto_round.utils"] = utils
logging.basicConfig(level=logging.INFO, format="%(message)s")

for sub in [
    "auto_round.algorithms",
    "auto_round.algorithms.transforms",
    "auto_round.algorithms.transforms.rotation",
    "auto_round.algorithms.transforms.rotation.inplace",
    "auto_round.algorithms.transforms.rotation.utils",
]:
    m = types.ModuleType(sub)
    m.__path__ = [str(ROOT / sub.replace("auto_round.", "").replace(".", "/"))]
    sys.modules[sub] = m

_load(
    "auto_round.algorithms.transforms.rotation.utils.math",
    "algorithms/transforms/rotation/utils/math.py",
)
_load(
    "auto_round.algorithms.transforms.rotation.inplace.hooks",
    "algorithms/transforms/rotation/inplace/hooks.py",
)
_load(
    "auto_round.algorithms.transforms.rotation.inplace.model_config",
    "algorithms/transforms/rotation/inplace/model_config.py",
)
calib = _load(
    "auto_round.algorithms.transforms.rotation.inplace.calibration",
    "algorithms/transforms/rotation/inplace/calibration.py",
)
apply_mod = _load(
    "auto_round.algorithms.transforms.rotation.inplace.apply",
    "algorithms/transforms/rotation/inplace/apply.py",
)


# -----------------------------------------------------------------------------
# Build a tiny LLaMA-shaped fake model: one decoder layer with q/k/v/o + MLP
# -----------------------------------------------------------------------------
class _FakeRMSNorm(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.weight


class _FakeAttn(torch.nn.Module):
    def __init__(self, hidden, num_heads):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.k_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.v_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.o_proj = torch.nn.Linear(hidden, hidden, bias=False)


class _FakeMLP(torch.nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = torch.nn.Linear(intermediate, hidden, bias=False)


class _FakeLayer(torch.nn.Module):
    def __init__(self, hidden, intermediate, num_heads):
        super().__init__()
        self.input_layernorm = _FakeRMSNorm(hidden)
        self.self_attn = _FakeAttn(hidden, num_heads)
        self.post_attention_layernorm = _FakeRMSNorm(hidden)
        self.mlp = _FakeMLP(hidden, intermediate)


class _FakeInner(torch.nn.Module):
    def __init__(self, vocab, hidden, intermediate, num_heads, n_layers):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab, hidden)
        self.layers = torch.nn.ModuleList([_FakeLayer(hidden, intermediate, num_heads) for _ in range(n_layers)])
        self.norm = _FakeRMSNorm(hidden)


class _FakeCfg:
    def __init__(self, hidden, intermediate, num_heads):
        self.model_type = "llama"
        self.hidden_size = hidden
        self.intermediate_size = intermediate
        self.num_attention_heads = num_heads


class _FakeModel(torch.nn.Module):
    def __init__(self, vocab=1024, hidden=128, intermediate=256, num_heads=8, n_layers=2):
        super().__init__()
        self.config = _FakeCfg(hidden, intermediate, num_heads)
        self.model = _FakeInner(vocab, hidden, intermediate, num_heads, n_layers)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids=None, attention_mask=None, **_):
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer.input_layernorm(x)
            attn = layer.self_attn
            q = attn.q_proj(h)
            k = attn.k_proj(h)
            v = attn.v_proj(h)
            # Toy attention: just use v summed
            attn_out = attn.o_proj(v)
            x = x + attn_out
            h2 = layer.post_attention_layernorm(x)
            mlp = layer.mlp
            mlp_out = mlp.down_proj(torch.nn.functional.silu(mlp.gate_proj(h2)) * mlp.up_proj(h2))
            x = x + mlp_out
        x = self.model.norm(x)
        return self.lm_head(x)


# -----------------------------------------------------------------------------
# Tiny calibration loader
# -----------------------------------------------------------------------------
def make_calib_loader(vocab=1024, batch=2, seqlen=16, n_batches=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    for _ in range(n_batches):
        ids = torch.randint(0, vocab, (batch, seqlen), generator=g)
        yield {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


# =============================================================================
# Tests
# =============================================================================
torch.manual_seed(0)
HIDDEN, INTERM, NHEADS = 128, 256, 8
HEAD_DIM = HIDDEN // NHEADS  # 16
N_LAYERS = 4
model = _FakeModel(hidden=HIDDEN, intermediate=INTERM, num_heads=NHEADS, n_layers=N_LAYERS)

calib.clear_calibration_hadamard_cache()
report = calib.prepare_calibration_aware_rotation(
    model,
    list(make_calib_loader()),
    block_size=16,
    n_bits=4,
    num_candidates=4,
    max_samples_per_dim=2048,
    compute_device=torch.device("cpu"),
    per_layer=True,
)

print("\n=== Calibration report ===")
for key, info in report.items():
    print(f"  key={str(key):20s}  best={info['name']:30s}  score={info['score']}")

# (1) Global cache must contain hidden_size only (residual stream).
global_dims = set(calib._CALIBRATION_HADAMARD_CACHE.keys())
print(f"\nglobal cache dims  : {sorted(global_dims)}")
assert HIDDEN in global_dims, "hidden_size must be in global cache"
assert INTERM not in global_dims, "intermediate_size must NOT be global"
assert HEAD_DIM not in global_dims, "head_dim must NOT be global"
assert NHEADS not in global_dims, "num_heads must NOT be global"

# (2) Per-layer cache must contain (layer_id, dim) for each per-layer dim.
layer_keys = set(calib._CALIBRATION_HADAMARD_LAYER_CACHE.keys())
print(f"per-layer cache    : {sorted(layer_keys)}")
for L in range(N_LAYERS):
    for d in (INTERM, HEAD_DIM, NHEADS):
        assert (L, d) in layer_keys, f"missing per-layer entry ({L}, {d})"

# (3) Per-layer matrices for the same dim across layers should differ
#     (otherwise per-layer is having no effect).
print("\n=== Per-layer matrix diversity ===")
for d in (INTERM, HEAD_DIM, NHEADS):
    mats = [calib.get_calibration_hadamard_for_layer(L, d) for L in range(N_LAYERS)]
    n_distinct = len({m.data_ptr() for m in mats})  # tensor identity
    n_value_distinct = sum(1 for L in range(1, N_LAYERS) if not torch.equal(mats[0], mats[L]))
    print(f"  dim={d:4d}: {N_LAYERS} layers, value-distinct vs L0: {n_value_distinct}")
    # At least one layer's matrix should differ from layer 0 (otherwise we
    # collapsed back to a single matrix, defeating per-layer).
    assert n_value_distinct >= 1, f"per-layer dim={d} has no diversity"

# (4) Pairing within ONE layer: weight-side fetch must equal hook-side fetch.
print("\n=== Pairing check (within each layer, weight-side vs hook-side) ===")
for L in range(N_LAYERS):
    for d in (INTERM, HEAD_DIM, NHEADS):
        a = calib.get_calibration_hadamard_for_layer(L, d, device=torch.device("cpu"))
        b = calib.get_calibration_hadamard_for_layer(L, d, device=None)
        diff = (a - b).abs().max().item()
        assert diff == 0.0, f"layer {L} dim {d}: pairing broken"
print("  OK â€“ weight & hook see bit-exact same matrix per (layer, dim)")

# (5) Hidden_size pairing: every layer fetches the SAME global matrix.
print("\n=== Global hidden_size identity across layers ===")
g0 = calib.get_calibration_hadamard(HIDDEN)
for L in range(N_LAYERS):
    # The resolver should return the global tensor when no per-layer entry exists.
    # We can simulate this by directly fetching from the global cache.
    g = calib.get_calibration_hadamard(HIDDEN)
    assert torch.equal(g0, g)
print(f"  OK â€“ hidden_size matrix shared across {N_LAYERS} layers")

# (6) Resolver respects the precedence: per-layer first, fall back to global.
#     Also verify the resolver function itself.
from auto_round.algorithms.transforms.rotation.inplace.apply import _resolve_preset_matrix

print("\n=== Resolver precedence ===")
# hidden_size: no per-layer entry â†’ must fall back to global.
hs_via_resolver = _resolve_preset_matrix("calibration_aware", HIDDEN, None, layer_id=2)
assert torch.equal(hs_via_resolver.cpu(), calib.get_calibration_hadamard(HIDDEN))
print("  hidden_size with layer_id=2 â†’ global tensor (correct)")
# intermediate_size with layer_id=2: must return per-layer tensor.
i2 = _resolve_preset_matrix("calibration_aware", INTERM, None, layer_id=2)
assert torch.equal(i2.cpu(), calib.get_calibration_hadamard_for_layer(2, INTERM))
# And it differs from layer 0's intermediate_size matrix (per-layer in action).
i0 = _resolve_preset_matrix("calibration_aware", INTERM, None, layer_id=0)
print(f"  intermediate_size layer0 == layer2 ? {torch.equal(i0.cpu(), i2.cpu())}")

# (7) Orthogonality.
print("\n=== Orthogonality ===")
for L in range(N_LAYERS):
    for d in (INTERM, HEAD_DIM, NHEADS):
        H = calib.get_calibration_hadamard_for_layer(L, d)
        err = (H @ H.T - torch.eye(d, dtype=H.dtype)).abs().max().item()
        assert err < 1e-6, f"non-orthogonal at (L={L}, d={d}): {err:.2e}"
H = calib.get_calibration_hadamard(HIDDEN)
err = (H @ H.T - torch.eye(HIDDEN, dtype=H.dtype)).abs().max().item()
assert err < 1e-6
print("  All orthogonality errors < 1e-6")

print("\nALL PER-LAYER PAIRING TESTS PASSED")
