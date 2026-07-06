# Rotation Transform — Details (Experimental)

> ⚠️ **Experimental feature.** Rotation transform is still in an experimental stage. Inference relies on forward hooks, which are currently only supported by the Hugging Face Transformers backend. As a result, inference may be slower compared to native (non-rotated) models.

This document is the in-depth reference for AutoRound's rotation transforms. For a
concise getting-started guide, see the *Rotation* section in
[step_by_step.md](./step_by_step.md).

Rotation redistributes outliers in weights and activations before quantization by
multiplying tensors with orthogonal (Hadamard) matrices. Because an orthogonal
rotation `R` satisfies `R @ Rᵀ = I`, a matched pair of rotations inserted around a
linear operation leaves the network's output unchanged (`x @ R @ Rᵀ @ Wᵀ = x @ Wᵀ`)
while making the intermediate distributions flatter and more quantization-friendly.
This is most useful for aggressive low-bit schemes such as MXFP4, NVFP4 and W4A4.

AutoRound provides two independent rotation implementations:

- **QuaRot / SpinQuant** — architecture-aware, full-model rotation applied at up to
  four positions (R1–R4). Selected via `rotation_config="quarot"`,
  `rotation_config="spinquant"`, or a `SpinQuantConfig` instance. **Recommended.**
- **Per-Linear Block Rotation** — an earlier, simpler implementation that applies a
  block-diagonal Hadamard uniformly to every `nn.Linear`. Selected via
  `rotation_config="default"` / a string Hadamard type, or a `RotationConfig`
  instance. Also exposed through the `--algorithm hadamard` CLI.

---

## 1. QuaRot / SpinQuant

QuaRot applies deterministic Hadamard rotations at up to four positions in the
transformer architecture. Unlike per-linear block rotation, it operates at the model
architecture level — rotating the residual stream, attention heads, and MLP
activations at specific positions for targeted outlier suppression.

The pipeline (implemented in `auto_round/algorithms/transforms/spinquant/preprocessor.py`,
`SpinQuantPreprocessor`) runs the following stages:

1. Fuse RMSNorm scale parameters into the following linear layers (`fuse_rmsnorm`).
2. Untie word embeddings if they are shared with the LM head (`untie_embeddings`).
3. Initialize rotation matrices R1–R4.
4. (Only if trainable) train the rotations and smooth values via KL/MSE loss.
5. Fuse the offline rotations (R2, and R1 when `online_r1_rotation=False`) into weights.
6. Register the online hooks (R1 when online, R3, R4) and clean up.

### 1.1 Rotation positions

| Position | Target | Dimension | Mode | Effect |
|----------|--------|-----------|------|--------|
| **R1** | Residual stream — inputs of `q/k/v_proj`, `gate/up_proj`; outputs of `o_proj`, `down_proj`, `embed_tokens`, `lm_head` | `hidden_size` (or `rotation_size`) | Online *or* Offline | Smooths weight outliers across all linear layers |
| **R2** | `v_proj` output / `o_proj` input (per attention head) | `head_dim` | Offline (fused) | Balances per-head value distributions |
| **R3** | Q/K after RoPE | `head_dim` | Online (RoPE monkeypatch) | Improves KV-cache quantization friendliness |
| **R4** | `down_proj` input (MLP activation) | `intermediate_size` (or `rotation_size`) | Online (hook) + weight fuse | Suppresses activation outliers in the FFN |

- **Online**: applied at runtime via forward hooks / a RoPE monkeypatch. Weights are
  pre-rotated so the paired inverse rotation happens at runtime; there is a small
  inference cost.
- **Offline (fused)**: the rotation is absorbed into adjacent weight matrices, leaving
  no runtime overhead.

R1 is applied online by default (`online_r1_rotation=True`): the target weights are
rotated and a `forward_pre_hook` is registered on the same modules. When
`online_r1_rotation=False`, R1 is instead fully fused into `embed_tokens`,
`q/k/v_proj`, `o_proj`, `gate/up/down_proj` and `lm_head`, cancelling out with no
runtime cost.

### 1.2 `SpinQuantConfig` reference

Defined in `auto_round/algorithms/transforms/spinquant/preprocessor.py`.

| Field | Default | Description |
|-------|---------|-------------|
| `algorithm` | `"spinquant"` | Registry key used by `BaseRotation.from_config()` to dispatch. |
| `r1` | `True` | Enable R1 (hidden_size residual-stream rotation). |
| `r2` | `True` | Enable R2 (head_dim per-head rotation, offline). |
| `r3` | `False` | Enable R3 (online Q/K rotation after RoPE). |
| `r4` | `False` | Enable R4 (online MLP activation rotation). |
| `rotation_size` | `None` (auto) | Overrides the R1/R4 block size (R1 uses it instead of `hidden_size`, R4 instead of `intermediate_size`). R2 always uses `head_dim`; R3 has no custom size. Must be a positive power of 2. |
| `random_r1` / `random_r2` / `random_r3` / `random_r4` | `False` | Use random Hadamard (`H × diag(±1)`) instead of the deterministic matrix at that position. Only relevant in QuaRot mode. |
| `online_r1_rotation` | `True` | Apply R1 online via hook (`True`) or fuse it fully into weights (`False`). |
| `trainable_rotation` | `False` | Learn the rotation matrices via Cayley SGD (SpinQuant mode). **Experimental**, requires a dataloader. |
| `trainable_smooth` | `False` | Learn SmoothQuant-style `smooth_values` jointly via Adam. **Experimental**, requires a dataloader. |
| `iters` | `200` | Training iterations (trainable modes only). |
| `lr` | `1e-4` | SGDG (Cayley) learning rate for rotation matrices. |
| `smooth_lr` | `1e-3` | Adam learning rate for smooth values. |
| `batch_size` | `1` | Training batch size. |
| `loss_type` | `"kl_top"` | Training loss: `"kl_top"`, `"kl_full"`, or `"mse"`. |
| `kl_top_k` | `1000` | Top-k logits used by the `kl_top` loss. |
| `fuse_rmsnorm` | `True` | Fuse RMSNorm scales into following linear layers. |
| `untie_embeddings` | `True` | Untie shared embedding / LM-head weights before rotating. |
| `dtype` | `torch.float32` | Numerical dtype used for rotation math. |
| `device` | `None` (auto) | Defaults to `"cuda"` if available, else `"cpu"`. |

### 1.3 String shortcuts

| Value | Equivalent |
|-------|-----------|
| `"quarot"` | `SpinQuantConfig(trainable_rotation=False, trainable_smooth=False)` — deterministic Hadamard, no training, no calibration data. |
| `"spinquant"` | `SpinQuantConfig(trainable_rotation=True, trainable_smooth=True)` — **experimental**, requires a dataloader. |

Both presets keep the default `r1=True, r2=True, r3=False, r4=False`. A dict with
`algorithm="spinquant"` is also parsed into a `SpinQuantConfig`.

> ⚠️ **SpinQuant trainable rotation** (`trainable_rotation=True`) enables learnable
> rotation matrices optimized via Cayley SGD. This feature is experimental and not
> fully validated on real models. Use `"quarot"` (fixed Hadamard) for production
> workloads.

### 1.4 Usage

```python
from auto_round import AutoRound
from auto_round.algorithms.transforms.spinquant import SpinQuantConfig

model_name = "Qwen/Qwen3-0.6B"

# QuaRot preset: R1+R2 deterministic Hadamard, no training
ar = AutoRound(model_name, scheme="MXFP4", rotation_config="quarot")

# Choose how many positions to rotate:
ar = AutoRound(model_name, scheme="MXFP4", rotation_config=SpinQuantConfig(r1=True))  # R1 only
ar = AutoRound(model_name, scheme="MXFP4", rotation_config=SpinQuantConfig(r1=True, r2=True))  # R1+R2
ar = AutoRound(
    model_name, scheme="MXFP4", rotation_config=SpinQuantConfig(r1=True, r2=True, r3=True, r4=True)
)  # all four

ar.quantize_and_save(output_dir="./Qwen3-0.6B-mxfp4-quarot", format="auto_round")
```

### 1.5 Deterministic vs random Hadamard

Rotation matrices are constructed in
`auto_round/algorithms/transforms/spinquant/rotation_utils.py`:

- **Deterministic** (default): `deterministic_hadamard_matrix(size)` builds a Sylvester
  Hadamard matrix (`H → [[H, H], [H, −H]]`) normalized by `√size`. The size must be a
  power of 2. Because the matrix is fully determined by its size, nothing needs to be
  stored.
- **Random**: `random_hadamard_matrix(size)` computes `H @ diag(±1) / √N`. It offers
  slightly better outlier suppression but the sign matrix must be persisted.

For non-power-of-2 dimensions, `get_hadamard_K(n)` factors `n = K · 2^m` and pulls a
known base Hadamard of size `K` (e.g. 12, 20, 28, 36, 40, 52, 60, 108, 140, 156, 172)
from `known_hadamard.py`, combining it with a Sylvester block.

```python
# Random Hadamard at every enabled position
ar = AutoRound(
    model_name,
    scheme="MXFP4",
    rotation_config=SpinQuantConfig(
        r1=True,
        r2=True,
        r3=True,
        r4=True,
        random_r1=True,
        random_r2=True,
        random_r3=True,
        random_r4=True,
    ),
)
```

### 1.6 Trainable rotation (SpinQuant mode)

When `trainable_rotation=True` (and/or `trainable_smooth=True`), the rotations become
learnable and are optimized on calibration data:

- `cayley_optimizer.py` provides `SGDG`, an orthogonality-preserving optimizer that
  keeps each rotation on the Stiefel manifold using a Cayley/QR retraction.
  `AdamAndSGDG` combines Adam (for `smooth_values`) with SGDG (for the `spinquant_R*`
  matrices).
- `training.py` runs a KL-/MSE-based distillation loop against a frozen clone of the
  original model (`create_dual_optimizer()`, `run_training_loop()`).
- `SpinQuantPreprocessor.preprocess()` **requires a dataloader** whenever either
  trainable flag is set.

### 1.7 Save & load

Quantized models with rotation are saved and loaded transparently. Serialization lives
in `auto_round/algorithms/transforms/spinquant/serialize.py`.

```python
# Save (rotation matrices stored automatically when needed)
ar.quantize_and_save(output_dir="./my_model", format="auto_round")

# Load (rotation hooks restored automatically)
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./my_model", device_map="auto")
```

What is stored per position (metadata keys prefixed with `spinquant_r1` / `spinquant_r4`,
with a `*_type` code: `0` = deterministic, `1` = random, `2` = trained):

- **Deterministic rotations**: only metadata (type + `rotation_size`) — the matrix is
  reconstructed on load.
- **Random rotations**: an `int8` (±1) matrix is stored (size ≈ `rotation_size²` bytes).
- **Trained rotations**: the full `float32` matrix is stored.
- **Online rotations**: rebuilt during model loading — `preregister_spinquant_buffers()`
  creates empty buffers before the `state_dict` load, and `rebuild_spinquant_online()`
  re-patches the QuantLinear forward (R1/R4) and re-applies the R3 RoPE monkeypatch.
  R3 is rebuilt purely from `config.json`, not from stored buffers.

---

## 2. Per-Linear Block Rotation

> ⚠️ This is an earlier experimental implementation that applies block-diagonal
> Hadamard rotation **per linear layer** by patching every `nn.Linear` module in the
> model. For most use cases the QuaRot / SpinQuant approach above is preferred — it
> provides architecture-aware rotation at specific positions (R1–R4) with better
> accuracy and lower overhead.

The per-linear block rotation iterates over all linear layers and applies a Hadamard of
a configurable `block_size` uniformly. It does not distinguish between residual stream,
attention and MLP layers, nor does it handle RoPE or attention-side rotation. It is
configured with `RotationConfig`
(`auto_round/algorithms/transforms/hadamard/config.py`), which is shared by three
backends:

| `backend` | Behaviour |
|-----------|-----------|
| `"inplace"` | QuaRot-style residual-stream rotation; works for any dtype and can fuse the online Hadamard into weights (`fuse_online_to_weight=True`). |
| `"transform"` | Per-Linear weight + activation Hadamard with a fused Triton kernel; supports only MXFP4 / NVFP4 and cannot fuse online to weight. |
| `"auto"` (default) | Picks `inplace` when a fused online rotation is requested, `transform` for MX/NV-FP data types, `inplace` otherwise. |

Application modes (`hadamard/apply.py`):

- **Input mode**: registers a `forward_pre_hook` that rotates the input activation
  before each linear layer (online).
- **Weight mode**: fuses the Hadamard directly into the module weight (offline) and
  patches the calibration wrappers accordingly.

### 2.1 `RotationConfig` reference

| Field | Default | Description |
|-------|---------|-------------|
| `algorithm` | `"hadamard"` (frozen) | Registry key. |
| `backend` | `"auto"` | `"auto"`, `"inplace"`, or `"transform"`. |
| `block_size` | `None` (auto) | Grouped Hadamard block size. Auto-filled from the data type by `normalize_rotation_config()`: **32** for MXFP, **16** for NVFP. `None`/`-1` means full-dimension Hadamard. |
| `hadamard_type` | `"hadamard"` | One of `hadamard`, `random_hadamard`, `inplace_quarot_hadamard`, `inplace_hadamard`, `inplace_random`. Deterministic Hadamard uses Sylvester construction (`block_size` must be a power of 2); `random_hadamard` supports non-power-of-2 sizes from the known-matrix library. |
| `fuse_online_to_weight` | `None` | Fuse online Hadamard rotation into weights when supported (`inplace` backend only). |
| `allow_online_rotation` | `True` | Allow online activation rotation. |

### 2.2 Usage

```python
from auto_round import AutoRound

model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"

# rotation_config="default": block_size auto (32 for MXFP), hadamard_type="hadamard"
ar = AutoRound(model_name_or_path, scheme="MXFP4", rotation_config="default")
ar.quantize_and_save(output_dir="./Llama-3.1-8B-Instruct-mxfp4-ht", format="auto_round")
```

The per-linear rotation is also exposed on the CLI via `--algorithm hadamard`:

```bash
auto-round --model meta-llama/Llama-3.1-8B-Instruct --scheme MXFP4 \
  --algorithm hadamard --rotation_type hadamard --rotation_backend auto
```

CLI flags: `--rotation_type {hadamard,random_hadamard,quarot_hadamard}`,
`--rotation_backend {auto,inplace,transform}`, `--rotation_block_size N`,
`--fuse_online_to_weight` / `--no-fuse_online_to_weight`,
`--allow_online_rotation` / `--no-allow_online_rotation`.

---

## 3. How `rotation_config` is dispatched

`rotation_config` is accepted by `AutoRound` and normalized by the unified
`apply_rotation()` entry point in
`auto_round/algorithms/transforms/__init__.py`:

- A `BaseRotationConfig` instance (`SpinQuantConfig` or `RotationConfig`) is used
  directly.
- The strings `"quarot"` / `"spinquant"` map to the `SpinQuantConfig` shortcuts above.
- A dict with `algorithm="spinquant"` becomes a `SpinQuantConfig`.
- Any other string (e.g. `"default"`, `"hadamard"`, `"random_hadamard"`) or dict maps to
  the per-linear `RotationConfig`.

---

## 4. Example Results

Enabling rotation consistently recovers accuracy lost to aggressive low-bit
quantization. The tables below compare three configurations for each model:

- **BF16** — the unquantized baseline (reference, 100%).
- **MXFP4** — MXFP4 quantization without rotation.
- **MXFP4 + rotation** — MXFP4 quantization with the QuaRot-style rotation shown in
  the *Rotation* column.

All numbers were measured with [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness)
on four tasks — **GSM8K**, **MMLU**, **PIQA** and **HellaSwag**. **Avg** is the mean
across the four tasks and **Ratio** is that average relative to the BF16 baseline.
Higher is better. Across all three model sizes (8B, 32B and 72B), rotation improves the
MXFP4 average and closes the gap to BF16.

**Qwen3-8B** — rotation lifts MXFP4 from 93.77% to 95.73% of BF16 (+1.96 pts), with
the largest gains on the quantization-sensitive MMLU and PIQA tasks.

| Model | Scheme | Rotation | GSM8K (strict, acc) | MMLU (acc) | PIQA (acc) | HellaSwag (acc) | Avg | Ratio |
|---------|---------|---------|---------:|---------:|---------:|---------:|---------:|---------:|
| Qwen/Qwen3-8B | BF16 | - | 0.9136 | 0.7290 | 0.7671 | 0.5708 | 0.7451 | 100.00% |
| Qwen/Qwen3-8B | MXFP4 | - | 0.8560 | 0.6775 | 0.7323 | 0.5289 | 0.6987 | 93.77% |
| Qwen/Qwen3-8B | MXFP4 | R1 (online, 32) + R2 (offline) + R4 (online, 32) | 0.8575 | 0.7009 | 0.7595 | 0.5352 | 0.7133 | 95.73% |

**Qwen3-32B** — rotation improves MXFP4 from 98.36% to 98.53% of BF16, recovering
most of the remaining PIQA and MMLU drop with R1 + R4 alone.

| Model | Scheme | Rotation | GSM8K (strict, acc) | MMLU (acc) | PIQA (acc_norm) | HellaSwag (acc) | Avg | Ratio |
|---------|---------|---------|---------:|---------:|---------:|---------:|---------:|---------:|
| Qwen/Qwen3-32B | BF16 | - | 0.9287 | 0.8076 | 0.8118 | 0.6389 | 0.7968 | 100.00% |
| Qwen/Qwen3-32B | MXFP4 | - | 0.9272 | 0.7876 | 0.7900 | 0.6300 | 0.7837 | 98.36% |
| Qwen/Qwen3-32B | MXFP4 | R1 (online, 32) + R4 (online, 32) | 0.9249 | 0.7908 | 0.8030 | 0.6215 | 0.7851 | 98.53% |

**Qwen2.5-72B** — rotation improves MXFP4 from 97.60% to 98.58% of BF16 (+0.98 pts),
with gains across every task.

| Model | Scheme | Rotation | GSM8K (strict, acc) | MMLU (acc) | PIQA (acc_norm) | HellaSwag (acc) | Avg | Ratio |
|---------|---------|---------|---------:|---------:|---------:|---------:|---------:|---------:|
| Qwen/Qwen2.5-72B | BF16 | - | 0.9454 | 0.8346 | 0.8335 | 0.7035 | 0.8293 | 100.00% |
| Qwen/Qwen2.5-72B | MXFP4 | - | 0.9242 | 0.8104 | 0.8226 | 0.6803 | 0.8094 | 97.60% |
| Qwen/Qwen2.5-72B | MXFP4 | R1 (online, 32) + R4 (online, 32) | 0.9333 | 0.8204 | 0.8292 | 0.6871 | 0.8175 | 98.58% |

> **Note:** Exact numbers depend on the model, task set, lm-eval version and hardware.
> Treat these as illustrative of the *trend* — rotation improving low-bit accuracy —
> rather than as guaranteed values.
