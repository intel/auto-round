# Wan2.2 T2V A14B：校准量化与 Nunchaku 推理交接

## 模型与硬件范围

目标是 `Wan-AI/Wan2.2-T2V-A14B-Diffusers`，固定 revision 为 `5be7df9619b54f4e2667b2755bc6a756675b5cd7`。`transformer`、`transformer_2` 两个专家各 40 个 Block、400 个量化投影，均需量化。保留源模型的 scheduler、VAE 和切换配置：该 revision 的 `flow_shift=3.0`、`boundary_ratio=0.875`。

**AutoRound 量化导出与 Nunchaku 推理的硬件要求不同。** 此处 AutoRound 的 SVD、QDQ、打包由 PyTorch 完成，不要求安装 Nunchaku，也不执行其 CUDA 内核。量化机器需要足够的 CUDA 显存和主机内存；先用单卡加 CPU model offload。当前没有 A14B 峰值内存测量，不能保证某一显存容量必然足够。

Nunchaku 当前分支的原生 MXFP4 内核只支持 SM120/SM121，例如 RTX 5090，**不支持 B200/SM100 或 H100/SM90**。现有 `mma.sync` block-scale 指令对 SM100 离线编译也失败，B200 需要另外实现 `tcgen05` 内核，单纯修改 `setup.py` 编译目标无效。可以在 B200 上用 AutoRound 准备导出文件，但不能用此分支在该卡上验证原生 Nunchaku 推理。

开发机器没有下载或运行 A14B 权重。本地有真实 5B Smooth + SignRound 导出、修复 RoPE loader 后的 33 帧可辨识视频，以及双专家定向回归。这些结果不等于 A14B 画质或 B200 推理已验证。

## 获取分支与安装量化环境

```bash
git clone --branch wangchang/wan-svdquant-nunchaku https://github.com/changwangss/auto-round.git auto-round-wan
git clone --branch wangchang/wan-mxfp4-runtime https://github.com/changwangss/nunchaku.git nunchaku-wan
```

使用 Python 3.12，并先准备适配目标 GPU、能够正常工作的 CUDA PyTorch 环境。本地检查版本为 PyTorch `2.13.0+cu130`、Diffusers `0.39.0`、Transformers `5.12.1`。不要直接搬运其他机器的 CUDA 扩展。

```bash
python -m pip install --no-build-isolation -e ./auto-round-wan
python -m pip install 'diffusers==0.39.0' 'transformers==5.12.1' accelerate sentencepiece protobuf safetensors
python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))'
```

## 仅在目标机器下载

磁盘需容纳源模型、完整导出目录和序列化临时文件。`low_cpu_mem_usage=False` 会把两个专家的工作权重保留在主机内存；CPU offload 节省显存，不代表主机内存需求也降低。每次使用不同输出目录，保留验证过的产物。

```bash
export MODEL=/data/wan/Wan2.2-T2V-A14B-Diffusers
hf download Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --revision 5be7df9619b54f4e2667b2755bc6a756675b5cd7 --local-dir "$MODEL"
```

已有完整本地 Diffusers 模型可直接使用；不要改用 native 格式权重。脚本保留本地模型自己的 scheduler。

## 对两个专家进行校准量化

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
python -u auto-round-wan/scripts/quantize_wan_a14b_svdquant.py \
  --model "$MODEL" --output /data/wan/a14b-svdquant-smoke \
  --profile smoke --height 256 --width 256 --num-frames 9
```

这条入口调用公开 AutoRound API，先执行启用 Smooth 的 `SVDQuantConfig`，再执行 `SignRoundConfig`；采用 MXFP4 W4A4、group 32、rank 32、BF16 低秩分支。每个专家分别采集输入和调优；校准时临时把所有步路由到当前专家，结束后恢复原始阈值。旧脚本 `quantize_wan_svdquant_nunchaku.py` 是无校准数据的导出路径，不是此处的 Smooth + SignRound 流程。

| Profile | 样本 | 校准步数 | SignRound 迭代 | Residual 迭代 | Smooth grids | Smooth calls |
|---|---:|---:|---:|---:|---:|---:|
| smoke | 1 | 2 | 1 | 1 | 2 | 1 |
| fast | 2 | 2 | 20 | 1 | 6 | 4 |
| quality | 8 | 4 | 200 | 3 | 20 | 16 |

smoke 和 fast 有内置 prompt；smoke 的一次 SignRound 迭代只用于测试流程，不是画质推荐。quality 必须传入有代表性的 `--prompts-file`（UTF-8，每个非空行一个 prompt）或 `--dataset`（AutoRound 数据集名或 caption TSV）。分辨率、帧数独立于 profile，需显式设置。

在 smoke 导出和目标机器运行验证通过后，可尝试更大的校准设置：

```bash
python -u auto-round-wan/scripts/quantize_wan_a14b_svdquant.py \
  --model "$MODEL" --output /data/wan/a14b-svdquant-quality \
  --profile quality --prompts-file /data/wan/calibration-prompts.txt \
  --height 480 --width 832 --num-frames 33
```

此设置尚未测量 A14B 显存需求；必要时减少分辨率、帧数或样本。支持 `--nsamples`、`--calib-steps`、`--iters`、`--rank`、`--residual-iters`、`--smooth-grids`、`--smooth-calls` 单项覆盖。`--device` 是逻辑 CUDA 编号，launcher 使用单卡。旧 shell 入口已移除机器路径依赖：

```bash
MODEL="$MODEL" OUT=/data/wan/a14b-other-smoke PY=python \
  bash auto-round-wan/scripts/launch_wan_svdquant_mxfp4_b200.sh smoke 0
```

## 导出验收

输出应包含两个专家各自的 `config.json` 和 `diffusion_pytorch_model.safetensors`，以及 scheduler、tokenizer、text encoder、VAE、`model_index.json`。两个 Transformer 项均指向 `nunchaku.NunchakuWanTransformer3DModel`，源 pipeline index 不变。

`quantization-run.json` 记录参数、软件/GPU 版本。只有两个专家各有 400 个打包投影且所有浮点权重均有限时，才写入 `export-audit.json`。文件和元数据检查不能替代真实运行或画质判断。

## 原生 Nunchaku 验证：仅 SM120/SM121

在推理机器使用匹配的 CUDA toolkit/PyTorch 编译。SM120 需 CUDA 12.8 或更高，SM121 需 CUDA 13.0 或更高。构建时暴露受支持的 GPU。只做 AutoRound 量化可跳过此步。

```bash
git -C nunchaku-wan submodule update --init --recursive
python -m pip install ninja wheel setuptools imageio imageio-ffmpeg
CUDA_VISIBLE_DEVICES=0 NUNCHAKU_INSTALL_MODE=FAST \
  python -m pip install --no-build-isolation -e ./nunchaku-wan
python -m pip install 'diffusers==0.39.0' 'transformers==5.12.1'
```

先验证真实调度下两个专家均执行、每步 latent 有限：

```bash
CUDA_VISIBLE_DEVICES=0 python -u nunchaku-wan/examples/wan22_t2v_a14b.py \
  --model /data/wan/a14b-svdquant-smoke --output /data/wan/a14b-latent-smoke \
  --latent-only --height 256 --width 256 --num-frames 9 --steps 4
```

再生成短视频：

```bash
CUDA_VISIBLE_DEVICES=0 python -u nunchaku-wan/examples/wan22_t2v_a14b.py \
  --model /data/wan/a14b-svdquant-smoke --output /data/wan/a14b-video-smoke \
  --height 384 --width 640 --num-frames 33 --steps 30 --seed 0
```

示例显式加载两个 Nunchaku 专家，使用 FP32 VAE、tiling、pipeline CPU offload，检查每步 latent 和解码像素，并要求两个专家都实际执行 forward。输出包括记录调用数和逐步检查的 `audit.json`，以及 `latents.pt` 或 `video.mp4`。仍需肉眼检查视频内容与运动，不能只检查数值有限。

## 对旧交接结论的修正

旧交接中的模糊/NaN 观察使用了有缺陷的 loader：`to_empty` 丢失了非持久化 RoPE buffer。这些结果不能直接归因于量化画质。修复后，同一个本地 5B checkpoint 能产生可辨识的运动。旧 A14B checkpoint 仍需在目标机器重新验证。本次同时移除了旧机器硬编码路径，以及“B200 原生推理已支持”的错误前提。

Nunchaku Wan loader 同时保留源模型的 FP32 时间嵌入、归一化参数和 scale-shift table。新增小型 GPU 集成测试，实际串联两个专家的 Smooth + SignRound、导出与原生 Nunchaku 推理。在受支持硬件上安装此 Nunchaku 分支后，可从 AutoRound 仓库运行 `python -m pytest test/integration/test_cuda/test_wan_svdquant_roundtrip.py -q`。测试使用随机小模型，不下载 A14B 权重。
