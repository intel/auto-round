# AutoRound 扩散模型量化（实验性）

本功能为实验性功能，目前仅量化扩散模型的 transformer 模块。

使用默认的 `auto_round` 格式导出打包后可重新加载的检查点；`fake` 格式仅用于研究或调试。

```python
autoround.quantize_and_save("./tmp_autoround", format="auto_round", inplace=True)
```

校准默认使用精简的 [OpenS2V 校准集](https://huggingface.co/datasets/changwangss/opens2v-calibration)。
也可通过 `--dataset coco2014` 显式选择 COCO2014，或传入自定义 `.tsv` 文件。
自定义数据需要包含 `id` 和 `caption` 列；图生视频还需要 `image` 列提供参考图像路径。

完整 API、CLI 示例及支持模型列表见[英文说明](README.md)。
