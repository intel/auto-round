import pytest

from auto_round.compressors.diffusion.dataset import DIFFUSION_DATASET, get_diffusion_dataloader


def test_text2img_dataset_rejects_missing_required_columns(tmp_path):
    dataset_path = tmp_path / "bad.tsv"
    dataset_path.write_text("image_id\ttext\n1\thello\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected columns"):
        DIFFUSION_DATASET["local"](str(dataset_path), nsamples=1)


def test_get_diffusion_dataloader_parses_coco2014_response_without_temp_file(monkeypatch):
    sample_tsv = "id\tcaption\n1\thello\n2\tworld\n"

    class _Response:
        text = sample_tsv

        def raise_for_status(self):
            return None

    def _fake_get(url, timeout):
        assert "captions_source.tsv" in url
        assert timeout == 30
        return _Response()

    monkeypatch.setattr("requests.get", _fake_get)

    dataloader, bs, grad_steps = get_diffusion_dataloader(dataset="coco2014", bs=1, nsamples=2)

    assert bs == 1
    assert grad_steps == 1
    dataset = dataloader.dataset
    assert dataset.caption_ids == [1, 2]
    assert dataset.captions == ["hello", "world"]
