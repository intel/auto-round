import json
import zipfile

import pytest

from auto_round.compressors.diffusion.dataset import DIFFUSION_DATASET, _load_coco_dataframe, get_diffusion_dataloader


def test_text2img_dataset_rejects_missing_required_columns(tmp_path):
    dataset_path = tmp_path / "bad.tsv"
    dataset_path.write_text("image_id\ttext\n1\thello\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected columns"):
        DIFFUSION_DATASET["local"](str(dataset_path), nsamples=1)


def test_text2img_dataset_returns_resolved_image_paths(tmp_path):
    image_path = tmp_path / "images" / "sample.png"
    dataset_path = tmp_path / "calibration.tsv"
    dataset_path.write_text("id\tcaption\timage\n1\thello\timages/sample.png\n", encoding="utf-8")

    dataset = DIFFUSION_DATASET["local"](str(dataset_path), nsamples=1)

    assert dataset[0] == (1, "hello", str(image_path))


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

    dataloader, bs = get_diffusion_dataloader(dataset="coco2014", bs=1, nsamples=2)

    assert bs == 1
    dataset = dataloader.dataset
    assert dataset.caption_ids == [1, 2]
    assert dataset.captions == ["hello", "world"]


def test_coco_i2v_filters_cc_by_before_selecting_samples(monkeypatch, tmp_path):
    manifest = (
        "id\timage_id\tcaption\tfile_name\tcoco_url\n"
        "1\t101\tnon-commercial\t101.jpg\thttp://images.cocodataset.org/val2014/101.jpg\n"
        "2\t102\tallowed one\t102.jpg\thttp://images.cocodataset.org/val2014/102.jpg\n"
        "3\t103\tallowed two\t103.jpg\thttp://images.cocodataset.org/val2014/103.jpg\n"
    )
    metadata = {
        "licenses": [
            {"id": 2, "url": "http://creativecommons.org/licenses/by-nc/2.0/", "name": "CC BY-NC"},
            {"id": 4, "url": "http://creativecommons.org/licenses/by/2.0/", "name": "CC BY"},
        ],
        "images": [
            {"id": 101, "license": 2, "flickr_url": "https://flickr.test/101"},
            {"id": 102, "license": 4, "flickr_url": "https://flickr.test/102"},
            {"id": 103, "license": 4, "flickr_url": "https://flickr.test/103"},
        ],
    }

    def _fake_download(url, destination, timeout):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.name == "captions_source.tsv":
            destination.write_text(manifest, encoding="utf-8")
        elif destination.suffix == ".zip":
            with zipfile.ZipFile(destination, "w") as archive:
                archive.writestr("annotations/captions_val2014.json", json.dumps(metadata))
        else:
            destination.write_bytes(b"image")

    monkeypatch.setattr("auto_round.compressors.diffusion.dataset._get_coco_cache_dir", lambda: tmp_path)
    monkeypatch.setattr("auto_round.compressors.diffusion.dataset._download_to_cache", _fake_download)

    dataframe = _load_coco_dataframe("coco2014", nsamples=1, image_required=True)

    assert dataframe[["image_id", "caption", "license_id"]].to_dict("records") == [
        {"image_id": 102, "caption": "allowed one", "license_id": 4}
    ]
    assert dataframe.iloc[0]["flickr_url"] == "https://flickr.test/102"
    assert dataframe.iloc[0]["license_url"] == "http://creativecommons.org/licenses/by/2.0/"
    assert dataframe.iloc[0]["image"].endswith("/images/102.jpg")
