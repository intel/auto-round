import pytest
from transformers import AutoTokenizer

from auto_round.calib_dataset import get_dataloader


class TestLocalCalibDataset:
    @pytest.mark.timeout(150)
    def test_combine_dataset(self, tiny_opt_model_path, monkeypatch):
        # Subprocess lifecycle is covered by test_calib_dataset_subprocess.py. The
        # streaming sources below cannot reuse its cache, so avoid loading each twice.
        monkeypatch.setenv("AR_DISABLE_DATASET_SUBPROCESS", "1")
        dataset = "NeelNanda/pile-10k:num=1,BAAI/CCI3-HQ:num=1,madao33/new-title-chinese:num=1"
        tokenizer = AutoTokenizer.from_pretrained(tiny_opt_model_path)

        dataloader = get_dataloader(tokenizer, seqlen=128, dataset_name=dataset, bs=3, nsamples=3)
        batch = next(iter(dataloader))

        assert len(dataloader.dataset) == 3
        assert batch is not None
        assert batch["input_ids"].shape == (3, 128)
        assert batch["attention_mask"].shape == (3, 128)
