from auto_round import envs


def test_work_space_keeps_path_case(monkeypatch):
    monkeypatch.setenv("AR_WORK_SPACE", "/data/MyRun/Work_Space")
    assert envs.AR_WORK_SPACE == "/data/MyRun/Work_Space"


def test_work_space_default(monkeypatch):
    monkeypatch.delenv("AR_WORK_SPACE", raising=False)
    assert envs.AR_WORK_SPACE == "ar_work_space"
