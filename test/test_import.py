# TODO: remove it before merge
import os
import pathlib
def test_ext():
    import auto_round
    AUTO_ROUND_PATH = pathlib.Path(auto_round.__path__[0])
    print(f"auto_round.__path__ = {auto_round.__path__}")
    # list all files in the auto_round's package directory
    files = os.listdir(AUTO_ROUND_PATH.parent)
    for f in files:
        if "auto" in f:
            print(f)
    
    if "auto_round_extension" in files:
        print("!!!!!!!!!!!! auto_round_extension exists")
        files_under_auto_round_extension = os.listdir(AUTO_ROUND_PATH.parent / "auto_round_extension")
        for f in files_under_auto_round_extension:
            print(f"file under ext: {f}")