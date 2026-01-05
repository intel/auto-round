import sys
import subprocess
import re

def get_torch_minor():
    try:
        import torch
        m = re.match(r'^(\d+)\.(\d+)', torch.__version__)
        return f"{m.group(1)}.{m.group(2)}" if m else None
    except ImportError:
        return None

def get_auto_round_minor():
    try:
        import auto_round
        m = re.match(r'^(\d+)\.(\d+)', auto_round.__version__)
        return f"{m.group(1)}.{m.group(2)}" if m else None
    except ImportError:
        return None

# Map torch minor version to kernel version
auto_round_minor = "0.9" if get_auto_round_minor() is None else get_auto_round_minor()
KERNEL_MAP = {
    "2.8": f"auto-round-kernel~={auto_round_minor}.1.0",
    "2.9": f"auto-round-kernel~={auto_round_minor}.2.0",
}

def main():
    torch_minor = get_torch_minor()
    if torch_minor and torch_minor in KERNEL_MAP:
        pkg = KERNEL_MAP[torch_minor]
        print(f"Detected torch {torch_minor}, installing {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--upgrade-strategy", "only-if-needed"])
    else:
        print("torch not found or no mapping for your version. Installing the latest auto-round-kernel ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "auto-round-kernel"])

if __name__ == "__main__":
    main()