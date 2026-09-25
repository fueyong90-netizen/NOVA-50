#!/usr/bin/env python3
"""Reports the ACTUAL hardware this was run on (section 16 of the brief).
All fields are read from the running system, never assumed."""
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def cpu_info():
    try:
        with open("/proc/cpuinfo") as f:
            text = f.read()
        model = "UNKNOWN"
        for line in text.splitlines():
            if line.lower().startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
        n_cores = text.count("processor\t:")
        return {"model": model, "logical_cores": n_cores or os.cpu_count()}
    except Exception:
        return {"model": "UNKNOWN", "logical_cores": os.cpu_count()}


def ram_info():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return {"total_gb": round(kb / 1024 / 1024, 2)}
    except Exception:
        pass
    return {"total_gb": "UNKNOWN"}


def main():
    report = {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cpu": cpu_info(),
        "ram": ram_info(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    }
    print(json.dumps(report, indent=2))

    out = ROOT / "results" / "hardware_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
