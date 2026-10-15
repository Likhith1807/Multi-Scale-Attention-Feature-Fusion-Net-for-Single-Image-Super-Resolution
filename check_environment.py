"""Report only the software and GPU details needed for this project."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/environment.json")
    args = parser.parse_args()
    report = {
        "python": platform.python_version(),
        "operating_system": platform.system(),
        "gpu_driver_query": None,
        "pytorch": None,
        "cuda_available": False,
        "gpus": [],
    }
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            text=True, capture_output=True, timeout=15, check=False,
        )
        if query.returncode == 0:
            report["gpu_driver_query"] = query.stdout.strip()
        else:
            report["gpu_driver_query"] = "nvidia-smi failed; check the NVIDIA driver."
    except (OSError, subprocess.TimeoutExpired):
        report["gpu_driver_query"] = "nvidia-smi unavailable or timed out."
    try:
        import torch
        report["pytorch"] = str(torch.__version__)
        report["pytorch_cuda_build"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        for index in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(index)
            report["gpus"].append({
                "index": index,
                "name": device.name,
                "memory_gib": round(device.total_memory / (1024 ** 3), 2),
            })
    except ImportError:
        report["action"] = "Install the CUDA build using the official PyTorch selector."
    except Exception as error:
        report["action"] = f"PyTorch GPU initialization failed: {error}"
    report["dependencies"] = {}
    for package in ("torch", "numpy", "Pillow", "scipy"):
        try:
            report["dependencies"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["dependencies"][package] = "not installed"
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nReport saved to {destination}")
    if not report["cuda_available"]:
        print("GPU training is not ready. See README.md for the installation check.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
