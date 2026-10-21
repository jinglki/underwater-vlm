"""Write versions from the actual server rather than inventing an environment lock."""
import importlib.metadata
import json
import platform
import sys

report = {"python": sys.version, "platform": platform.platform(), "packages": {}}
for name in ("torch", "torchvision", "transformers", "accelerate", "causal-conv1d", "numpy", "opencv-python-headless", "Pillow", "PyYAML"):
    try:
        report["packages"][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        report["packages"][name] = None
try:
    import torch
    report["cuda_build"] = torch.version.cuda
    report["cuda_available"] = torch.cuda.is_available()
    report["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
except ImportError:
    pass
print(json.dumps(report, ensure_ascii=False, indent=2))
