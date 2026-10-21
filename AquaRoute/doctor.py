"""Fast preflight checks; never downloads a model or allocates its weights."""
import importlib.util
from pathlib import Path

from .config import project_path
from .storage import versions


def check(cfg, *, require_model=True, require_enhancement=False):
    checks = []
    for name in ("yaml", "PIL", "numpy", "cv2", "tqdm"):
        checks.append((name, importlib.util.find_spec(name) is not None, "Required runtime package"))
    if require_model:
        for name in ("torch", "transformers", "accelerate"):
            checks.append((name, importlib.util.find_spec(name) is not None, "Required inference package"))
        model_dir = project_path(cfg["model"]["path"])
        checks.append(("model directory", model_dir.is_dir(), str(model_dir)))
        checks.append(("model config", (model_dir / "config.json").is_file(), "config.json"))
        checks.append(("model weights", any(model_dir.glob("*.safetensors")) or any(model_dir.glob("pytorch_model*.bin")),
                       "Download a complete model snapshot, including processor/tokenizer files"))
        if importlib.util.find_spec("torch") is not None:
            import torch
            if str(cfg["model"].get("device", "cuda")).startswith("cuda"):
                checks.append(("CUDA", torch.cuda.is_available(), str(torch.version.cuda)))
        if importlib.util.find_spec("transformers") is not None and cfg["model"]["type"] == "qwen35":
            try:
                from transformers import Qwen3_5ForConditionalGeneration
                available = Qwen3_5ForConditionalGeneration is not None
                detail = "Qwen3_5ForConditionalGeneration"
            except ImportError as exc:
                available, detail = False, str(exc)
            checks.append(("Qwen3.5 model class", available, detail))
    if require_enhancement:
        weights = project_path(cfg["enhancement"]["waternet_weights"])
        checks.append(("WaterNet weights", weights.is_file(), str(weights)))
        checks.append(("WaterNet code", (project_path("enhance/waternet/waternet/net.py")).is_file(), "Bundled vendor source"))
    return {"ok": all(ok for _, ok, _ in checks), "checks": checks, "versions": versions()}
