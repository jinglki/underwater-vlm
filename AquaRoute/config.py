"""Configuration validation without importing torch or loading model weights."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("gamma", "dcp", "zero_dce", "retinex", "clahe", "white_balance", "waternet", "reserved_identity")


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_config(path: str | Path) -> dict:
    import yaml

    with project_path(path).open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping")
    for section in ("model", "preprocess", "enhancement", "inference", "evaluation"):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"Missing config section: {section}")
    if cfg["model"].get("type") not in {"qwen35", "llava", "custom"}:
        raise ValueError("model.type must be qwen35, llava or custom")
    if not cfg["model"].get("path"):
        raise ValueError("model.path is required")
    if cfg["model"].get("dtype", "bfloat16") not in {"bfloat16", "float16", "float32"}:
        raise ValueError("model.dtype must be bfloat16, float16 or float32")
    for key in ("width", "height"):
        positive_int(cfg["preprocess"].get(key), f"preprocess.{key}")
    positive_int(cfg["model"].get("max_new_tokens", 128), "model.max_new_tokens")
    positive_int(cfg["inference"].get("batch_size", 8), "inference.batch_size")
    positive_int(cfg["inference"].get("image_workers", 4), "inference.image_workers")
    positive_int(cfg["enhancement"].get("workers", 4), "enhancement.workers")
    strengths = cfg["enhancement"].get("strengths", {})
    if (not isinstance(strengths, dict) or set(strengths) != {"Low", "Med", "High"}
            or any(type(v) not in (int, float) or not 0 < v <= 1 for v in strengths.values())):
        raise ValueError("enhancement.strengths must define Low, Med, High in (0, 1]")
    if cfg.get("task", "recognition") not in {"count", "recognition"}:
        raise ValueError("task must be count or recognition")
    if cfg.get("task", "recognition") == "recognition":
        if not isinstance(cfg.get("recognition", {}), dict):
            raise ValueError("recognition must be a YAML mapping")
        mode = cfg.get("recognition", {}).get("mode", "presence")
        if mode not in {"presence", "classification"}:
            raise ValueError("recognition.mode must be presence or classification")
        if mode == "classification":
            labels = cfg["recognition"].get("labels")
            if (not isinstance(labels, list) or not labels
                    or any(not isinstance(s, str) or not s.strip() for s in labels)
                    or len(set(labels)) != len(labels) or "unknown" in labels):
                raise ValueError("classification labels must be unique nonempty strings; 'unknown' is reserved")
    return cfg


def positive_int(value, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value
