"""JSONL loading, manifest checks, and ordered condition expansion."""
import json
import uuid
from pathlib import Path

from .config import METHODS
from .protocol import validate_truth


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("Each line must be a JSON object")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            records.append(value)
    return records


def read_samples(path: str | Path, cfg: dict) -> list[dict]:
    path = Path(path).resolve()
    samples = load_jsonl(path)
    if not samples:
        raise ValueError("Manifest is empty")
    seen = set()
    mode = cfg.get("recognition", {}).get("mode", "presence")
    labels = cfg.get("recognition", {}).get("labels", [])
    for number, sample in enumerate(samples, 1):
        sample_id = sample.get("id") or str(uuid.uuid4())
        if not isinstance(sample_id, str) or not sample_id.strip() or sample_id in seen:
            raise ValueError(f"Sample {number}: duplicate or invalid id")
        seen.add(sample_id)
        sample["id"] = sample_id
        for field in ("target", "question"):
            if not isinstance(sample.get(field), str) or not sample[field].strip():
                raise ValueError(f"Sample {number}: {field} is required")
        paths = sample.get("images")
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"Sample {number}: images must be a nonempty list")
        resolved = []
        for value in paths:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Sample {number}: image paths must be nonempty strings")
            image = Path(value).expanduser()
            image = image if image.is_absolute() else path.parent / image
            if not image.is_file():
                raise FileNotFoundError(image)
            resolved.append(str(image.resolve()))
        sample["images"] = resolved
        validate_truth(sample.get("truth"), len(paths), mode, labels)
    return samples


def expand_conditions(prepared: list[dict], cfg: dict, path_maps=(), base_dir: Path | None = None) -> list[dict]:
    """Keep each sample/condition independent, with its original image order."""
    tasks, seen = [], set()
    mode = cfg.get("recognition", {}).get("mode", "presence")
    labels = cfg.get("recognition", {}).get("labels", [])
    for sample in prepared:
        sample_id = sample.get("id")
        if not isinstance(sample_id, str) or not sample_id.strip() or sample_id in seen:
            raise ValueError("Prepared data contains duplicate or invalid sample IDs")
        seen.add(sample_id)
        for field in ("target", "question"):
            if not isinstance(sample.get(field), str) or not sample[field].strip():
                raise ValueError(f"{sample_id}: {field} is required")
        conditions = sample.get("conditions")
        if not isinstance(conditions, dict) or not conditions:
            raise ValueError(f"{sample_id}: conditions must be a nonempty mapping")
        image_count = None
        for condition, values in conditions.items():
            if condition == "original" and not cfg["inference"].get("include_original", True):
                continue
            if not isinstance(condition, str):
                raise ValueError("Condition names must be strings")
            if condition != "original":
                method, separator, level = condition.rpartition("_")
                if not separator or method not in METHODS or level not in cfg["enhancement"]["strengths"]:
                    raise ValueError(f"Invalid condition: {condition}")
            if not isinstance(values, list) or not values:
                raise ValueError(f"{sample_id}/{condition}: images must be nonempty")
            if image_count is not None and image_count != len(values):
                raise ValueError(f"{sample_id}: different image counts across conditions")
            image_count = len(values)
            paths = []
            for value in values:
                if not isinstance(value, str) or not value:
                    raise ValueError("Image paths must be nonempty strings")
                # Explicit prefix remapping allows reuse of prepared images moved
                # from a different server. No implicit suffix matching is used.
                for old, new in path_maps:
                    normalized = value.replace("\\", "/")
                    prefix = old.replace("\\", "/").rstrip("/")
                    if normalized == prefix or normalized.startswith(prefix + "/"):
                        value = new.rstrip("/\\") + normalized[len(prefix):]
                        break
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = (base_dir or Path.cwd()) / path
                if not path.stem.endswith("_" + condition):
                    raise ValueError(f"Condition {condition} contains mismatched images: {path}")
                if not path.is_file():
                    raise FileNotFoundError(f"Prepared image missing: {path}; use --path-map OLD=NEW after moving data")
                paths.append(str(path.resolve()))
            validate_truth(sample.get("truth"), len(paths), mode, labels)
            tasks.append({"id": sample_id, "target": sample["target"], "question": sample["question"],
                          "condition": condition, "images": paths, "truth": sample.get("truth"),
                          "task": "recognition", "recognition_mode": mode})
    if not tasks:
        raise ValueError("No inference conditions remain after filtering")
    return tasks
