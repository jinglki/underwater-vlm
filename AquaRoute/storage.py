"""Atomic metadata, cross-platform run locks, and append-only prediction files."""
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def versions() -> dict:
    result = {}
    for name in ("torch", "transformers", "Pillow", "numpy", "causal-conv1d", "PyYAML"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def fingerprint(tasks: list[dict], cfg: dict) -> str:
    """Exclude only batch size/CPU workers so a resumed run can reduce memory use.

    Model configuration files are hashed; weight file sizes and mtimes detect
    common weight replacements without hashing multi-GB checkpoints on startup.
    Image metadata detects common image replacements; it is not a content proof.
    """
    from .config import project_path

    digest = hashlib.sha256()
    relevant = {key: cfg.get(key) for key in ("model", "recognition", "task")}
    relevant["include_original"] = cfg["inference"].get("include_original", True)
    digest.update(json.dumps({"tasks": tasks, "config": relevant}, sort_keys=True, ensure_ascii=False).encode())
    for name in ("model.py", "protocol.py", "engine.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    digest.update(json.dumps(versions(), sort_keys=True).encode())
    model_dir = project_path(cfg["model"]["path"])
    if model_dir.is_dir():
        for path in sorted(model_dir.iterdir()):
            if path.is_file():
                stat = path.stat()
                digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
                if path.suffix in {".json", ".jinja"} and stat.st_size < 2_000_000:
                    digest.update(path.read_bytes())
    for path in sorted({p for task in tasks for p in task["images"]}):
        stat = Path(path).stat()
        digest.update(f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


@contextmanager
def run_lock(directory: Path):
    """Advisory lock released by the OS on normal exit, exception or termination."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".aquaroute.lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0, os.SEEK_END)
            if lock.tell() == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another AquaRoute process is using this run directory") from exc
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another AquaRoute process is using this run directory") from exc
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class PredictionStore:
    """Public predictions.jsonl contains the same records returned by inference."""
    def __init__(self, directory: Path, tasks: list[dict], cfg: dict, resume: bool):
        self.directory = directory
        self.path = directory / "predictions.jsonl"
        self.meta_path = directory / "run.json"
        expected = fingerprint(tasks, cfg)
        if self.meta_path.exists():
            if not resume:
                raise ValueError("Run already exists. Use --resume or a new --run-id")
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if meta.get("fingerprint") != expected:
                raise ValueError("Inputs/model/code/environment changed; use a new --run-id")
        else:
            if self.path.exists():
                raise ValueError("Existing predictions have no AquaRoute metadata; use a new --run-id")
            meta = {"format": 1, "fingerprint": expected, "task": "recognition",
                    "mode": cfg.get("recognition", {}).get("mode", "presence"),
                    "total_conditions": len(tasks), "config": cfg, "versions": versions(),
                    "invocations": []}
            atomic_json(self.meta_path, meta)
        self.records = self._read_prefix(tasks) if self.path.exists() else []
        meta.setdefault("invocations", []).append({"batch_size": cfg["inference"].get("batch_size", 8),
                                                   "starting_condition": len(self.records)})
        self.meta = meta
        self.set_state("running")
        self.stream = self.path.open("a", encoding="utf-8", newline="\n")

    def _read_prefix(self, tasks):
        records = []
        with self.path.open("rb+") as stream:
            while True:
                start = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    stream.truncate(start)
                    break
                record = json.loads(line)
                index = len(records)
                if index >= len(tasks) or any(record.get(k) != v for k, v in tasks[index].items()):
                    raise ValueError(f"Checkpoint record {index} does not match prepared input")
                if "prediction" not in record or "error" not in record:
                    raise ValueError(f"Incomplete checkpoint record {index}")
                records.append(record)
        return records

    def append(self, records: list[dict]):
        self.stream.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.records.extend(records)

    def set_state(self, state):
        self.meta["state"] = state
        self.meta["completed_conditions"] = len(self.records)
        atomic_json(self.meta_path, self.meta)

    def close(self):
        self.stream.close()
