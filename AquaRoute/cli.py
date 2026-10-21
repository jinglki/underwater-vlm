"""Thin CLI dispatch; orchestration, model execution and metrics live separately."""
import argparse
import json
import logging
from pathlib import Path
import sys
import uuid

from .config import load_config, positive_int, project_path


def parser():
    result = argparse.ArgumentParser(description="AquaRoute image enhancement and multimodal recognition experiments")
    result.add_argument("command", choices=("download", "doctor", "validate", "prepare", "infer", "evaluate", "all"))
    result.add_argument("--config", help="Project-relative YAML; default config/recognition.yaml")
    result.add_argument("--task", choices=("count", "recognition"), help="Defaults to config task")
    result.add_argument("--manifest", help="JSONL manifest for prepare/all; paths relative to manifest")
    result.add_argument("--prepared", help="Prepared JSONL for infer; relative image paths use this file's directory")
    result.add_argument("--predictions", help="Prediction JSONL for evaluate")
    result.add_argument("--run-id", help="Output directory name; use the same ID with --resume")
    result.add_argument("--batch-size", type=int, help="Recognition GPU batch size override")
    result.add_argument("--limit", type=int, help="Recognition infer: at most this many NEW conditions")
    result.add_argument("--resume", action="store_true", help="Recognition infer: resume validated predictions")
    result.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW", help="Explicit prepared image path prefix replacement")
    result.add_argument("--source", choices=("huggingface", "modelscope"), default="huggingface", help="Model download provider")
    result.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return result


def legacy_count(args, config_path):
    """Delegate to the byte-preserved count CLI, without changing its semantics."""
    if args.resume or args.batch_size is not None or args.limit is not None or args.path_map:
        raise ValueError("Count is frozen: --resume/--batch-size/--limit/--path-map apply only to recognition")
    if args.command == "download":
        raise ValueError("Use the task-independent download handler")
    from mama.__main__ import main as count_main
    previous = sys.argv
    forwarded = [previous[0], args.command, "--config", str(project_path(config_path))]
    for name in ("manifest", "prepared", "predictions", "run_id"):
        value = getattr(args, name)
        if value is not None:
            forwarded.extend(["--" + name.replace("_", "-"), value])
    try:
        sys.argv = forwarded
        count_main()
    finally:
        sys.argv = previous


def run(args):
    config_path = args.config or ("config/count.yaml" if args.task == "count" else "config/recognition.yaml")
    cfg = load_config(config_path)
    task = args.task or cfg.get("task", "recognition")
    cfg["task"] = task
    if args.batch_size is not None:
        cfg["inference"]["batch_size"] = positive_int(args.batch_size, "batch-size")
    if args.limit is not None:
        positive_int(args.limit, "limit")
    if (args.resume or args.limit is not None) and args.command != "infer":
        raise ValueError("--resume and --limit are supported by recognition infer only")
    if args.command == "doctor":
        from .doctor import check
        report = check(cfg, require_enhancement=True)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 1
    if args.command == "validate":
        if bool(args.manifest) == bool(args.prepared):
            raise ValueError("validate requires exactly one of --manifest or --prepared")
        if task == "count":
            if not args.manifest:
                raise ValueError("Count validation supports --manifest; its original prepared path remains unchanged")
            from mama.pipeline import read_samples
            samples = read_samples(args.manifest)
            print(f"Valid count manifest: {len(samples)} samples")
        else:
            from .data import expand_conditions, load_jsonl, read_samples
            if args.manifest:
                samples = read_samples(args.manifest, cfg)
                print(f"Valid recognition manifest: {len(samples)} samples")
            else:
                if args.path_map:
                    raise ValueError("Use infer --path-map for relocated prepared files; validate uses stored paths")
                path = Path(args.prepared).resolve()
                tasks = expand_conditions(load_jsonl(path), cfg, base_dir=path.parent)
                print(f"Valid prepared input: {len(tasks)} conditions")
        return 0
    if args.command == "download":
        repo_id = cfg["model"].get("repo_id")
        if not repo_id:
            raise ValueError("model.repo_id is required")
        destination = project_path(cfg["model"]["path"])
        if args.source == "huggingface":
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=repo_id, local_dir=str(destination))
        else:
            from modelscope import snapshot_download
            snapshot_download(repo_id, local_dir=str(destination))
        print(f"Model downloaded: {destination}")
        return 0
    if task == "count":
        legacy_count(args, config_path)
        return 0
    from .data import expand_conditions, load_jsonl, read_samples
    from .metrics import evaluate
    from .storage import atomic_json

    run_id = args.run_id or uuid.uuid4().hex[:12]
    if Path(run_id).name != run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise ValueError("run-id must be a directory name")
    run_dir = project_path(cfg["inference"].get("output_dir", "runs")) / run_id
    path_maps = []
    for mapping in args.path_map:
        old, separator, new = mapping.partition("=")
        if not separator or not old.strip() or not new.strip():
            raise ValueError("path-map must be OLD=NEW with nonempty prefixes")
        path_maps.append((old, new))
    if args.command in {"prepare", "all"}:
        if not args.manifest:
            raise ValueError("--manifest is required")
        # Import only here: evaluate/help do not require OpenCV or PyTorch.
        from mama.pipeline import prepare, save_jsonl
        samples = read_samples(args.manifest, cfg)
        if (run_dir / "prepared.jsonl").exists() or (run_dir / "predictions.jsonl").exists():
            raise ValueError("Run output already exists; choose a new run-id or use infer --resume")
        prepared = prepare(samples, cfg, run_id)
        prepared_path = run_dir / "prepared.jsonl"
        save_jsonl(prepared_path, prepared)
        atomic_json(run_dir / "prepare_config.json", cfg)
        print(f"Prepared: {prepared_path}")
        if args.command == "prepare":
            return 0
    if args.command == "infer":
        if not args.prepared:
            raise ValueError("--prepared is required")
        prepared_path = Path(args.prepared).resolve()
        prepared = load_jsonl(prepared_path)
    if args.command in {"infer", "all"}:
        from .engine import infer
        tasks = expand_conditions(prepared, cfg, path_maps, prepared_path.parent)
        records = infer(tasks, cfg, run_dir, resume=args.resume, limit=args.limit)
        print(f"Predictions: {run_dir / 'predictions.jsonl'}")
    if args.command == "evaluate":
        if not args.predictions:
            raise ValueError("--predictions is required")
        prediction_path = Path(args.predictions)
        records = load_jsonl(prediction_path)
        # Fail fast when evaluating partial files, unless the user explicitly
        # extracts a standalone prediction file without its run metadata.
        meta_path = prediction_path.parent / "run.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if len(records) != meta["total_conditions"]:
                raise ValueError("Run is incomplete; resume inference before evaluation")
            if meta.get("task") != "recognition" or meta["config"].get("recognition", {}) != cfg.get("recognition", {}):
                raise ValueError("Evaluation recognition mode/labels differ from the saved run configuration")
    if args.command in {"evaluate", "all"}:
        mode_cfg = cfg.get("recognition", {})
        report = evaluate(records, mode_cfg.get("mode", "presence"), mode_cfg.get("labels", []))
        output = project_path(cfg["evaluation"].get("output_dir", "results")) / f"{run_id}_metrics.json"
        atomic_json(output, report)
        print(f"Metrics: {output}")
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")
    try:
        code = run(args)
    except KeyboardInterrupt:
        logging.warning("Interrupted. Recognition batches already written are retained; rerun infer with --resume.")
        code = 130
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        logging.error("%s", exc)
        if args.log_level == "DEBUG":
            raise
        code = 1
    raise SystemExit(code)
