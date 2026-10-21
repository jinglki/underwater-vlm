"""Recognition orchestration: one model, ordered batches, OOM splitting, durable resume."""
import logging
import time

from .protocol import parse_prediction
from .storage import PredictionStore, run_lock

LOG = logging.getLogger(__name__)


def infer(tasks, cfg, run_dir, *, resume=False, limit=None, model_factory=None):
    """Return ordered records; each completed batch is also saved immediately.

    model_factory is an injection point for deterministic CPU-only integration
    tests. Production imports the heavy backend only when new work remains.
    """
    from tqdm import tqdm

    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    mode_cfg = cfg.get("recognition", {})
    mode = mode_cfg.get("mode", "presence")
    labels = mode_cfg.get("labels", [])
    batch_size = cfg["inference"].get("batch_size", 8)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("inference.batch_size must be a positive integer")
    with run_lock(run_dir):
        store = PredictionStore(run_dir, tasks, cfg, resume)
        try:
            start = len(store.records)
            stop = len(tasks) if limit is None else min(len(tasks), start + limit)
            if start == len(tasks):
                store.set_state("complete")
                LOG.info("All %s conditions already saved", start)
                return store.records
            if model_factory is None:
                from .model import VisionModel
                model_factory = lambda: VisionModel(cfg["model"], mode_cfg, cfg["inference"].get("image_workers", 4))
            model = model_factory()
            fallback_batches = 0
            errors = 0
            began = time.perf_counter()

            def run_batch(batch, progress):
                nonlocal batch_size, fallback_batches, errors
                failure = None
                try:
                    requests = [(r["images"], r["question"], r["target"]) for r in batch]
                    raw_answers = model.predict_batch(requests) if len(batch) > 1 else [model.predict(*requests[0])]
                    if len(raw_answers) != len(batch) or any(not isinstance(raw, str) for raw in raw_answers):
                        raise ValueError("Backend returned an invalid answer batch")
                except (ValueError, RuntimeError, TypeError, OSError) as exc:
                    failure = str(exc)
                    oom = "out of memory" in failure.lower() or type(exc).__name__ == "OutOfMemoryError"
                    unsupported = isinstance(exc, TypeError)
                if failure is not None and len(batch) > 1:
                    fallback_batches += 1
                    progress.write(f"Batch {len(batch)} failed; retrying smaller batches: {failure}")
                    middle = len(batch) // 2
                    if oom or unsupported:
                        batch_size = min(batch_size, max(1, middle) if oom else 1)
                    if oom:
                        # The exception handler has exited; failed GPU tensors can be freed.
                        try:
                            import torch
                        except ImportError:
                            pass
                        else:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                    run_batch(batch[:middle], progress)
                    run_batch(batch[middle:], progress)
                    return
                output = []
                for index, original in enumerate(batch):
                    record = dict(original)
                    if failure is not None:
                        record.update(prediction=None, error=failure)
                    else:
                        raw = raw_answers[index]
                        record["raw"] = raw
                        try:
                            record["prediction"] = parse_prediction(raw, len(record["images"]), mode, labels)
                            record["error"] = None
                        except ValueError as exc:
                            record.update(prediction=None, error=str(exc))
                    errors += record["error"] is not None
                    output.append(record)
                store.append(output)
                progress.update(len(output))
                progress.set_postfix(sample=batch[-1]["id"], condition=batch[-1]["condition"], errors=errors, refresh=False)

            with tqdm(total=len(tasks), initial=start, desc="Running recognition", unit="condition", dynamic_ncols=True) as progress:
                offset = start
                while offset < stop:
                    batch = tasks[offset:min(offset + batch_size, stop)]
                    run_batch(batch, progress)
                    offset += len(batch)
            store.set_state("complete" if len(store.records) == len(tasks) else "partial")
            LOG.info("Saved %s/%s; %.3f s/condition excluding model load; new errors=%s; fallback batches=%s",
                     len(store.records), len(tasks), (time.perf_counter() - began) / (stop - start), errors, fallback_batches)
            return store.records
        except BaseException:
            store.set_state("interrupted")
            raise
        finally:
            store.close()
