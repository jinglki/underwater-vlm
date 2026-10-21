"""Recognition-only GPU batching. The historical count model lives in mama/model.py."""
import inspect
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from .config import project_path

from .protocol import make_prompt



class VisionModel:
    def __init__(self, config: dict, recognition: dict | None = None, image_workers: int = 4):
        self.recognition = recognition or {"mode": "presence"}
        self.image_workers = image_workers
        kind = config["type"]
        self.kind = kind
        path = project_path(config["path"])
        if not path.is_dir():
            raise FileNotFoundError(f"Model directory missing: {path}. Download weights first.")
        requested = config.get("device", "cuda")
        if str(requested).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; set model.device: cpu for a smoke test")
        self.device = requested
        dtype_name = config.get("dtype", "bfloat16")
        if dtype_name not in {"bfloat16", "float16", "float32"}:
            raise ValueError("model.dtype must be bfloat16, float16 or float32")
        dtype = getattr(torch, dtype_name)
        if requested == "cpu" and dtype != torch.float32:
            dtype = torch.float32
        trust = bool(config.get("trust_remote_code", False))
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=trust)
        if kind == "qwen35":
            from transformers import Qwen3_5ForConditionalGeneration as model_class
        elif kind == "llava":
            from transformers import LlavaForConditionalGeneration as model_class
        else:
            model_class = AutoModelForImageTextToText
        print(requested)
        load_kwargs = {"torch_dtype": dtype, "trust_remote_code": trust, "low_cpu_mem_usage": True}
        if config.get("attn_implementation"):
            load_kwargs["attn_implementation"] = config["attn_implementation"]
        self.model = model_class.from_pretrained(path, **load_kwargs).to(requested).eval()
        self.dtype = dtype
        self.max_new_tokens = int(config.get("max_new_tokens", 128))
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")

    def _messages(self, paths: list[str], question: str, target: str) -> tuple[list[dict], list]:
        """Read one request. Each request retains its own conversation and images."""
        if not paths:
            raise ValueError("At least one image is required")
        images = []
        try:
            for path in paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            system_prompt, user_prompt = make_prompt(
                question, target, self.recognition.get("mode", "presence"), self.recognition.get("labels", []))
            content = [{"type": "image", "image": image} for image in images]
            content.append({"type": "text", "text": user_prompt})
            if self.kind == "llava":
                content[-1]["text"] = system_prompt + "\n用户问题：" + content[-1]["text"]
                messages = [{"role": "user", "content": content}]
            else:
                messages = [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": content}]
            return messages, images
        except BaseException:
            for image in images:
                image.close()
            raise

    def _generate(self, conversations: list[list[dict]], batched: bool) -> list[str]:
        template_kwargs = {"enable_thinking": False} if self.kind == "qwen35" else {}
        processor_kwargs = {}
        tokenizer = self.processor.tokenizer
        previous_padding = tokenizer.padding_side
        try:
            if batched:
                # Decoder-only generation must select the last real prompt token.
                tokenizer.padding_side = "left"
                if tokenizer.pad_token_id is None:
                    if tokenizer.eos_token_id is None:
                        raise ValueError("Batched inference requires a pad or EOS token")
                    raise ValueError("Tokenizer has no pad token; retry requests individually")
                # New Transformers versions require processing options inside
                # processor_kwargs. Keep support for older installed versions.
                if "processor_kwargs" in inspect.signature(self.processor.apply_chat_template).parameters:
                    processor_kwargs["padding"] = True
                else:
                    template_kwargs["padding"] = True
            if processor_kwargs:
                template_kwargs["processor_kwargs"] = processor_kwargs
            inputs = self.processor.apply_chat_template(
                conversations if batched else conversations[0], tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt", **template_kwargs
            )
            inputs = {key: value.to(self.device, dtype=self.dtype if value.is_floating_point() else None)
                      if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
            with torch.inference_mode():
                tokens = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            # All rows include the full padded prompt; strip its common width.
            generated = tokens[:, inputs["input_ids"].shape[1]:]
            raw = self.processor.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if len(raw) != len(conversations):
                raise ValueError("Generated output count does not match request count")
            return raw
        finally:
            tokenizer.padding_side = previous_padding

    def predict(self, paths: list[str], question: str, target: str) -> str:
        """Original single-request interface and generation settings."""
        messages, images = self._messages(paths, question, target)
        try:
            return self._generate([messages], batched=False)[0]
        finally:
            for image in images:
                image.close()

    def predict_batch(self, requests: list[tuple[list[str], str, str]]) -> list[str]:
        """Return one independent answer per (paths, question, target), in order.

        Image decoding runs in CPU threads; one GPU model handles all requests
        in a single generate call. Do not call generate on this model concurrently.
        """
        if not requests:
            return []
        if len(requests) == 1:
            return [self.predict(*requests[0])]
        loaded = []
        try:
            # Bounded by batch size; no dataset-wide image cache or extra model copies.
            with ThreadPoolExecutor(max_workers=min(self.image_workers, len(requests))) as pool:
                futures = [pool.submit(self._messages, *request) for request in requests]
                failure = None
                for future in futures:
                    try:
                        loaded.append(future.result())
                    except Exception as exc:
                        if failure is None:
                            failure = exc
                if failure is not None:
                    raise failure
            return self._generate([messages for messages, _ in loaded], batched=True)
        finally:
            for _, images in loaded:
                for image in images:
                    image.close()

