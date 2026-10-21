"""Recognition prompts and strict parsing; no model or enhancement dependencies."""
import json
import re

PRESENCE_PROMPT = """你是图像目标识别器。每张图独立判断指定目标是否可见，不猜测被遮挡或无法辨认的目标。
只判断是否存在，不计数，不输出位置。按输入图片顺序，仅输出一个 JSON 对象，不解释。
格式：{"images":[{"image_index":0,"present":true}]}。
images 长度等于输入图片数，image_index 从 0 连续递增；present 必须是布尔值。不能确定时回答 false。"""

CLASSIFICATION_PROMPT = """你是图像物体分类器。每张图独立识别问题指定的主要目标，不计数，不猜测。
从用户提供的候选类别中选择一个标签；无法确定或不属于候选类别时选择 unknown。
按输入图片顺序，仅输出一个 JSON 对象，不解释。
格式：{"images":[{"image_index":0,"label":"unknown"}]}。
images 长度等于输入图片数，image_index 从 0 连续递增。"""


def make_prompt(question: str, target: str, mode: str, labels: list[str]) -> tuple[str, str]:
    text = f"目标物体：{target}\n用户问题：{question}\n"
    if mode == "presence":
        return PRESENCE_PROMPT, text + "只按要求判断目标是否存在，返回 JSON。"
    return CLASSIFICATION_PROMPT, text + "候选类别：" + json.dumps(labels, ensure_ascii=False) + "\n返回 JSON。"


def parse_prediction(raw: str, n_images: int, mode: str = "presence", labels=()) -> dict:
    candidate = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw.strip(), flags=re.I)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model did not return JSON")
        data = json.loads(candidate[start:end + 1])
    if (not isinstance(data, dict) or set(data) != {"images"}
            or not isinstance(data["images"], list) or len(data["images"]) != n_images):
        raise ValueError("Invalid images array")
    for index, item in enumerate(data["images"]):
        field = "present" if mode == "presence" else "label"
        if not isinstance(item, dict) or set(item) != {"image_index", field}:
            raise ValueError(f"Invalid image object at index {index}")
        if type(item["image_index"]) is not int or item["image_index"] != index:
            raise ValueError("image_index must match input order")
        if mode == "presence":
            if type(item["present"]) is not bool:
                raise ValueError("present must be a boolean")
        elif not isinstance(item["label"], str) or item["label"] not in (*labels, "unknown"):
            raise ValueError("label must be one of the configured labels or unknown")
    return data


def validate_truth(truth, n_images: int, mode: str, labels=()) -> None:
    if truth is None:
        return
    if not isinstance(truth, list) or len(truth) != n_images:
        raise ValueError("truth length must match images")
    for label in truth:
        if not isinstance(label, dict):
            raise ValueError("Each truth entry must be a mapping")
        if mode == "classification":
            if label.get("label") not in labels:
                raise ValueError("truth.label must be one of the configured classes")
        else:
            has_present = "present" in label
            has_count = "count" in label
            if not has_present and not has_count:
                raise ValueError("Presence truth requires present or a nonnegative count")
            if has_present and type(label["present"]) is not bool:
                raise ValueError("truth.present must be a boolean")
            if has_count and (type(label["count"]) is not int or label["count"] < 0):
                raise ValueError("truth.count must be a nonnegative integer")
            if has_present and has_count and label["present"] != (label["count"] > 0):
                raise ValueError("truth.present and truth.count disagree")


def actual_presence(label: dict) -> bool:
    return label["present"] if "present" in label else label["count"] > 0
