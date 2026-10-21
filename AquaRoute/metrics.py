"""Recognition metrics with explicit denominators and invalid-answer accounting."""
from collections import defaultdict
import json

from .protocol import actual_presence, parse_prediction, validate_truth


def divide(a, b):
    return a / b if b else None


def evaluate(records: list[dict], mode="presence", labels=()) -> dict:
    groups = defaultdict(list)
    for record in records:
        if record.get("task") != "recognition" or record.get("recognition_mode") != mode:
            raise ValueError("Prediction task/mode does not match recognition configuration")
        groups[record["condition"]].append(record)
    report = {}
    for condition, items in groups.items():
        total = labeled = valid = correct = 0
        matrix = {"tn": 0, "fp": 0, "fn": 0, "tp": 0, "invalid": 0}
        actual_positive = actual_negative = invalid_positive = invalid_negative = 0
        class_matrix = {label: {pred: 0 for pred in (*labels, "unknown")} for label in labels}
        invalid_classes = {label: 0 for label in labels}
        class_totals = {label: 0 for label in labels}
        for record in items:
            truth = record.get("truth")
            if truth is None:
                continue
            validate_truth(truth, len(record["images"]), mode, labels)
            labeled += 1
            prediction = record.get("prediction")
            if prediction is not None:
                try:
                    parse_prediction(json.dumps(prediction), len(truth), mode, labels)
                except (ValueError, TypeError):
                    prediction = None
            for index, actual in enumerate(truth):
                total += 1
                item = prediction["images"][index] if prediction is not None else None
                if mode == "presence":
                    positive = actual_presence(actual)
                    actual_positive += positive
                    actual_negative += not positive
                    if item is None:
                        matrix["invalid"] += 1
                        invalid_positive += positive
                        invalid_negative += not positive
                        continue
                    predicted = item["present"]
                    key = ("tp" if predicted else "fn") if positive else ("fp" if predicted else "tn")
                    matrix[key] += 1
                    correct += predicted == positive
                else:
                    label = actual["label"]
                    class_totals[label] += 1
                    if item is None:
                        invalid_classes[label] += 1
                        continue
                    class_matrix[label][item["label"]] += 1
                    correct += item["label"] == label
                valid += 1
        result = {"samples": len(items), "labeled_samples": labeled, "labeled_images": total,
                  "accuracy": divide(correct, total), "valid_accuracy": divide(correct, valid),
                  "error_rate": divide(total - correct, total), "valid_response_rate": divide(valid, total)}
        if mode == "presence":
            tn, fp, fn, tp = (matrix[k] for k in ("tn", "fp", "fn", "tp"))
            recall_all = divide(tp, actual_positive)
            specificity_all = divide(tn, actual_negative)
            result.update(confusion_matrix=matrix, actual_positive=actual_positive, actual_negative=actual_negative,
                          invalid_positive=invalid_positive, invalid_negative=invalid_negative,
                          precision=divide(tp, tp + fp), recall=divide(tp, tp + fn),
                          f1=divide(2 * tp, 2 * tp + fp + fn),
                          recall_all=recall_all, specificity_all=specificity_all,
                          f1_all=divide(2 * tp, 2 * tp + fp + fn + invalid_positive),
                          balanced_accuracy_all=(recall_all + specificity_all) / 2
                          if recall_all is not None and specificity_all is not None else None,
                          majority_baseline=divide(max(actual_positive, actual_negative), total))
        else:
            per_class = {}
            for label in labels:
                tp = class_matrix[label][label]
                fp = sum(class_matrix[other][label] for other in labels if other != label)
                fn = class_totals[label] - tp  # includes invalid and unknown predictions
                per_class[label] = {"support": class_totals[label], "precision": divide(tp, tp + fp),
                                    "recall_all": divide(tp, tp + fn), "f1_all": divide(2 * tp, 2 * tp + fp + fn)}
            supported = [v for v in per_class.values() if v["support"]]
            result.update(confusion_matrix=class_matrix, invalid_by_class=invalid_classes, per_class=per_class,
                          macro_f1_all=sum((v["f1_all"] or 0) for v in supported) / len(supported) if supported else None,
                          majority_baseline=divide(max(class_totals.values(), default=0), total))
        report[condition] = result
    return report
