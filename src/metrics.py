from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Iterable

CLASSES = ["normal", "suspected_opacity", "uncertain"]


def accuracy(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    y_true = list(y_true); y_pred = list(y_pred)
    if not y_true:
        return 0.0
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


def macro_f1(y_true: Iterable[str], y_pred: Iterable[str], classes: list[str] = CLASSES) -> float:
    y_true = list(y_true); y_pred = list(y_pred)
    scores = []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        scores.append(f1)
    return sum(scores) / len(scores)


def confusion_counts(y_true: Iterable[str], y_pred: Iterable[str]) -> dict[str, int]:
    counts = Counter()
    for t, p in zip(y_true, y_pred):
        counts[f"{t}__{p}"] += 1
    return dict(counts)


def confusion_matrix(y_true: Iterable[str], y_pred: Iterable[str], classes: list[str] = CLASSES) -> list[dict[str, int | str]]:
    y_true = list(y_true); y_pred = list(y_pred)
    return [
        {"expected_label": c, **{p: sum(t == c and pred == p for t, pred in zip(y_true, y_pred)) for p in classes}}
        for c in classes
    ]


def per_class_metrics(y_true: Iterable[str], y_pred: Iterable[str], classes: list[str] = CLASSES) -> list[dict[str, float | int | str]]:
    y_true = list(y_true); y_pred = list(y_pred)
    rows = []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "class": c,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall_sensitivity": round(recall, 4),
            "f1": round(f1, 4),
        })
    return rows


def specificity_metrics(y_true: Iterable[str], y_pred: Iterable[str], classes: list[str] = CLASSES) -> list[dict[str, float | int | str]]:
    y_true = list(y_true); y_pred = list(y_pred)
    rows = []
    for c in classes:
        tn = sum(t != c and p != c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        specificity = tn / (tn + fp) if tn + fp else 0.0
        rows.append({"class": c, "tn": tn, "fp": fp, "specificity": round(specificity, 4)})
    return rows


def summarize_metrics(rows: list[dict]) -> dict[str, float]:
    y_true = [r["label"] for r in rows]
    y_pred = [r["predicted_class"] for r in rows]
    json_valid = [r.get("json_valid", True) for r in rows]
    warnings = [bool(r.get("warning")) for r in rows]
    latencies = [float(r.get("latency_ms", 0) or 0) for r in rows]
    return {
        "n": len(rows),
        "accuracy": round(accuracy(y_true, y_pred), 4),
        "macro_f1": round(macro_f1(y_true, y_pred), 4),
        "json_valid_rate": round(sum(json_valid) / len(json_valid), 4) if rows else 0,
        "warning_rate": round(sum(warnings) / len(warnings), 4) if rows else 0,
        "uncertain_rate": round(sum(p == "uncertain" for p in y_pred) / len(y_pred), 4) if rows else 0,
        "latency_median_ms": round(median(latencies), 4) if latencies else 0,
        "interpretation": "technical validation only; not medical performance",
    }
