"""Deterministic group partitions and binary metrics; standard library only."""
from __future__ import annotations

import math
import random


def split_groups(groups: list[str], seed: int = 42, smoke_test: bool = False) -> dict[str, list[int]]:
    unique = sorted(set(groups))
    minimum = 2 if smoke_test else 3
    if len(unique) < minimum:
        raise ValueError(f"Need at least {minimum} independent groups; found {len(unique)}. "
                         "Collect more series, or use --smoke-test for a development check only.")
    random.Random(seed).shuffle(unique)
    count = max(1, int(len(unique) * 0.2))
    validation = set(unique[:count])
    test = set() if smoke_test else set(unique[count:2 * count])
    training = set(unique) - validation - test
    return {name: [i for i, group in enumerate(groups) if group in selected]
            for name, selected in (("train", training), ("validation", validation), ("test", test))}


def binary_metrics(labels: list[int], probabilities: list[float]) -> dict:
    if not labels or len(labels) != len(probabilities):
        raise ValueError("Metrics require nonempty aligned labels and probabilities")
    if any(y not in (0, 1) for y in labels):
        raise ValueError("Labels must be binary")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    predictions = [int(p > 0.5) for p in probabilities]  # ties favor defender
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    positive = [p for y, p in zip(labels, probabilities) if y == 1]
    negative = [p for y, p in zip(labels, probabilities) if y == 0]
    auc = None
    balanced = None
    if positive and negative:
        auc = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative) / (len(positive) * len(negative))
        balanced = (tp / len(positive) + tn / len(negative)) / 2
    clipped = [min(max(p, 1e-7), 1 - 1e-7) for p in probabilities]
    return {
        "n": len(labels), "attacker_win_rate": sum(labels) / len(labels),
        "accuracy": (tp + tn) / len(labels), "balanced_accuracy": balanced,
        "attacker_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "roc_auc": auc,
        "log_loss": -sum(y * math.log(p) + (1 - y) * math.log(1 - p)
                         for y, p in zip(labels, clipped)) / len(labels),
        "brier_score": sum((y - p) ** 2 for y, p in zip(labels, probabilities)) / len(labels),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }
