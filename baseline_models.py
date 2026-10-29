"""Baseline fitting (optional ML imports) and bounded JSON coefficient storage.

Importing this module does not fit or load a model. Its public model functions
are for an approved ML run, not the standard-library data tests or viewer.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from baseline_features import SUMMARY_FEATURE_NAMES, temporal_summary
from data_processing import FEATURE_NAMES, SCHEMA_VERSION

BASELINE_IDS = ("train_prior", "logistic_snapshot", "logistic_temporal")


def fit_baselines(windows, labels, seed=42):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    bundle = {"schema_version": 1, "feature_schema_version": SCHEMA_VERSION,
              "observation_seconds": len(windows[0]),
              "train_prior": sum(labels) / len(labels), "models": {}}
    for name, names, transform in (
        ("logistic_snapshot", FEATURE_NAMES, lambda window: window[-1]),
        ("logistic_temporal", SUMMARY_FEATURE_NAMES, temporal_summary),
    ):
        matrix = np.asarray([transform(window) for window in windows], dtype=np.float64)
        pipeline = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000,
                                                                      random_state=seed))
        pipeline.fit(matrix, labels)
        scaler, estimator = pipeline.steps[0][1], pipeline.steps[1][1]
        if estimator.classes_.tolist() != [0, 1]:
            raise ValueError("Baseline fitting requires both outcome classes")
        bundle["models"][name] = {
            "feature_names": names, "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(), "coefficients": estimator.coef_[0].tolist(),
            "intercept": float(estimator.intercept_[0]), "C": 1.0,
        }
    return bundle


def save_baselines(bundle, path: Path):
    path.write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_baselines(path: Path):
    if path.stat().st_size > 100_000:
        raise ValueError("Baseline file exceeds the size limit")
    bundle = json.loads(path.read_text(encoding="utf-8"))
    if (bundle.get("schema_version") != 1 or bundle.get("feature_schema_version") != SCHEMA_VERSION
            or type(bundle.get("observation_seconds")) is not int
            or not 8 <= bundle["observation_seconds"] <= 120
            or set(bundle.get("models", {})) != {"logistic_snapshot", "logistic_temporal"}
            or type(bundle.get("train_prior")) not in (int, float)
            or not 0 <= bundle["train_prior"] <= 1):
        raise ValueError("Incompatible baseline schema")
    for name, names in (("logistic_snapshot", FEATURE_NAMES), ("logistic_temporal", SUMMARY_FEATURE_NAMES)):
        model = bundle["models"][name]
        if model.get("feature_names") != names:
            raise ValueError("Incompatible baseline feature names")
        for field in ("mean", "scale", "coefficients"):
            values = model[field]
            if len(values) != len(names) or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
                raise ValueError("Malformed baseline coefficients")
        if any(v <= 0 for v in model["scale"]) or not math.isfinite(model["intercept"]):
            raise ValueError("Invalid baseline scaling or intercept")
    return bundle


def baseline_probabilities(bundle, windows):
    result = {name: [] for name in BASELINE_IDS}
    for window in windows:
        if len(window) != bundle["observation_seconds"]:
            raise ValueError("Observation window differs from the fitted baselines")
        result["train_prior"].append(bundle["train_prior"])
        for name, values in (("logistic_snapshot", window[-1]),
                             ("logistic_temporal", temporal_summary(window))):
            model = bundle["models"][name]
            score = model["intercept"] + sum(
                (value - mean) / scale * coefficient for value, mean, scale, coefficient in
                zip(values, model["mean"], model["scale"], model["coefficients"]))
            if not math.isfinite(score):
                raise ValueError("Nonfinite baseline score")
            # Stable sigmoid for both signs; no pickle or sklearn object loading.
            probability = 1 / (1 + math.exp(-score)) if score >= 0 else math.exp(score) / (1 + math.exp(score))
            result[name].append(probability)
    return result
