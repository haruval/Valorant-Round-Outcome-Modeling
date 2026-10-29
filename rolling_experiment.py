"""Rolling run contracts, JSON baseline inference, and round-balanced metrics.

No third-party imports. Baseline fitting/loading/inference still requires approval.
"""
from __future__ import annotations

import json
import hashlib
import math
from collections import defaultdict
from pathlib import Path

from evaluation import binary_metrics
from experiment import is_hash, partition_for, partition_index, round_key
from rolling_features import FEATURE_NAMES, HISTORY_SECONDS, STRIDE_SECONDS, schema

MODEL_LABELS = {"train_prior": "Training prior", "snapshot_logistic": "Rolling snapshot logistic",
                "history_logistic": "Window-summary logistic", "temporal_cnn": "Temporal CNN"}


def schema_hash(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def summary(window):
    return list(window[-1]) + [sum(row[i] for row in window) / len(window) for i in range(len(FEATURE_NAMES))]


def fit_baselines(windows, labels, weights, seed):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    bundle = {"schema": schema(), "train_prior": sum(y * w for y, w in zip(labels, weights)) / sum(weights), "models": {}}
    for name, transform in (("snapshot_logistic", lambda w: w[-1]), ("history_logistic", summary)):
        matrix = np.asarray([transform(w) for w in windows], dtype=float)
        scaler = StandardScaler().fit(matrix, sample_weight=weights)
        estimator = LogisticRegression(C=1, max_iter=3000, random_state=seed)
        estimator.fit(scaler.transform(matrix), labels, sample_weight=weights)
        if estimator.classes_.tolist() != [0, 1]:
            raise ValueError("Training requires both outcome classes")
        bundle["models"][name] = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                                  "coefficients": estimator.coef_[0].tolist(), "intercept": float(estimator.intercept_[0])}
    return bundle


def load_baselines(path):
    path = Path(path)
    if path.stat().st_size > 500_000:
        raise ValueError("Rolling baseline file exceeds size limit")
    bundle = json.loads(path.read_text())
    if (bundle.get("schema") != schema() or set(bundle.get("models", {})) != {"snapshot_logistic", "history_logistic"}
            or type(bundle.get("train_prior")) not in (int, float) or not 0 <= bundle["train_prior"] <= 1):
        raise ValueError("Incompatible rolling baseline schema")
    for name, model in bundle["models"].items():
        width = len(FEATURE_NAMES) * (2 if name == "history_logistic" else 1)
        for key in ("mean", "scale", "coefficients"):
            if len(model[key]) != width or any(type(v) not in (float, int) or not math.isfinite(v) for v in model[key]):
                raise ValueError("Malformed rolling coefficients")
        if any(v <= 0 for v in model["scale"]) or type(model["intercept"]) not in (float, int) or not math.isfinite(model["intercept"]):
            raise ValueError("Invalid rolling scaling/intercept")
    return bundle


def baseline_probabilities(bundle, windows):
    probabilities = {name: [] for name in ("train_prior", "snapshot_logistic", "history_logistic")}
    for window in windows:
        if len(window) != HISTORY_SECONDS or any(len(row) != len(FEATURE_NAMES) for row in window):
            raise ValueError("Incompatible rolling window")
        probabilities["train_prior"].append(bundle["train_prior"])
        for name, values in (("snapshot_logistic", window[-1]), ("history_logistic", summary(window))):
            model = bundle["models"][name]
            z = model["intercept"] + sum((x - mu) / scale * coef for x, mu, scale, coef in
                                         zip(values, model["mean"], model["scale"], model["coefficients"]))
            if not math.isfinite(z):
                raise ValueError("Nonfinite rolling prediction")
            probabilities[name].append(1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z)))
    return probabilities


def metrics(dataset, indices, probabilities):
    if len(indices) != len(probabilities):
        raise ValueError("Metrics require aligned windows and probabilities")
    by_round, by_time = defaultdict(list), defaultdict(list)
    for i, probability in zip(indices, probabilities):
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid rolling probability")
        by_round[dataset.records[i].key].append((dataset.labels[i], probability))
        by_time[dataset.endpoints[i]].append((dataset.labels[i], probability))
    if not by_round:
        raise ValueError("No windows to evaluate")
    per_round = [binary_metrics([y for y, _ in rows], [p for _, p in rows]) for rows in by_round.values()]
    return {"windows": len(indices), "rounds": len(by_round),
            "round_mean_log_loss": sum(r["log_loss"] for r in per_round) / len(per_round),
            "round_mean_brier": sum(r["brier_score"] for r in per_round) / len(per_round),
            "by_elapsed_seconds": {str(t): binary_metrics([y for y, _ in rows], [p for _, p in rows])
                                   for t, rows in sorted(by_time.items())}}


def read_manifest(path):
    path = Path(path)
    if path.stat().st_size > 5_000_000:
        raise ValueError("Rolling manifest exceeds size limit")
    content = path.read_bytes()
    manifest = json.loads(content)
    if (manifest.get("schema_version") != "rolling-run-v1" or manifest.get("status") != "complete"
            or manifest.get("schema") != schema() or not is_hash(manifest.get("run_id"))
            or manifest.get("mode") not in ("development_smoke_test", "series_holdout")
            or manifest.get("purpose") not in ("experiment", "ui_fixture")):
        raise ValueError("Incompatible rolling run manifest")
    models = manifest.get("models", {})
    if not {"train_prior", "snapshot_logistic", "history_logistic"} <= set(models) <= set(MODEL_LABELS):
        raise ValueError("Rolling run requires the comparison baselines")
    for name, item in models.items():
        if (not is_hash(item.get("artifact_sha256")) or
                item.get("artifact") != ("model" if name == "temporal_cnn" else "baselines.json")):
            raise ValueError("Invalid rolling model provenance")
    sources = manifest.get("data", {}).get("sources", [])
    if not sources or any(not is_hash(s.get("sha256")) for s in sources):
        raise ValueError("Missing rolling data hashes")
    parts = manifest.get("partitions", {})
    if set(parts) != {"train", "validation", "test"} or not parts["train"] or not parts["validation"]:
        raise ValueError("Invalid rolling partitions")
    if bool(parts["test"]) != (manifest["mode"] == "series_holdout"):
        raise ValueError("Invalid rolling test partition")
    partition_index(manifest)
    return manifest, hashlib.sha256(content).hexdigest()


def prediction_row(record, at, probabilities, manifest, manifest_hash):
    row = {"schema_version": "rolling-prediction-v1", "series_id": record.series_id,
           "game_id": record.game_id, "round_id": record.round_id, "at_seconds": at,
           "window_start": at - HISTORY_SECONDS, "schema_sha256": schema_hash(manifest["schema"]),
           "run_id": manifest["run_id"], "run_manifest_sha256": manifest_hash,
           "source_sha256": record.source_sha256,
           "partition": partition_for(record.key, record.source_sha256, manifest),
           "model_artifact_sha256": {name: item["artifact_sha256"] for name, item in manifest["models"].items()},
           "probabilities": probabilities}
    validate_prediction(row, manifest, manifest_hash)
    return row


def validate_prediction(row, manifest, manifest_hash):
    key = round_key(row)
    at = row.get("at_seconds")
    if (row.get("schema_version") != "rolling-prediction-v1" or row.get("schema_sha256") != schema_hash(manifest["schema"])
            or type(at) is not int or not HISTORY_SECONDS <= at < 600 or at % STRIDE_SECONDS
            or type(row.get("window_start")) is not int or row["window_start"] != at - HISTORY_SECONDS
            or row.get("run_id") != manifest["run_id"]
            or row.get("run_manifest_sha256") != manifest_hash or not is_hash(row.get("source_sha256"))
            or row.get("partition") != partition_for(key, row["source_sha256"], manifest)
            or row.get("model_artifact_sha256") != {n: m["artifact_sha256"] for n, m in manifest["models"].items()}):
        raise ValueError("Rolling prediction provenance mismatch")
    probabilities = row.get("probabilities", {})
    if set(probabilities) != set(manifest["models"]) or any(type(p) not in (int, float) or
            not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
        raise ValueError("Invalid rolling probabilities")


def read_predictions(path, manifest_path):
    if (path is None) != (manifest_path is None):
        raise ValueError("Supply both rolling predictions and their manifest")
    if path is None:
        return {}, None
    manifest, digest = read_manifest(manifest_path)
    if path.stat().st_size > 50_000_000:
        raise ValueError("Rolling prediction file exceeds 50 MB")
    grouped, seen = defaultdict(list), set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        validate_prediction(row, manifest, digest)
        key = (*round_key(row), row["at_seconds"])
        if key in seen:
            raise ValueError("Duplicate rolling prediction")
        seen.add(key)
        grouped[key[:3]].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: row["at_seconds"])
    public = {"run_id": manifest["run_id"], "manifest_sha256": digest, "mode": manifest["mode"],
              "purpose": manifest["purpose"], "models": {name: MODEL_LABELS[name] for name in manifest["models"]},
              "schema_id": manifest["schema"]["id"], "history_seconds": HISTORY_SECONDS, "stride_seconds": STRIDE_SECONDS}
    return dict(grouped), public
