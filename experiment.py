"""Run/prediction provenance contracts. Standard library only; no model loading."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from baseline_features import SUMMARY_FEATURE_NAMES
from data_processing import FEATURE_NAMES, SCHEMA_VERSION

CNN_ARCHITECTURE = "valorant-round-cnn-small-v2"
PREDICTION_SCHEMA_VERSION = 2
MODEL_LABELS = {"train_prior": "Training prior", "logistic_snapshot": "Snapshot logistic",
                "logistic_temporal": "Temporal-summary logistic", "cnn": "Small 1D CNN"}
BASELINE_IDS = {"train_prior", "logistic_snapshot", "logistic_temporal"}
PARTITIONS = ("train", "validation", "test")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cnn_artifact_sha256(directory: Path) -> str:
    hashes = {name: file_sha256(directory / name) for name in ("config.json", "model.safetensors")}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def feature_schema() -> dict:
    return {"sequence_version": SCHEMA_VERSION, "sequence_names": FEATURE_NAMES,
            "temporal_version": 1, "temporal_names": SUMMARY_FEATURE_NAMES}


def is_hash(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def round_key(row) -> tuple:
    key = tuple(row[name] for name in ("series_id", "game_id", "round_id"))
    if any(not isinstance(value, str) or not value or len(value) > 200 for value in key):
        raise ValueError("Invalid round identity")
    return key


def partition_index(manifest: dict) -> dict:
    index, groups = {}, {}
    source_hashes = {source["sha256"] for source in manifest["data"]["sources"]}
    for partition in PARTITIONS:
        for item in manifest["partitions"][partition]:
            key = tuple(item["round"])
            if len(key) != 3 or any(not isinstance(value, str) or not value for value in key):
                raise ValueError("Invalid partition round identity")
            if key in index:
                raise ValueError("Duplicate round or overlapping partitions in run manifest")
            if item.get("source_sha256") not in source_hashes:
                raise ValueError("Partition source is absent from the run dataset")
            group = key[:2] if manifest["mode"] == "development_smoke_test" else key[:1]
            if group in groups and groups[group] != partition:
                raise ValueError("A data group crosses run partitions")
            groups[group] = partition
            index[key] = (partition, item["source_sha256"])
    return index


def validate_manifest(manifest: dict) -> dict:
    if (manifest.get("schema_version") != 1 or manifest.get("status") != "complete"
            or manifest.get("feature_schema") != feature_schema()
            or manifest.get("mode") not in {"development_smoke_test", "series_holdout"}
            or manifest.get("purpose", "experiment") not in {"experiment", "ui_fixture"}
            or not isinstance(manifest.get("run_id"), str)
            or re.fullmatch(r"[A-Za-z0-9_-]{1,80}", manifest["run_id"]) is None
            or type(manifest.get("observation_seconds")) is not int
            or not 8 <= manifest["observation_seconds"] <= 120):
        raise ValueError("Incompatible or incomplete run manifest")
    models = manifest.get("models", {})
    if not BASELINE_IDS <= models.keys() or not models.keys() <= MODEL_LABELS.keys():
        raise ValueError("Run must contain the three baselines and only supported models")
    for name, model in models.items():
        expected_features = ([] if name == "train_prior" else
                             SUMMARY_FEATURE_NAMES if name == "logistic_temporal" else FEATURE_NAMES)
        if (not is_hash(model.get("artifact_sha256")) or model.get("feature_names") != expected_features
                or model.get("artifact") != ("model" if name == "cnn" else "baselines.json")
                or (name == "cnn" and model.get("architecture") != CNN_ARCHITECTURE)):
            raise ValueError("Incompatible model provenance")
    sources = manifest.get("data", {}).get("sources", [])
    if not sources or any(not is_hash(source.get("sha256")) for source in sources):
        raise ValueError("Run requires dataset source hashes")
    if set(manifest.get("partitions", {})) != set(PARTITIONS):
        raise ValueError("Run requires explicit training, validation, and test partitions")
    if not manifest["partitions"]["train"] or not manifest["partitions"]["validation"]:
        raise ValueError("Run requires nonempty training and validation partitions")
    if manifest["mode"] == "development_smoke_test" and manifest["partitions"]["test"]:
        raise ValueError("A development smoke test cannot declare a test partition")
    if manifest["mode"] == "series_holdout" and not manifest["partitions"]["test"]:
        raise ValueError("Series holdout requires an independent test partition")
    partition_index(manifest)
    return manifest


def read_manifest(path: Path) -> tuple[dict, str]:
    if path.stat().st_size > 5_000_000:
        raise ValueError("Run manifest exceeds 5 MB")
    content = path.read_bytes()
    return validate_manifest(json.loads(content)), hashlib.sha256(content).hexdigest()


def partition_for(key: tuple, source_hash: str, manifest: dict) -> str:
    known = partition_index(manifest).get(key)
    if known is None:
        return "unassigned"
    if known[1] != source_hash:
        raise ValueError("Round source hash differs from the run manifest")
    return known[0]


def prediction_row(record, probabilities: dict, manifest: dict, manifest_hash: str) -> dict:
    row = {"prediction_schema_version": PREDICTION_SCHEMA_VERSION,
           "series_id": record.series_id, "game_id": record.game_id, "round_id": record.round_id,
           "source_sha256": record.source_sha256, "run_id": manifest["run_id"],
           "run_manifest_sha256": manifest_hash, "feature_schema": feature_schema(),
           "observation_seconds": manifest["observation_seconds"],
           "partition": partition_for(record.key, record.source_sha256, manifest),
           "model_artifact_sha256": {name: item["artifact_sha256"] for name, item in manifest["models"].items()},
           "probabilities": probabilities}
    validate_prediction(row, manifest, manifest_hash)
    return row


def validate_prediction(row: dict, manifest: dict, manifest_hash: str) -> None:
    key = round_key(row)
    if (row.get("prediction_schema_version") != PREDICTION_SCHEMA_VERSION
            or row.get("feature_schema") != manifest["feature_schema"]
            or row.get("run_id") != manifest["run_id"]
            or row.get("run_manifest_sha256") != manifest_hash
            or type(row.get("observation_seconds")) is not int
            or row["observation_seconds"] != manifest["observation_seconds"]
            or not is_hash(row.get("source_sha256"))):
        raise ValueError("Prediction provenance differs from the run manifest; legacy rows are unsupported")
    expected_artifacts = {name: item["artifact_sha256"] for name, item in manifest["models"].items()}
    if row.get("model_artifact_sha256") != expected_artifacts:
        raise ValueError("Prediction model artifact hashes differ from the run manifest")
    probabilities = row.get("probabilities", {})
    if set(probabilities) != set(manifest["models"]) or any(
            type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
            for p in probabilities.values()):
        raise ValueError("Invalid prediction probabilities")
    if row.get("partition") != partition_for(key, row["source_sha256"], manifest):
        raise ValueError("Prediction partition differs from the run manifest")


def public_run(manifest: dict, manifest_hash: str) -> dict:
    return {"run_id": manifest["run_id"], "manifest_sha256": manifest_hash,
            "mode": manifest["mode"], "purpose": manifest.get("purpose", "experiment"),
            "models": {name: MODEL_LABELS[name] for name in manifest["models"]},
            "observation_seconds": manifest["observation_seconds"],
            "feature_schema_version": SCHEMA_VERSION, "temporal_schema_version": 1}
