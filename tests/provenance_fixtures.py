"""Synthetic metadata only. These fixtures contain no fitted model artifacts."""
from baseline_features import SUMMARY_FEATURE_NAMES
from data_processing import FEATURE_NAMES
from experiment import feature_schema


def run_manifest(source_hash="a" * 64, key=("series-1", "game-1", "round-1"), cutoff=30):
    return {
        "schema_version": 1, "status": "complete", "purpose": "ui_fixture",
        "run_id": "synthetic-ui-check", "mode": "development_smoke_test",
        "observation_seconds": cutoff, "feature_schema": feature_schema(),
        "data": {"sources": [{"sha256": source_hash}]},
        "models": {name: {"artifact": "baselines.json", "artifact_sha256": "b" * 64,
                          "feature_names": names} for name, names in
                   (("train_prior", []), ("logistic_snapshot", FEATURE_NAMES),
                    ("logistic_temporal", SUMMARY_FEATURE_NAMES))},
        "partitions": {
            "train": [{"round": list(key), "source_sha256": source_hash}],
            "validation": [{"round": [key[0], "another-map", "round-1"], "source_sha256": source_hash}],
            "test": [],
        },
    }
