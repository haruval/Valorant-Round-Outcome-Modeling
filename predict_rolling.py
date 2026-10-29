"""Approved-runtime export of time-indexed predictions from a completed rolling run."""
import argparse
import json
from pathlib import Path

from data_processing import discover_data_files
from experiment import cnn_artifact_sha256, file_sha256
from rolling_features import load_rolling_dataset
from rolling_experiment import baseline_probabilities, load_baselines, prediction_row, read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    manifest, digest = read_manifest(args.run_dir / "run.json")
    if manifest["purpose"] != "experiment":
        parser.error("UI fixtures cannot be used as trained runs")
    for name, item in manifest["models"].items():
        artifact = args.run_dir / item["artifact"]
        actual = cnn_artifact_sha256(artifact) if name == "temporal_cnn" else file_sha256(artifact)
        if actual != item["artifact_sha256"]:
            parser.error("Rolling model artifact hash mismatch")
    dataset = load_rolling_dataset(discover_data_files(args.data))
    if not dataset.features:
        parser.error("No eligible rolling windows")
    probabilities = baseline_probabilities(load_baselines(args.run_dir / "baselines.json"), dataset.features)
    if "temporal_cnn" in manifest["models"]:
        import torch
        from rolling_model import RollingTemporalCNN
        model = RollingTemporalCNN.load(args.run_dir / "model")
        probabilities["temporal_cnn"] = []
        with torch.inference_mode():
            for i in range(0, len(dataset.features), 64):
                batch = torch.tensor(dataset.features[i:i + 64], dtype=torch.float32)
                probabilities["temporal_cnn"].extend(torch.softmax(model(batch), dim=-1)[:, 1].tolist())
    rows = [prediction_row(r, dataset.endpoints[i], {name: p[i] for name, p in probabilities.items()}, manifest, digest)
            for i, r in enumerate(dataset.records)]
    for row in rows:
        print(json.dumps(row, allow_nan=False))


if __name__ == "__main__":
    main()
