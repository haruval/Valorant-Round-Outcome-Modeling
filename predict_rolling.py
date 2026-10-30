"""Approved-runtime export of time-indexed predictions from a completed rolling run."""
import argparse
import json
from pathlib import Path

from data_processing import discover_data_files
from experiment import cnn_artifact_sha256, file_sha256
from rolling_features import load_rolling_dataset
from rolling_experiment import baseline_probabilities, load_baselines, prediction_row, read_manifest, NEURAL_MODELS


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
        actual = cnn_artifact_sha256(artifact) if name in NEURAL_MODELS else file_sha256(artifact)
        if actual != item["artifact_sha256"]:
            parser.error("Rolling model artifact hash mismatch")
    dataset = load_rolling_dataset(discover_data_files(args.data))
    if not dataset.features:
        parser.error("No eligible rolling windows")
    baselines = load_baselines(args.run_dir / "baselines.json")
    if baselines["input_policy"] != manifest["input_policy"]:
        parser.error("Baseline input policy differs from run manifest")
    probabilities = baseline_probabilities(baselines, dataset.features)
    for name in NEURAL_MODELS & set(manifest["models"]):
        import torch
        from rolling_model import MODEL_CLASSES
        torch.set_num_threads(1)
        model = MODEL_CLASSES[name].load(args.run_dir / "model")
        if model.input_policy != manifest["input_policy"]:
            parser.error("Neural input policy differs from run manifest")
        probabilities[name] = []
        with torch.inference_mode():
            for i in range(0, len(dataset.features), 64):
                batch = torch.tensor(dataset.features[i:i + 64], dtype=torch.float32)
                probabilities[name].extend(torch.softmax(model(batch), dim=-1)[:, 1].tolist())
    rows = [prediction_row(r, dataset.endpoints[i], {name: p[i] for name, p in probabilities.items()}, manifest, digest)
            for i, r in enumerate(dataset.records)]
    for row in rows:
        print(json.dumps(row, allow_nan=False))


if __name__ == "__main__":
    main()
