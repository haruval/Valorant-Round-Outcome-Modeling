"""Export predictions from an approved local run with bound provenance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from baseline_models import baseline_probabilities, load_baselines
from data_processing import load_dataset
from experiment import cnn_artifact_sha256, file_sha256, prediction_row, read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Completed run containing run.json and its fitted artifacts")
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    try:
        manifest, manifest_hash = read_manifest(args.run_dir / "run.json")
        if manifest.get("purpose", "experiment") != "experiment":
            raise ValueError("UI fixtures are not trained model runs")
        baseline_path = args.run_dir / "baselines.json"
        baseline_hash = file_sha256(baseline_path)
        for name in ("train_prior", "logistic_snapshot", "logistic_temporal"):
            if baseline_hash != manifest["models"][name]["artifact_sha256"]:
                raise ValueError("Baseline artifact hash differs from the run manifest")
        if "cnn" in manifest["models"]:
            if cnn_artifact_sha256(args.run_dir / "model") != manifest["models"]["cnn"]["artifact_sha256"]:
                raise ValueError("CNN artifact hash differs from the run manifest")
        dataset = load_dataset([args.jsonl], manifest["observation_seconds"])
        if not dataset.rounds:
            raise ValueError("No eligible rounds at the run's observation cutoff")
        # This command loads fitted models and requires approval, even for JSON-only baselines.
        baselines = load_baselines(baseline_path)
        if baselines["observation_seconds"] != manifest["observation_seconds"]:
            raise ValueError("Baseline cutoff differs from the run manifest")
        probabilities = baseline_probabilities(baselines, dataset.features)
        if "cnn" in manifest["models"]:
            import torch
            from model import ValorantRoundCNN
            model = ValorantRoundCNN.load(args.run_dir / "model")
            if model.observation_seconds != manifest["observation_seconds"]:
                raise ValueError("CNN cutoff differs from the run manifest")
            probabilities["cnn"] = []
            with torch.inference_mode():
                for offset in range(0, len(dataset.rounds), args.batch_size):
                    values = torch.tensor(dataset.features[offset:offset + args.batch_size], dtype=torch.float32)
                    probabilities["cnn"].extend(torch.softmax(model(values), dim=-1)[:, 1].tolist())
        # Build all rows first, so provenance errors cannot emit a partially accepted export.
        rows = [prediction_row(record, {name: values[i] for name, values in probabilities.items()}, manifest, manifest_hash)
                for i, record in enumerate(dataset.rounds)]
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))
    for row in rows:
        print(json.dumps(row, allow_nan=False))


if __name__ == "__main__":
    main()
