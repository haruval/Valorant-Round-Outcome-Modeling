"""Approved-runtime re-export of predictions from a completed train_structure.py run."""
import argparse
import json
from pathlib import Path

from data_processing import discover_data_files
from experiment import cnn_artifact_sha256, file_sha256
from rolling_features import load_rolling_dataset
from rolling_experiment import prediction_row, read_manifest, NEURAL_MODELS
from structure_inputs import (extended_baseline_probabilities, extension_windows, load_extended_baselines,
                              validate_extension_block)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    manifest, digest = read_manifest(args.run_dir / "run.json")
    if manifest["purpose"] != "experiment":
        parser.error("UI fixtures cannot be used as trained runs")
    block = validate_extension_block(manifest)
    for name, item in manifest["models"].items():
        artifact = args.run_dir / item["artifact"]
        actual = cnn_artifact_sha256(artifact) if name in NEURAL_MODELS else file_sha256(artifact)
        if actual != item["artifact_sha256"]:
            parser.error("Extended model artifact hash mismatch")
    dataset = load_rolling_dataset(discover_data_files(args.data))
    if not dataset.features:
        parser.error("No eligible rolling windows")
    extensions = extension_windows(dataset.records, dataset.endpoints)
    baselines = load_extended_baselines(args.run_dir / "baselines.json")
    if baselines["input_policy"] != manifest["input_policy"] or baselines["extension_policy"] != block["policy"]:
        parser.error("Baseline input policies differ from run manifest")
    probabilities = extended_baseline_probabilities(baselines, dataset.features, extensions)
    for name in NEURAL_MODELS & set(manifest["models"]):
        import torch
        from structure_model import MODEL_CLASSES
        torch.set_num_threads(1)
        model = MODEL_CLASSES[name].load(args.run_dir / "model")
        if (model.input_policy != manifest["input_policy"] or model.extension_policy != block["policy"]
                or model.architecture != block["architecture"]):
            parser.error("Neural input policies differ from run manifest")
        probabilities[name] = []
        with torch.inference_mode():
            for i in range(0, len(dataset.features), 64):
                batch = torch.tensor(dataset.features[i:i + 64], dtype=torch.float32)
                extra = torch.tensor(extensions[i:i + 64], dtype=torch.float32)
                probabilities[name].extend(torch.softmax(model(batch, extra), dim=-1)[:, 1].tolist())
    rows = [prediction_row(r, dataset.endpoints[i], {name: p[i] for name, p in probabilities.items()}, manifest, digest)
            for i, r in enumerate(dataset.records)]
    for row in rows:
        print(json.dumps(row, allow_nan=False))


if __name__ == "__main__":
    main()
