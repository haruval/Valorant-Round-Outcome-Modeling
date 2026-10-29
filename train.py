"""Fit three baselines first; optionally compare a small CNN on fixed partitions."""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from baseline_features import SUMMARY_FEATURE_NAMES
from baseline_models import fit_baselines, baseline_probabilities, save_baselines
from data_processing import DEFAULT_OBSERVATION_SECONDS, FEATURE_NAMES, discover_data_files, load_dataset
from evaluation import binary_metrics, split_groups
from experiment import (CNN_ARCHITECTURE, cnn_artifact_sha256, feature_schema, file_sha256,
                        prediction_row, read_manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing runs are never overwritten")
    parser.add_argument("--observation-seconds", type=int, default=DEFAULT_OBSERVATION_SECONDS)
    parser.add_argument("--include-cnn", action="store_true", help="Train the optional small CNN after the baselines")
    parser.add_argument("--epochs", type=int, default=30, help="CNN epochs; unused for baseline-only runs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model-seed", "--seed", dest="model_seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42, help="Independent of model initialization; keep fixed across comparisons")
    parser.add_argument("--smoke-test", action="store_true", help="Hold out a map, omit test set; development only")
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or not math.isfinite(args.lr) or args.lr <= 0
            or not 8 <= args.observation_seconds <= 120
            or any(not 0 <= seed < 2**32 for seed in (args.model_seed, args.split_seed))):
        parser.error("Require positive epochs/batch size/lr, 8–120 seconds, and seeds in [0, 2^32)")
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new run name")
    try:
        dataset = load_dataset(discover_data_files(args.data_dir), args.observation_seconds)
        groups = [json.dumps([r.series_id, r.game_id] if args.smoke_test else [r.series_id])
                  for r in dataset.rounds]
        partitions = split_groups(groups, args.split_seed, args.smoke_test)
        if {dataset.labels[i] for i in partitions["train"]} != {0, 1}:
            raise ValueError("Training partition needs both outcome classes")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    # No third-party imports before data/split validation; this command requires ML approval.
    import numpy as np
    random.seed(args.model_seed)
    np.random.seed(args.model_seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    def write_json(name, value):
        (args.output_dir / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    train_indices = partitions["train"]
    baselines = fit_baselines([dataset.features[i] for i in train_indices],
                             [dataset.labels[i] for i in train_indices], args.model_seed)
    save_baselines(baselines, args.output_dir / "baselines.json")
    baseline_hash = file_sha256(args.output_dir / "baselines.json")
    models = {name: {"artifact": "baselines.json", "artifact_sha256": baseline_hash, "feature_names": names}
              for name, names in (("train_prior", []), ("logistic_snapshot", FEATURE_NAMES),
                                  ("logistic_temporal", SUMMARY_FEATURE_NAMES))}
    # Fit/scaling sees training data only. Evaluation is postponed until model selection ends.
    history, best_epoch, parameter_count = [], None, None
    if args.include_cnn:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
        from model import ValorantRoundCNN

        torch.manual_seed(args.model_seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        features = np.asarray(dataset.features, dtype=np.float32)
        labels = np.asarray(dataset.labels, dtype=np.int64)
        loader = DataLoader(TensorDataset(torch.from_numpy(features[train_indices]),
                                         torch.from_numpy(labels[train_indices])),
                            batch_size=args.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(args.model_seed), num_workers=0)
        model = ValorantRoundCNN(args.observation_seconds)
        parameter_count = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        def cnn_probabilities(indices):
            model.eval()
            values = []
            with torch.inference_mode():
                for offset in range(0, len(indices), args.batch_size):
                    batch = torch.from_numpy(features[indices[offset:offset + args.batch_size]])
                    values.extend(torch.softmax(model(batch), dim=-1)[:, 1].tolist())
            return values

        best_loss = float("inf")
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            for values, targets in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(values), targets)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss; run stopped")
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(targets)
            validation = binary_metrics([dataset.labels[i] for i in partitions["validation"]],
                                        cnn_probabilities(partitions["validation"]))
            history.append({"epoch": epoch, "training_loss": total_loss / len(train_indices), "validation": validation})
            if validation["log_loss"] < best_loss:
                best_loss, best_epoch = validation["log_loss"], epoch
                model.save(args.output_dir / "model")
            write_json("history.json", history)
            print(f"epoch={epoch} train_loss={total_loss / len(train_indices):.4f} val_loss={validation['log_loss']:.4f}")
        model = ValorantRoundCNN.load(args.output_dir / "model")
        models["cnn"] = {"artifact": "model", "artifact_sha256": cnn_artifact_sha256(args.output_dir / "model"),
                         "architecture": CNN_ARCHITECTURE, "feature_names": FEATURE_NAMES}

    probabilities = baseline_probabilities(baselines, dataset.features)
    if args.include_cnn:
        probabilities["cnn"] = cnn_probabilities(list(range(len(dataset.rounds))))
    results = {"mode": "development_smoke_test" if args.smoke_test else "series_holdout",
               "best_epoch": best_epoch, "partitions": {}}
    for partition in ("validation", "test"):
        indices = partitions[partition]
        if not indices:
            continue
        truth = [dataset.labels[i] for i in indices]
        results["partitions"][partition] = {
            name: binary_metrics(truth, [values[i] for i in indices]) for name, values in probabilities.items()}
        majority = float(baselines["train_prior"] > .5)
        results["partitions"][partition]["majority"] = binary_metrics(truth, [majority] * len(indices))
    write_json("metrics.json", results)
    libraries = ["numpy", "scikit-learn"] + (["torch", "safetensors"] if args.include_cnn else [])
    manifest = {
        "schema_version": 1, "status": "complete", "purpose": "experiment", "run_id": uuid.uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(), "mode": results["mode"],
        "observation_seconds": args.observation_seconds, "feature_schema": feature_schema(),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "data": dataset.audit, "models": models,
        "partitions": {name: [{"round": list(dataset.rounds[i].key), "label": dataset.labels[i],
                               "source_sha256": dataset.rounds[i].source_sha256} for i in indices]
                       for name, indices in partitions.items()},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        **{name: version(name) for name in libraries}},
        "implementation_sha256": {name: file_sha256(Path(__file__).resolve().parent / name) for name in
                                  ("train.py", "model.py", "baseline_models.py", "baseline_features.py",
                                   "data_processing.py", "evaluation.py", "experiment.py")},
        "cnn_parameter_count": parameter_count, "cnn_best_epoch": best_epoch,
        "selection": {"baselines": "fixed C=1; training-only scaler; no hyperparameter search",
                      "cnn": "validation_log_loss" if args.include_cnn else None}, "device": "cpu",
    }
    write_json("run.json", manifest)
    manifest, manifest_hash = read_manifest(args.output_dir / "run.json")
    with (args.output_dir / "predictions.jsonl").open("x", encoding="utf-8") as output:
        for i, record in enumerate(dataset.rounds):
            row = prediction_row(record, {name: values[i] for name, values in probabilities.items()}, manifest, manifest_hash)
            output.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
