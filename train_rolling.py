"""Approved-runtime entry point: rolling baselines and optional temporal CNN."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import uuid
from importlib.metadata import version
from pathlib import Path

from data_processing import discover_data_files
from evaluation import split_groups
from experiment import cnn_artifact_sha256, file_sha256
from rolling_features import load_rolling_dataset, schema
from rolling_experiment import fit_baselines, baseline_probabilities, metrics, prediction_row, read_manifest
from rolling_inputs import FEATURE_SETS, derive_input_policy, support_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    neural = parser.add_mutually_exclusive_group()
    neural.add_argument("--include-cnn", action="store_const", const="temporal_cnn", dest="neural_model")
    neural.add_argument("--neural-model", choices=("temporal_cnn", "pooled_mlp"))
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default="all")
    parser.add_argument("--validation-map", choices=("breeze", "ascent"), help="Explicit two-map smoke-test direction")
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    if args.output_dir.exists() or args.epochs < 1 or any(not 0 <= s < 2**32 for s in (args.model_seed, args.split_seed)):
        parser.error("Require a new output directory, positive epochs, and seeds in [0, 2^32)")
    dataset = load_rolling_dataset(discover_data_files(args.data_dir))
    groups = [json.dumps(list(r.key[:2] if args.smoke_test else r.key[:1])) for r in dataset.records]
    try:
        partitions = split_groups(groups, args.split_seed, args.smoke_test)
        if args.validation_map:
            if not args.smoke_test or {r.map_name for r in dataset.records} != {"breeze", "ascent"} or len(set(groups)) != 2:
                raise ValueError("Explicit map direction requires a two-map smoke test")
            partitions = {"train": [i for i, r in enumerate(dataset.records) if r.map_name != args.validation_map],
                          "validation": [i for i, r in enumerate(dataset.records) if r.map_name == args.validation_map], "test": []}
        train, validation = partitions["train"], partitions["validation"]
        if {dataset.labels[i] for i in train} != {0, 1}:
            raise ValueError("Training needs both classes")
    except ValueError as exc:
        parser.error(str(exc))
    input_policy = derive_input_policy([dataset.features[i] for i in train], args.feature_set)
    input_support = {part: support_report([dataset.features[i] for i in indices], input_policy)
                     for part, indices in partitions.items()}
    # Everything below requires approval to execute ML dependencies and fit models.
    import numpy as np
    random.seed(args.model_seed)
    np.random.seed(args.model_seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        (args.output_dir / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    write("input-support.json", input_support)

    baselines = fit_baselines([dataset.features[i] for i in train], [dataset.labels[i] for i in train],
                             [dataset.weights[i] for i in train], args.model_seed, input_policy)
    write("baselines.json", baselines)
    models = {name: {"artifact": "baselines.json", "artifact_sha256": file_sha256(args.output_dir / "baselines.json")}
              for name in ("train_prior", "snapshot_logistic", "history_logistic")}
    best_epoch, parameter_count = None, None
    if args.neural_model:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from rolling_model import MODEL_CLASSES
        model_class = MODEL_CLASSES[args.neural_model]
        torch.manual_seed(args.model_seed)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        x = torch.tensor(dataset.features, dtype=torch.float32)
        y = torch.tensor(dataset.labels, dtype=torch.long)
        weights = torch.tensor(dataset.weights, dtype=torch.float32)
        loader = DataLoader(TensorDataset(x[train], y[train], weights[train]), batch_size=32, shuffle=True,
                            generator=torch.Generator().manual_seed(args.model_seed), num_workers=0)
        model = model_class(input_policy)
        parameter_count = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        mean_weight = weights[train].mean()

        def neural_probabilities(indices):
            model.eval()
            values = []
            with torch.inference_mode():
                for start in range(0, len(indices), 64):
                    values.extend(torch.softmax(model(x[indices[start:start + 64]]), dim=-1)[:, 1].tolist())
            return values

        history, best = [], float("inf")
        for epoch in range(1, args.epochs + 1):
            model.train()
            for batch_x, batch_y, batch_w in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = (torch.nn.functional.cross_entropy(model(batch_x), batch_y, reduction="none") * batch_w / mean_weight).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite rolling training loss")
                loss.backward()
                optimizer.step()
            result = metrics(dataset, validation, neural_probabilities(validation))
            history.append({"epoch": epoch, "train": metrics(dataset, train, neural_probabilities(train)), "validation": result})
            if result["round_mean_log_loss"] < best:
                best, best_epoch = result["round_mean_log_loss"], epoch
                model.save(args.output_dir / "model")
            write("history.json", history)
            print(f"epoch={epoch} validation_round_mean_log_loss={result['round_mean_log_loss']:.4f}")
        model = model_class.load(args.output_dir / "model")
        models[args.neural_model] = {"artifact": "model", "artifact_sha256": cnn_artifact_sha256(args.output_dir / "model")}
    probabilities = baseline_probabilities(baselines, dataset.features)
    if args.neural_model:
        probabilities[args.neural_model] = neural_probabilities(list(range(len(dataset.features))))
    results = {part: {name: metrics(dataset, indices, [values[i] for i in indices]) for name, values in probabilities.items()}
               for part, indices in partitions.items() if part != "train" and indices}
    write("metrics.json", results)
    unique_parts = {}
    for name, indices in partitions.items():
        unique = {dataset.records[i].key: dataset.records[i] for i in indices}
        unique_parts[name] = [{"round": list(key), "source_sha256": r.source_sha256} for key, r in unique.items()]
    libraries = ["numpy", "scikit-learn"] + (["torch", "safetensors"] if args.neural_model else [])
    manifest = {"schema_version": "rolling-run-v1", "status": "complete", "purpose": "experiment",
                "run_id": hashlib.sha256(uuid.uuid4().bytes).hexdigest(), "schema": schema(),
                "mode": "development_smoke_test" if args.smoke_test else "series_holdout",
                "models": models, "partitions": unique_parts, "data": dataset.audit,
                "input_policy": input_policy,
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "environment": {name: version(name) for name in libraries},
                "neural_parameter_count": parameter_count, "best_epoch": best_epoch,
                "selection": "Neural model: validation round-mean log loss; baselines: fixed C=1",
                "implementation_sha256": {name: file_sha256(Path(__file__).parent / name) for name in
                                          ("rolling_features.py", "ability_taxonomy.py", "rolling_inputs.py", "rolling_experiment.py",
                                           "rolling_model.py", "train_rolling.py", "data_processing.py")}}
    write("run.json", manifest)
    manifest, digest = read_manifest(args.output_dir / "run.json")
    with (args.output_dir / "predictions.jsonl").open("x") as output:
        for i, record in enumerate(dataset.records):
            output.write(json.dumps(prediction_row(record, dataset.endpoints[i], {name: p[i] for name, p in probabilities.items()},
                                                  manifest, digest), allow_nan=False) + "\n")
    print(json.dumps({"run": str(args.output_dir), "windows": len(dataset.features),
                      "rounds": dataset.audit["counts"]["eligible_rounds"], "mode": manifest["mode"]}))


if __name__ == "__main__":
    main()
