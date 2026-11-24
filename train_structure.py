"""Approved-runtime entry point: v2 inputs plus one utility-structure arm.

Mirrors train_rolling.py statement for statement. Differences: a second, aligned
input array; an extension policy; and metrics for windows with a living attacker.
`--preflight` stops before any ML import and prints the derived input support.
"""
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
from rolling_experiment import metrics, prediction_row, read_manifest
from rolling_inputs import FEATURE_SETS, derive_input_policy, support_report
from structure_inputs import (ARMS, derive_extension_policy, extended_baseline_probabilities, extension_contract,
                              extension_support, extension_windows, fit_extended_baselines, living_attacker,
                              validate_extension_block)

IMPLEMENTATION = ("rolling_features.py", "ability_taxonomy.py", "rolling_inputs.py", "rolling_experiment.py",
                  "rolling_model.py", "data_processing.py", "entry_support.py", "utility_structure.py",
                  "structure_inputs.py", "structure_model.py", "train_structure.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--neural-model", choices=("temporal_cnn", "pooled_mlp"))
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default="no_spatial")
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--validation-map", choices=("breeze", "ascent"), help="Explicit two-map smoke-test direction")
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--preflight", action="store_true", help="Print input support and exit; no ML imports")
    args = parser.parse_args()
    if args.output_dir.exists() or args.epochs < 1 or any(not 0 <= s < 2**32 for s in (args.model_seed, args.split_seed)):
        parser.error("Require a new output directory, positive epochs, and seeds in [0, 2^32)")
    dataset = load_rolling_dataset(discover_data_files(args.data_dir))
    extensions = extension_windows(dataset.records, dataset.endpoints)
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
    extension_policy = derive_extension_policy([extensions[i] for i in train], args.arm)
    input_support = {part: support_report([dataset.features[i] for i in indices], input_policy)
                     for part, indices in partitions.items()}
    extension_report = {part: extension_support([extensions[i] for i in indices], extension_policy)
                        for part, indices in partitions.items()}
    strata = {part: {"all_windows": indices, "living_attacker": living_attacker(dataset.features, indices)}
              for part, indices in partitions.items() if part != "train" and indices}
    if args.preflight:
        print(json.dumps({"arm": args.arm, "feature_set": args.feature_set, "validation_map": args.validation_map,
                          "windows": {part: len(indices) for part, indices in partitions.items()},
                          "living_attacker_windows": {part: len(s["living_attacker"]) for part, s in strata.items()},
                          "retained_base_channels": sum(input_policy["keep"]),
                          "retained_extension_channels": extension_report["train"]["kept_channels"]}, allow_nan=False))
        return
    # Everything below requires approval to execute ML dependencies and fit models.
    import numpy as np
    random.seed(args.model_seed)
    np.random.seed(args.model_seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        (args.output_dir / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    write("input-support.json", input_support)
    write("extension-support.json", extension_report)

    baselines = fit_extended_baselines([dataset.features[i] for i in train], [extensions[i] for i in train],
                                       [dataset.labels[i] for i in train], [dataset.weights[i] for i in train],
                                       args.model_seed, input_policy, extension_policy)
    write("baselines.json", baselines)
    models = {name: {"artifact": "baselines.json", "artifact_sha256": file_sha256(args.output_dir / "baselines.json")}
              for name in ("train_prior", "snapshot_logistic", "history_logistic")}
    best_epoch, parameter_count, architecture = None, None, None
    if args.neural_model:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from structure_model import MODEL_CLASSES
        model_class = MODEL_CLASSES[args.neural_model]
        architecture = model_class.architecture
        torch.manual_seed(args.model_seed)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        x = torch.tensor(dataset.features, dtype=torch.float32)
        e = torch.tensor(extensions, dtype=torch.float32)
        y = torch.tensor(dataset.labels, dtype=torch.long)
        weights = torch.tensor(dataset.weights, dtype=torch.float32)
        loader = DataLoader(TensorDataset(x[train], e[train], y[train], weights[train]), batch_size=32, shuffle=True,
                            generator=torch.Generator().manual_seed(args.model_seed), num_workers=0)
        model = model_class(input_policy, extension_policy, init_seed=args.model_seed)
        parameter_count = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        mean_weight = weights[train].mean()

        def neural_probabilities(indices):
            model.eval()
            values = []
            with torch.inference_mode():
                for start in range(0, len(indices), 64):
                    batch = indices[start:start + 64]
                    values.extend(torch.softmax(model(x[batch], e[batch]), dim=-1)[:, 1].tolist())
            return values

        history, best = [], float("inf")
        for epoch in range(1, args.epochs + 1):
            model.train()
            for batch_x, batch_e, batch_y, batch_w in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = (torch.nn.functional.cross_entropy(model(batch_x, batch_e), batch_y, reduction="none") * batch_w / mean_weight).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite extended training loss")
                loss.backward()
                optimizer.step()
            validation_probabilities = neural_probabilities(validation)
            result = metrics(dataset, validation, validation_probabilities)
            alive = set(strata["validation"]["living_attacker"])
            history.append({"epoch": epoch, "train": metrics(dataset, train, neural_probabilities(train)), "validation": result,
                            "validation_living_attacker": metrics(
                                dataset, [i for i in validation if i in alive],
                                [p for i, p in zip(validation, validation_probabilities) if i in alive])})
            if result["round_mean_log_loss"] < best:
                best, best_epoch = result["round_mean_log_loss"], epoch
                model.save(args.output_dir / "model")
            write("history.json", history)
            print(f"epoch={epoch} validation_round_mean_log_loss={result['round_mean_log_loss']:.4f}")
        model = model_class.load(args.output_dir / "model")
        models[args.neural_model] = {"artifact": "model", "artifact_sha256": cnn_artifact_sha256(args.output_dir / "model")}
    probabilities = extended_baseline_probabilities(baselines, dataset.features, extensions)
    if args.neural_model:
        probabilities[args.neural_model] = neural_probabilities(list(range(len(dataset.features))))
    # Checkpoint selection above uses all validation windows, as in every earlier study.
    results = {part: {stratum: {name: metrics(dataset, indices, [values[i] for i in indices])
                                for name, values in probabilities.items()}
                      for stratum, indices in parts.items()} for part, parts in strata.items()}
    write("metrics.json", {part: values["all_windows"] for part, values in results.items()})
    write("metrics-living-attacker.json", {part: values["living_attacker"] for part, values in results.items()})
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
                "extension": {"contract": extension_contract(), "policy": extension_policy, "arm": args.arm,
                              "architecture": architecture},
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "environment": {name: version(name) for name in libraries},
                "neural_parameter_count": parameter_count, "best_epoch": best_epoch,
                "selection": "Neural model: validation round-mean log loss over all windows; baselines: fixed C=1",
                "implementation_sha256": {name: file_sha256(Path(__file__).parent / name) for name in IMPLEMENTATION}}
    write("run.json", manifest)
    manifest, digest = read_manifest(args.output_dir / "run.json")
    validate_extension_block(manifest)
    with (args.output_dir / "predictions.jsonl").open("x") as output:
        for i, record in enumerate(dataset.records):
            output.write(json.dumps(prediction_row(record, dataset.endpoints[i], {name: p[i] for name, p in probabilities.items()},
                                                  manifest, digest), allow_nan=False) + "\n")
    print(json.dumps({"run": str(args.output_dir), "windows": len(dataset.features), "arm": args.arm,
                      "rounds": dataset.audit["counts"]["eligible_rounds"], "mode": manifest["mode"]}))


if __name__ == "__main__":
    main()
