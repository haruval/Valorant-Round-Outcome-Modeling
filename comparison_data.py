"""Attach separately verified validation runs and an untrained feature review."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from data_processing import discover_data_files, extract_rounds
from entry_support import preview_entry_support
from rolling_experiment import read_manifest, read_predictions


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_comparisons(config_path, repo):
    if config_path.stat().st_size > 100_000:
        raise ValueError("Comparison config exceeds 100 KB")
    config = json.loads(config_path.read_text())
    if config.get("schema") != "comparison-view-v1" or not 1 <= len(config.get("runs", [])) <= 16:
        raise ValueError("Invalid comparison config")
    loaded, names = [], set()
    for entry in config["runs"]:
        directory = (repo / entry["directory"]).resolve()
        allowed = (repo / "runs").resolve()
        if not directory.is_relative_to(allowed) or directory == allowed:
            raise ValueError("Comparison artifacts must be under project runs/")
        for name in ("run.json", "predictions.jsonl"):
            path = directory / name
            if not path.resolve().is_relative_to(allowed) or path.stat().st_size > 10_000_000:
                raise ValueError("Invalid comparison artifact path or size")
        manifest, manifest_hash = read_manifest(directory / "run.json")
        if manifest_hash != entry["manifest_sha256"] or digest(directory / "predictions.jsonl") != entry["predictions_sha256"]:
            raise ValueError("Comparison artifact differs from pinned hash")
        arguments = manifest["arguments"]
        if (manifest["purpose"] != "experiment" or manifest["mode"] != "development_smoke_test"
                or arguments["validation_map"] != entry["validation_map"]
                or arguments["model_seed"] != entry["model_seed"]
                or arguments["neural_model"] != entry["architecture"]
                or entry["architecture"] not in manifest["models"]
                or arguments["feature_set"] != entry["feature_set"]):
            raise ValueError("Comparison descriptor disagrees with run manifest")
        if entry["id"] in names:
            raise ValueError("Duplicate comparison ID")
        names.add(entry["id"])
        rows, public = read_predictions(directory / "predictions.jsonl", directory / "run.json")
        public.update(model_seed=arguments["model_seed"], validation_map=arguments["validation_map"])
        loaded.append({"id": entry["id"], "label": entry["label"], "run": public,
                       "architecture": entry["architecture"], "rows": rows,
                       "predictions_sha256": entry["predictions_sha256"]})
    if config.get("default_by_map", {}).keys() != {run["run"]["validation_map"] for run in loaded}:
        raise ValueError("Every validation map requires an explicit default")
    return config, loaded


def attach_comparisons(catalog, payloads, config, loaded):
    """Never combine model probabilities or manifests across runs."""
    used = set()
    for token, content in list(payloads.items()):
        payload = json.loads(content)
        key = tuple(payload[k] for k in ("series_id", "game_id", "round_id"))
        expected = {w["at_seconds"] for w in payload["ml"]["rolling"]["windows"]}
        choices = []
        for run in loaded:
            if run["run"]["validation_map"] != payload["map"]:
                continue
            rows = run["rows"].get(key, [])
            if {row["at_seconds"] for row in rows} != expected or len(rows) != len(expected):
                raise ValueError("Comparison predictions do not cover the same eligible windows")
            if any(row["partition"] != "validation" or row["source_sha256"] != payload["source_sha256"] for row in rows):
                raise ValueError("Comparison requires validation predictions from this replay source")
            if run["run"]["schema_id"] != payload["ml"]["rolling"]["schema_id"]:
                raise ValueError("Comparison feature contract differs from replay")
            used.add((run["id"], key))
            choices.append({k: run[k] for k in ("id", "label", "run", "architecture", "predictions_sha256")}
                           | {"predictions": rows})
        if choices:
            default = next((r for r in choices if r["id"] == config["default_by_map"][payload["map"]]), None)
            if default is None:
                raise ValueError("Default comparison run is not available")
            payload["ml"].update(comparisons=choices, selected_comparison=default["id"],
                                 run=default["run"], predictions=default["predictions"])
        payloads[token] = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    for run in loaded:
        for key, rows in run["rows"].items():
            if any(r["partition"] == "validation" for r in rows) and (run["id"], key) not in used:
                raise ValueError("Comparison validation round is absent from replay")
    catalog["comparison_selection"] = config["selection"]
    catalog["comparison_runs"] = len(loaded)
    catalog["prediction_run"] = None  # Each round/run carries its own provenance.
    catalog["prediction_count"] = sum(len(r["rows"][key]) for r in loaded for run_id, key in used if r["id"] == run_id)


def add_entry_review(data_dir, payloads):
    index = {}
    for token, raw in payloads.items():
        payload = json.loads(raw)
        index[tuple(payload[k] for k in ("series_id", "game_id", "round_id"))] = token, payload
    for source in discover_data_files(data_dir):
        source_hash = digest(source)
        for record in extract_rounds(source)[0]:
            if record.key not in index:
                continue
            token, payload = index[record.key]
            if payload["source_sha256"] != source_hash:
                raise ValueError("Entry-review source differs from replay")
            endpoints = [w["at_seconds"] for w in payload["ml"]["rolling"]["windows"]]
            payload["ml"]["entry_support"] = preview_entry_support(record, endpoints)
            payloads[token] = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
