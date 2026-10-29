import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiment import (file_sha256, prediction_row, read_manifest,
                        validate_manifest, validate_prediction)
from provenance_fixtures import run_manifest
from replay_data import build_catalog, read_predictions
from test_data_processing import record, stream_events, write_stream


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.manifest = validate_manifest(run_manifest())
        self.record = record()
        self.record.source_sha256 = "a" * 64
        self.row = prediction_row(self.record, {name: .6 for name in self.manifest["models"]},
                                  self.manifest, "c" * 64)

    def test_prediction_changes_cannot_claim_existing_provenance(self):
        changes = {"run_id": "different-run", "run_manifest_sha256": "d" * 64,
                   "source_sha256": "e" * 64, "feature_schema": {},
                   "observation_seconds": 29, "partition": "test",
                   "model_artifact_sha256": {}, "prediction_schema_version": 1}
        for field, value in changes.items():
            row = {**self.row, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_prediction(row, self.manifest, "c" * 64)

    def test_invalid_probabilities_are_rejected(self):
        for value in (True, float("nan"), float("inf"), -1, 1.1):
            row = copy.deepcopy(self.row)
            row["probabilities"]["logistic_temporal"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_prediction(row, self.manifest, "c" * 64)

    def test_unseen_round_is_unassigned_not_test(self):
        self.record.round_id = "round-new"
        row = prediction_row(self.record, self.row["probabilities"], self.manifest, "c" * 64)
        self.assertEqual(row["partition"], "unassigned")

    def test_group_overlap_and_smoke_test_partition_are_rejected(self):
        for change in ("overlap", "test", "series"):
            manifest = copy.deepcopy(self.manifest)
            if change == "overlap":
                manifest["partitions"]["validation"][0]["round"] = ["series-1", "game-1", "round-2"]
            elif change == "test":
                manifest["partitions"]["test"] = manifest["partitions"]["validation"]
            else:
                manifest["mode"] = "series_holdout"
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_manifest(manifest)

    def test_import_requires_manifest_and_rejects_duplicates_and_legacy_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(json.dumps(self.row))
            with self.assertRaisesRegex(ValueError, "require"):
                read_predictions(path)
            parsed = read_predictions(path, self.manifest, "c" * 64)
            self.assertEqual(parsed[self.record.key]["run"]["purpose"], "ui_fixture")
            path.write_text(json.dumps(self.row) + "\n" + json.dumps(self.row))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_predictions(path, self.manifest, "c" * 64)
            path.write_text(json.dumps({"series_id": "s", "game_id": "g", "round_id": "r",
                                        "p_attacker_win": .6, "observation_seconds": 30}))
            with self.assertRaisesRegex(ValueError, "legacy"):
                read_predictions(path, self.manifest, "c" * 64)

    def test_catalog_binds_predictions_to_actual_source_bytes_and_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            path = data / "events.jsonl"
            entries = stream_events()
            bounds = {"name": "test-map", "bounds": {"min": {"x": 0, "y": 0}, "max": {"x": 100, "y": 100}}}
            entries[0][1]["actor"]["state"]["map"] = bounds
            entries[1][1]["seriesState"]["games"][0]["map"] = bounds
            write_stream(path, entries)
            self.record.source_sha256 = file_sha256(path)
            manifest_path = root / "run.json"
            manifest_path.write_text(json.dumps(run_manifest(self.record.source_sha256, cutoff=20)))
            manifest, digest = read_manifest(manifest_path)
            prediction_path = root / "predictions.jsonl"
            row = prediction_row(self.record, self.row["probabilities"], manifest, digest)
            prediction_path.write_text(json.dumps(row))
            catalog, payloads = build_catalog(data, prediction_path, manifest_path)
            self.assertEqual(catalog["prediction_count"], 1)
            replay = json.loads(next(iter(payloads.values())))
            self.assertEqual(replay["ml"]["cutoff"], 20)
            self.assertEqual(len(replay["ml"]["features"]), 20)
            self.assertEqual(len(replay["ml"]["temporal_features"]), 8)
            self.assertEqual(replay["ml"]["prediction"]["partition"], "train")
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "source hash"):
                build_catalog(data, prediction_path, manifest_path)
