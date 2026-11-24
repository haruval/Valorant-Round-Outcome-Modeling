"""Synthetic prediction JSON only; no models are fitted or loaded in these tests."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from comparison_data import attach_comparisons, digest, load_comparisons
from rolling_experiment import prediction_row, read_manifest
from rolling_features import observe_round
from rolling_inputs import derive_input_policy
from test_rolling import located_record, rolling_manifest


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.repo = Path(self.directory.name)
        self.run = self.repo / "runs" / "synthetic"
        self.run.mkdir(parents=True)
        self.record = located_record()
        self.record.source_sha256 = "a" * 64
        manifest = rolling_manifest(self.record)
        manifest["purpose"] = "experiment"  # Locally fabricated test metadata, never served as a real run.
        manifest["partitions"]["train"], manifest["partitions"]["validation"] = manifest["partitions"]["validation"], manifest["partitions"]["train"]
        manifest["models"]["temporal_cnn"] = {"artifact": "model", "artifact_sha256": "c" * 64}
        manifest["input_policy"] = derive_input_policy([observe_round(self.record)["rows"][:20]], "no_spatial")
        manifest["arguments"] = dict(validation_map="ascent", model_seed=42, neural_model="temporal_cnn", feature_set="no_spatial")
        (self.run / "run.json").write_text(json.dumps(manifest))
        manifest, manifest_hash = read_manifest(self.run / "run.json")
        row = prediction_row(self.record, 20, {n: .6 for n in manifest["models"]}, manifest, manifest_hash)
        (self.run / "predictions.jsonl").write_text(json.dumps(row))
        self.config = dict(schema="comparison-view-v1", selection="Synthetic test", default_by_map={"ascent": "test"},
                           runs=[dict(id="test", label="Synthetic", directory="runs/synthetic", validation_map="ascent",
                                      model_seed=42, architecture="temporal_cnn", feature_set="no_spatial",
                                      manifest_sha256=manifest_hash, predictions_sha256=digest(self.run / "predictions.jsonl"))])
        self.path = self.repo / "comparison.json"
        self.payload = dict(series_id=self.record.series_id, game_id=self.record.game_id, round_id=self.record.round_id,
                            map="ascent", source_sha256=self.record.source_sha256,
                            ml={"rolling": {"schema_id": "rolling-gameplay-v2", "windows": [{"at_seconds": 20}]}})

    def load(self):
        self.path.write_text(json.dumps(self.config))
        return load_comparisons(self.path, self.repo)

    def attach(self, loaded, payload=None):
        payloads = {"round": json.dumps(payload or self.payload).encode()}
        catalog = {}
        attach_comparisons(catalog, payloads, self.config, loaded)
        return catalog, json.loads(payloads["round"])

    def test_json_run_keeps_model_artifact_and_partition_provenance(self):
        _, loaded = self.load()
        catalog, result = self.attach(loaded)
        choice, = result["ml"]["comparisons"]
        row, = choice["predictions"]
        self.assertEqual(row["partition"], "validation")
        self.assertEqual(row["run_id"], choice["run"]["run_id"])
        self.assertEqual(row["run_manifest_sha256"], choice["run"]["manifest_sha256"])
        self.assertEqual(row["model_artifact_sha256"]["temporal_cnn"], "c" * 64)
        self.assertEqual(result["ml"]["run"], choice["run"])
        self.assertIsNone(catalog["prediction_run"])

    def test_hash_mismatch_rejects_changed_predictions(self):
        (self.run / "predictions.jsonl").write_text("{}")
        with self.assertRaisesRegex(ValueError, "pinned hash"):
            self.load()

    def test_paths_cannot_escape_runs_including_symlinks(self):
        self.config["runs"][0]["directory"] = "../outside"
        with self.assertRaisesRegex(ValueError, "under project"):
            self.load()
        (self.repo / "runs" / "escape").symlink_to(self.repo)
        self.config["runs"][0]["directory"] = "runs/escape"
        with self.assertRaisesRegex(ValueError, "under project"):
            self.load()

    def test_tracked_demo_directory_is_accepted_and_cannot_be_escaped(self):
        demo = self.repo / "demo" / "synthetic"
        demo.parent.mkdir()
        self.run.rename(demo)
        self.config["runs"][0]["directory"] = "demo/synthetic"
        _, loaded = self.load()
        self.assertEqual(loaded[0]["id"], "test")
        for directory in ("demo", "viewer/synthetic"):
            self.config["runs"][0]["directory"] = directory
            with self.assertRaisesRegex(ValueError, "under project"):
                self.load()
        (self.repo / "demo" / "escape").symlink_to(self.repo)
        self.config["runs"][0]["directory"] = "demo/escape"
        with self.assertRaisesRegex(ValueError, "under project"):
            self.load()

    def test_training_or_wrong_source_is_never_presented_as_validation(self):
        for field, value in (("partition", "train"), ("source_sha256", "b" * 64)):
            _, loaded = self.load()
            loaded[0]["rows"][self.record.key][0][field] = value
            with self.assertRaisesRegex(ValueError, "validation predictions"):
                self.attach(loaded)

    def test_missing_window_or_round_is_rejected(self):
        _, loaded = self.load()
        payload = copy.deepcopy(self.payload)
        payload["ml"]["rolling"]["windows"].append({"at_seconds": 25})
        with self.assertRaisesRegex(ValueError, "same eligible windows"):
            self.attach(loaded, payload)
        with self.assertRaisesRegex(ValueError, "absent from replay"):
            attach_comparisons({}, {}, self.config, loaded)

    def test_descriptor_and_default_cannot_silently_select_different_experiment(self):
        self.config["runs"][0]["feature_set"] = "all"
        with self.assertRaisesRegex(ValueError, "descriptor"):
            self.load()
        self.config["runs"][0]["feature_set"] = "no_spatial"
        _, loaded = self.load()
        self.config["default_by_map"]["ascent"] = "absent"
        with self.assertRaisesRegex(ValueError, "Default"):
            self.attach(loaded)


if __name__ == "__main__":
    unittest.main()
