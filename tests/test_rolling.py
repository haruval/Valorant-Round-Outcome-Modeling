import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from evaluation import split_groups
from rolling_features import FEATURE_NAMES, observe_round, load_rolling_dataset, schema
from rolling_experiment import metrics, prediction_row, read_predictions, read_manifest, schema_hash, validate_prediction
from replay_data import build_catalog
from test_data_processing import record, player, event, stream_events, write_stream


def located_record():
    r = record()
    r.map_name = "ascent"
    for team in r.start_state["teams"]:
        for p in team["players"]:
            p.update(position={"x": 0, "y": 0}, currentHealth=100, currentArmor=25, loadoutValue=2000)
    return r


def rolling_manifest(r):
    return {"schema_version": "rolling-run-v1", "status": "complete", "purpose": "ui_fixture",
            "run_id": "1" * 64, "mode": "development_smoke_test", "schema": schema(),
            "data": {"sources": [{"sha256": r.source_sha256}]},
            "models": {name: {"artifact": "baselines.json", "artifact_sha256": "b" * 64}
                       for name in ("train_prior", "snapshot_logistic", "history_logistic")},
            "partitions": {"train": [{"round": list(r.key), "source_sha256": r.source_sha256}],
                           "validation": [{"round": [r.series_id, "other-map", "round-1"], "source_sha256": r.source_sha256}],
                           "test": []}}


class RollingTests(unittest.TestCase):
    def column(self, observations, endpoint, name):
        return observations["rows"][endpoint - 1][FEATURE_NAMES.index(name)]

    def test_schedule_and_half_open_observations(self):
        r = located_record()
        r.ended_at = r.live_at + 31
        r.events = [(19.9, event("player-used-ability", player("A"), {"id": "recon-bolt"})),
                    (20, event("player-used-ability", player("A"), {"id": "flash-drive"}))]
        result = observe_round(r)
        self.assertEqual(result["endpoints"], [20, 25, 30])
        self.assertEqual(self.column(result, 20, "atk_ability_recon-bolt_uses"), 1)
        self.assertEqual(self.column(result, 20, "atk_ability_flash-drive_uses"), 0)
        self.assertEqual(self.column(result, 21, "atk_ability_flash-drive_uses"), 1)

    def test_future_locations_abilities_and_winner_do_not_change_prefix(self):
        a = located_record()
        b = copy.deepcopy(a)
        actor = player("A")
        actor["state"]["game"] = {"position": {"x": 300, "y": 500}, "currentHealth": 20}
        b.events = [(20, event("player-used-ability", actor, {"id": "owl-drone"}))]
        b.atk_won = 0
        self.assertEqual(observe_round(a)["rows"][:20], observe_round(b)["rows"][:20])

    def test_stale_positions_are_masked_not_carried_as_current_presence(self):
        result = observe_round(located_record())
        self.assertEqual(self.column(result, 2, "atk_position_fresh"), 1)
        self.assertEqual(self.column(result, 3, "atk_position_fresh"), 0)
        self.assertEqual(self.column(result, 3, "atk_position_unknown"), 0)
        self.assertEqual(sum(self.column(result, 3, f"atk_presence_cell_{i}") for i in range(9)), 0)

    def test_unknown_abilities_and_maps_have_explicit_fallbacks(self):
        r = located_record()
        r.map_name = "new-map"
        r.events = [(1, event("player-used-ability", player("A"), {"id": "future-ability"}))]
        result = observe_round(r)
        self.assertEqual(self.column(result, 2, "atk_ability_other_uses"), 1)
        self.assertEqual(self.column(result, 2, "atk_cast_origin_unknown"), 1)
        self.assertEqual(self.column(result, 2, "map_other"), 1)

    def test_trade_requires_killing_the_prior_killer_within_five_seconds(self):
        r = located_record()
        r.events = [(10, event("player-killed-player", player("D", 0), player("A", 0))),
                    (12, event("player-killed-player", player("A", 1), player("D", 0))),
                    (13, event("player-killed-player", player("D", 1), player("A", 2))),
                    (19, event("player-killed-player", player("A", 3), player("D", 1)))]
        result = observe_round(r)
        self.assertEqual(self.column(result, 13, "atk_trade_candidates_1s"), 1)
        self.assertEqual(self.column(result, 13, "atk_trade_delay_5s"), .4)
        self.assertEqual(self.column(result, 20, "atk_trade_candidates_1s"), 0)

    def test_kill_origin_and_victim_location_remain_distinct(self):
        r = located_record()
        actor, target = player("A"), player("D")
        actor["state"]["game"] = {"position": {"x": 0, "y": 0}}
        target["state"]["game"] = {"position": {"x": 4000, "y": -8000}}
        r.events = [(10, event("player-killed-player", actor, target))]
        result = observe_round(r)
        origin = [self.column(result, 11, f"atk_kill_origin_cell_{i}") for i in range(9)]
        victim = [self.column(result, 11, f"atk_kill_victim_cell_{i}") for i in range(9)]
        self.assertEqual(sum(origin), 1)
        self.assertEqual(sum(victim), 1)
        self.assertNotEqual(origin, victim)

    def test_all_dead_attackers_only_decide_an_unplanted_round(self):
        a = located_record()
        a.events = [(24, event("player-killed-player", player("D"), player("A", n))) for n in range(5)]
        b = copy.deepcopy(a)
        b.events.insert(0, (15, event("player-completed-plantBomb", player("A"))))
        self.assertEqual(observe_round(a)["endpoints"], [20])
        self.assertIn(25, observe_round(b)["endpoints"])

    def test_round_weights_and_group_partitions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            write_stream(path, stream_events())
            dataset = load_rolling_dataset([path])
            self.assertAlmostEqual(sum(dataset.weights), 1)
            self.assertEqual(dataset.endpoints, list(range(20, 60, 5)))
            self.assertEqual(dataset.audit["counts"]["eligible_rounds"], 1)
            groups = ["series1"] * 5 + ["series2"] * 10 + ["series3"] * 2
            parts = split_groups(groups)
            self.assertEqual(sum(len({groups[i] for i in indices}) for indices in parts.values()), 3)
            values = metrics(dataset, list(range(len(dataset.features))), [.5] * len(dataset.features))
            self.assertEqual(values["rounds"], 1)
            self.assertAlmostEqual(values["round_mean_log_loss"], math.log(2))

    def test_rolling_prediction_provenance_and_duplicate_timestamps(self):
        r = located_record()
        r.source_sha256 = "a" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "run.json"
            manifest_path.write_text(json.dumps(rolling_manifest(r)))
            manifest, digest = read_manifest(manifest_path)
            row = prediction_row(r, 20, {name: .5 for name in manifest["models"]}, manifest, digest)
            for name, value in (("at_seconds", 21), ("window_start", 5), ("window_start", False), ("source_sha256", "d" * 64),
                                ("partition", "test"), ("schema_sha256", "e" * 64), ("run_manifest_sha256", "f" * 64)):
                with self.subTest(field=name), self.assertRaises(ValueError):
                    validate_prediction({**row, name: value}, manifest, digest)
            self.assertEqual(row["schema_sha256"], schema_hash(schema()))
            path = root / "predictions.jsonl"
            path.write_text(json.dumps(row) + "\n" + json.dumps(row))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_predictions(path, manifest_path)
            path.write_text(json.dumps(row))
            grouped, public = read_predictions(path, manifest_path)
            self.assertEqual(grouped[r.key][0]["at_seconds"], 20)
            self.assertEqual(public["purpose"], "ui_fixture")

    def test_viewer_rejects_prediction_after_outcome_and_changed_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            path = data / "events.jsonl"
            events = stream_events()
            map_state = {"name": "ascent", "bounds": {"min": {"x": -100, "y": -100}, "max": {"x": 100, "y": 100}}}
            events[0][1]["actor"]["state"]["map"] = map_state
            events[1][1]["seriesState"]["games"][0]["map"] = map_state
            write_stream(path, events)
            dataset = load_rolling_dataset([path])
            r = dataset.records[0]
            manifest_path = root / "run.json"
            manifest_path.write_text(json.dumps(rolling_manifest(r)))
            manifest, digest = read_manifest(manifest_path)
            prediction_path = root / "predictions.jsonl"
            def write_prediction(at):
                prediction_path.write_text(json.dumps(prediction_row(r, at, {n: .5 for n in manifest['models']}, manifest, digest)))
            write_prediction(20)
            catalog, payloads = build_catalog(data, rolling_predictions=prediction_path, rolling_run_manifest=manifest_path)
            self.assertEqual(catalog["prediction_count"], 1)
            self.assertEqual(json.loads(next(iter(payloads.values())))["ml"]["predictions"][0]["at_seconds"], 20)
            write_prediction(60)
            with self.assertRaisesRegex(ValueError, "observation time"):
                build_catalog(data, rolling_predictions=prediction_path, rolling_run_manifest=manifest_path)
            write_prediction(20)
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "source"):
                build_catalog(data, rolling_predictions=prediction_path, rolling_run_manifest=manifest_path)
