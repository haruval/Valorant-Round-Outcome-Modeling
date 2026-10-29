import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from data_processing import discover_data_files
from replay_data import build_catalog, build_round_replay, merge_player, read_predictions, round_token
from test_data_processing import event, player, record, stream_events, write_stream


class ReplayTests(unittest.TestCase):
    def make_record(self):
        r = record()
        r.start_state["map"] = {"bounds": {"min": {"x": 0, "y": 0}, "max": {"x": 100, "y": 100}}}
        for team in r.start_state["teams"]:
            for p in team["players"]:
                p.update(position={"x": 10, "y": 20}, currentHealth=100,
                         character={"name": "test-agent"}, abilities=[{"id": "one", "ready": True}])
        return r

    def test_future_position_and_death_do_not_change_earlier_frames(self):
        r = self.make_record()
        target = player("D")
        target["state"]["game"] = {"position": {"x": 90, "y": 80}, "alive": False}
        r.events = [(10, event("player-killed-player", player("A"), target))]
        replay = build_round_replay(r)
        initial = next(p for p in replay["frames"][0]["players"] if p["id"] == "D0")
        later = next(p for p in replay["frames"][-1]["players"] if p["id"] == "D0")
        self.assertTrue(initial["alive"])
        self.assertEqual(initial["position"], {"x": 10, "y": 20})
        self.assertFalse(later["alive"])
        self.assertEqual(later["position_at"], 10)
        self.assertEqual(later["position"], {"x": 90, "y": 80})

    def test_partial_player_updates_preserve_health_and_other_abilities(self):
        p = {"currentHealth": 85, "abilities": {"one": {"ready": True}, "two": {"ready": True}}}
        merge_player(p, {"abilities": [{"id": "one", "ready": False}]}, 4)
        self.assertEqual(p["currentHealth"], 85)
        self.assertFalse(p["abilities"]["one"]["ready"])
        self.assertTrue(p["abilities"]["two"]["ready"])

    def test_ability_marker_uses_caster_position_at_cast(self):
        r = self.make_record()
        actor = player("A")
        actor["state"]["game"] = {"position": {"x": 40, "y": 60}}
        r.events = [(8, event("player-used-ability", actor, {"id": "test-cast"}))]
        replay = build_round_replay(r)
        self.assertEqual(replay["events"][0]["position"], {"x": 40, "y": 60})
        self.assertEqual(replay["events"][0]["category"], "ability")
        self.assertIsNone(replay["ml"]["prediction"])

    def test_early_round_still_replays_without_eligible_features(self):
        r = self.make_record()
        r.ended_at = 125
        replay = build_round_replay(r)
        self.assertEqual(replay["duration"], 25)
        self.assertEqual(replay["ml"]["features"], [])
        with self.assertRaisesRegex(ValueError, "after round outcome"):
            build_round_replay(r, {"observation_seconds": 30, "p_attacker_win": .6})

    def test_simultaneous_updates_share_one_complete_frame(self):
        r = self.make_record()
        r.events = [(10, event("player-used-ability", player("A"), {"id": "one"})),
                    (10, event("player-used-ability", player("D"), {"id": "two"}))]
        replay = build_round_replay(r)
        self.assertEqual(len(replay["frames"]), 2)
        self.assertEqual(replay["frames"][1]["ability_uses"], {"attacker": 1, "defender": 1})

    def test_nested_exports_and_external_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            (data / "series-1").mkdir(parents=True)
            (data / "series-2").mkdir()
            a, b = data / "series-1/a.jsonl", data / "series-2/b.jsonl"
            a.write_text("")
            b.write_text("")
            self.assertEqual(discover_data_files(data), [a.resolve(), b.resolve()])
            outside = root / "outside.jsonl"
            outside.write_text("")
            (data / "external.jsonl").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "symlinks"):
                discover_data_files(data)

    def test_round_identifier_is_stable_and_distinguishes_series(self):
        self.assertEqual(round_token(("s", "g", "r")), round_token(("s", "g", "r")))
        self.assertNotEqual(round_token(("s", "g", "r")), round_token(("s2", "g", "r")))

    def test_catalog_accepts_additional_series_in_nested_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = stream_events()
            map_state = {"name": "test-map", "bounds": {"min": {"x": 0, "y": 0}, "max": {"x": 100, "y": 100}}}
            entries[0][1]["actor"]["state"]["map"] = map_state
            entries[1][1]["seriesState"]["games"][0]["map"] = map_state
            for index in (1, 2):
                directory = root / f"series-{index}"
                directory.mkdir()
                path = directory / "events.jsonl"
                write_stream(path, entries)
                if index == 2:
                    path.write_text(path.read_text().replace('"series-1"', '"series-2"'))
            catalog, payloads = build_catalog(root)
            self.assertEqual(catalog["series_count"], 2)
            self.assertEqual(catalog["map_count"], 2)
            self.assertEqual(len(payloads), 2)
            self.assertEqual(len(catalog["sources"]), 2)
            self.assertEqual({s["id"] for s in catalog["series"]}, {"series-1", "series-2"})
            self.assertTrue(all(s["tournament"] is None for s in catalog["series"]))

            source_hash = hashlib.sha256((root / "series-1/events.jsonl").read_bytes()).hexdigest()
            metadata = {"series-1": {"source_sha256": source_hash, "tournament": "Example Cup",
                                      "date": "2024-02-16", "timezone": "America/Los_Angeles"}}
            (root / "series.json").write_text(json.dumps(metadata))
            catalog, _ = build_catalog(root)
            one, two = catalog["series"]
            self.assertEqual(one["tournament"], "Example Cup")
            self.assertEqual(one["date"], "2024-02-16")
            self.assertIsNone(two["tournament"])
            metadata["series-1"]["source_sha256"] = "wrong-source"
            (root / "series.json").write_text(json.dumps(metadata))
            catalog, _ = build_catalog(root)
            self.assertTrue(all(s["tournament"] is None for s in catalog["series"]))

    def test_outcome_is_absent_from_initial_replay_state(self):
        r = self.make_record()
        r.events = [(100, event("team-won-round", {"id": "A"}, {"id": "round-1"}))]
        replay = build_round_replay(r)
        self.assertIsNone(replay["frames"][0]["winner"])
        self.assertEqual(replay["frames"][-1]["winner"], "attacker")


if __name__ == "__main__":
    unittest.main()
