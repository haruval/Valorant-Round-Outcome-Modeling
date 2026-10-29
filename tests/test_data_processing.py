import copy
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from data_processing import (
    RoundRecord, extract_rounds, load_dataset, parse_timestamp, round_to_features,
)


def teams():
    return [{"id": team, "side": side, "loadoutValue": 10000,
             "players": [{"id": f"{team}{n}", "alive": True} for n in range(5)]}
            for team, side in (("A", "attacker"), ("D", "defender"))]


def player(team, number=0):
    return {"id": f"{team}{number}", "type": "player", "state": {"teamId": team}}


def record():
    return RoundRecord("series-1", "game-1", "round-1", "test-map", "A", "D",
                       {"teams": teams()}, live_at=100, ended_at=200, atk_won=1)


def event(kind, actor=None, target=None, **extra):
    return {"type": kind, "actor": actor or {}, "target": target or {}, **extra}


def stream_events():
    return [
        (0, event("game-started-round", {"type": "game", "id": "game-1",
                  "state": {"teams": teams(), "map": {"name": "test-map"}}}, {"id": "round-1"})),
        (20, event("round-ended-freezetime", {"id": "round-1"},
                   seriesState={"games": [{"id": "game-1", "teams": teams()}]})),
        (21, event("player-killed-player", player("A"), player("D"))),
        (80, event("team-won-round", {"id": "A"}, {"id": "round-1"})),
    ]


def write_stream(path, entries):
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    with path.open("w", encoding="utf-8") as stream:
        for sequence, (seconds, data) in enumerate(entries, 1):
            stream.write(json.dumps({"seriesId": "series-1", "sequenceNumber": sequence,
                                     "occurredAt": (base + timedelta(seconds=seconds)).isoformat(),
                                     "events": [{"id": f"event-{sequence}", **data}]}) + "\n")


class FeatureTests(unittest.TestCase):
    def test_event_team_id_updates_deaths_kills_and_abilities(self):
        r = record()
        r.events = [(0.2, event("player-killed-player", player("D"), player("A"))),
                    (0.3, event("player-used-ability", player("D")))]
        row = round_to_features(r, 30)[0]
        self.assertEqual(row[1:5], [0.8, 1.0, 0.0, 0.2])
        self.assertEqual(row[9:12], [0.05, 1.0, 0.0])

    def test_future_events_and_winner_cannot_change_features(self):
        a = record()
        b = copy.deepcopy(a)
        b.atk_won = 0
        b.events = [(30.0, event("player-killed-player", player("A"), player("D"))),
                    (50.0, event("player-completed-plantBomb"))]
        self.assertEqual(round_to_features(a, 30), round_to_features(b, 30))

    def test_half_open_bins(self):
        r = record()
        r.events = [(1.0, event("player-used-ability", player("A")))]
        rows = round_to_features(r, 30)
        self.assertEqual(rows[0][8], 0.0)
        self.assertEqual(rows[1][8], 0.05)

    def test_no_kill_differs_from_defender_first_kill(self):
        a = record()
        b = copy.deepcopy(a)
        b.events = [(0.1, event("player-killed-player", player("D"), player("A")))]
        self.assertEqual(round_to_features(a, 30)[0][-2:], [0.0, 0.0])
        self.assertEqual(round_to_features(b, 30)[0][-2:], [1.0, 0.0])

    def test_terminal_at_cutoff_and_pause_are_rejected(self):
        for attribute, value in (("ended_at", 130), ("terminal_at", 125), ("paused", True)):
            with self.subTest(attribute=attribute):
                r = record()
                setattr(r, attribute, value)
                with self.assertRaises(ValueError):
                    round_to_features(r, 30)

    def test_unknown_player_team_fails_instead_of_silently_dropping_features(self):
        r = record()
        r.events = [(0.1, event("player-used-ability", {"id": "unknown"}))]
        with self.assertRaisesRegex(ValueError, "Unknown team"):
            round_to_features(r, 30)

    def test_timestamp_offsets_and_nanoseconds(self):
        self.assertEqual(parse_timestamp("2024-01-01T02:00:00.123456789+02:00"),
                         parse_timestamp("2024-01-01T00:00:00.123456Z"))
        with self.assertRaises(ValueError):
            parse_timestamp("2024-01-01T00:00:00")


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "events.jsonl"

    def test_live_window_and_team_swap_labels(self):
        entries = stream_events()
        start = entries[0][1]["actor"]["state"]["teams"]
        start[0]["side"], start[1]["side"] = "defender", "attacker"
        write_stream(self.path, entries)
        data = load_dataset([self.path], 30)
        self.assertEqual(data.labels, [0])
        self.assertEqual(data.rounds[0].series_id, "series-1")
        self.assertEqual(data.features[0][1][1], 0.8)
        self.assertEqual(data.audit["counts"]["eligible_rounds"], 1)

    def test_buy_time_does_not_consume_observation_window(self):
        write_stream(self.path, stream_events())
        data = load_dataset([self.path], 30)
        self.assertEqual(data.features[0][0][3], 0.0)
        self.assertEqual(data.features[0][1][3], 0.2)

    def test_correction_excludes_previously_completed_game(self):
        entries = stream_events() + [(81, event("game-!started-round",
                                                {"type": "game", "id": "game-1"}))]
        write_stream(self.path, entries)
        records, audit = extract_rounds(self.path)
        self.assertEqual(records, [])
        self.assertEqual(audit["rounds_excluded_for_corrections"], 1)

    def test_missing_live_start_and_incomplete_rounds_are_counted(self):
        entries = stream_events()
        write_stream(self.path, [entries[0], entries[-1]])
        data = load_dataset([self.path])
        self.assertEqual(data.audit["counts"]["missing_live_start"], 1)
        write_stream(self.path, entries[:-1])
        self.assertEqual(extract_rounds(self.path)[1]["unfinished_rounds"], 1)

    def test_duplicate_rounds_across_files_are_rejected(self):
        write_stream(self.path, stream_events())
        second = self.path.with_name("copy.jsonl")
        second.write_bytes(self.path.read_bytes())
        with self.assertRaisesRegex(ValueError, "Duplicate round"):
            load_dataset([self.path, second])

    def test_unknown_winner_and_time_reversal_fail(self):
        entries = stream_events()
        entries[-1][1]["actor"]["id"] = "other-team"
        write_stream(self.path, entries)
        with self.assertRaisesRegex(ValueError, "Winner"):
            extract_rounds(self.path)
        write_stream(self.path, list(reversed(stream_events())))
        with self.assertRaisesRegex(ValueError, "out-of-order"):
            extract_rounds(self.path)


if __name__ == "__main__":
    unittest.main()
