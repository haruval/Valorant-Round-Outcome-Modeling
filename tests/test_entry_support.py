import copy
import unittest

from entry_support import FEATURE_NAMES, extract_relations, preview_entry_support, window_evidence
from test_data_processing import event, player, record


def report(at, team="A", number=0, point=(0, 0), ability=None):
    actor = player(team, number)
    actor["state"]["game"] = {"position": dict(zip(("x", "y"), point)) if point is not None else None}
    return at, event("player-used-ability" if ability else "player-update", actor,
                     {"id": ability} if ability else {}, id=f"{team}{number}-{at}")


class EntrySupportTests(unittest.TestCase):
    def fixture(self):
        r = record()
        r.events = [report(10, number=1, point=(30, 40), ability="flash-drive"),
                    report(11, point=(0, 0), ability="tailwind"),
                    report(11.5, number=2, point=(90, 90), ability="paranoia"),
                    report(12, point=(3, 4), ability="cloudburst")]
        return r

    def moving(self, r):
        return [p for p in extract_relations(r, 20)["pairs"] if p["displacement"] > 0]

    def test_player_join_separates_preceding_overlapping_self_and_same_time(self):
        pair, = self.moving(self.fixture())
        self.assertEqual(pair["player_id"], "A0")
        self.assertEqual(pair["displacement"], 5)
        self.assertEqual([(c["ability"], c["relation"], c["timing"]) for c in pair["casts"]],
                         [("flash-drive", "ally", "preceding"), ("tailwind", "self", "preceding"),
                          ("paranoia", "ally", "overlapping")])
        self.assertEqual(pair["casts"][0]["spacing_at_first_report"], 50)
        self.assertIsNone(pair["casts"][2]["spacing_at_first_report"])
        rows = preview_entry_support(self.fixture(), [20])["windows"][0]["rows"]
        self.assertEqual(rows[12][FEATURE_NAMES.index("atk_preceding_ally_pairs")], 1)
        self.assertEqual(rows[12][FEATURE_NAMES.index("atk_overlapping_ally_pairs")], 1)

    def test_future_endpoint_and_outcome_cannot_change_features(self):
        r = self.fixture()
        expected = preview_entry_support(r, [20])
        changed = copy.deepcopy(r)
        changed.atk_won = 0
        changed.events += [report(20, point=(999, 999), ability="recon-bolt"),
                           (21, event("team-won-round", {"id": "D"}))]
        self.assertEqual(preview_entry_support(changed, [20]), expected)
        full = extract_relations(changed, 30)
        self.assertEqual(window_evidence(full, 20), expected["windows"][0])

    def test_report_gaps_missing_positions_and_death_do_not_form_movement(self):
        for middle in (report(11.5, point=None),
                       (11.5, event("player-killed-player", player("D"), player("A")))):
            r = record()
            r.events = [report(11), middle, report(12, point=(3, 4))]
            self.assertEqual(self.moving(r), [])
        r = record()
        r.events = [report(5), report(10, point=(3, 4))]
        self.assertEqual(self.moving(r), [])
        r.events = [report(10, number=1), report(11, point=(3, 4))]
        self.assertEqual(self.moving(r), [])

    def test_opponent_and_old_casts_are_excluded(self):
        r = record()
        r.events = [report(5, number=1, ability="flash-drive"),
                    report(10, team="D", ability="paranoia"), report(11), report(12, point=(3, 4))]
        pair, = self.moving(r)
        self.assertEqual(pair["casts"], [])

    def test_equal_timestamp_reports_are_merged_without_fake_movement(self):
        r = record()
        r.events = [report(10), report(10, point=(9, 12)), report(11, point=(12, 16))]
        pair, = self.moving(r)
        self.assertEqual(pair["from"], [9, 12])
        self.assertEqual(pair["displacement"], 5)

    def test_absolute_location_identity_and_map_do_not_enter_numeric_channels(self):
        r = self.fixture()
        a = preview_entry_support(r, [20])["windows"][0]["rows"]
        r.map_name = "different-map"
        for _, e in r.events:
            p = e["actor"]["state"]["game"]["position"]
            p["x"], p["y"] = 1000 - p["y"], 2000 + p["x"]
        self.assertEqual(preview_entry_support(r, [20])["windows"][0]["rows"], a)
        self.assertFalse(any("id" in n or "map" in n or n.endswith(("_x", "_y")) for n in FEATURE_NAMES))

    def test_window_does_not_bridge_hidden_history(self):
        r = record()
        r.events = [report(4, number=1, ability="flash-drive"), report(4.5),
                    report(5.5, point=(3, 4)), report(6, point=(6, 8))]
        window = preview_entry_support(r, [25])["windows"][0]
        pair, = window["pairs"]
        self.assertEqual(pair["start_t"], 5.5)
        self.assertEqual(pair["casts"], [])

    def test_dead_caster_spacing_is_unknown_but_the_cast_is_retained(self):
        r = self.fixture()
        r.events.insert(1, (10.5, event("player-killed-player", player("D"), player("A", 1))))
        pair, = self.moving(r)
        self.assertIsNone(pair["casts"][0]["spacing_at_first_report"])
        self.assertEqual(pair["casts"][0]["ability"], "flash-drive")


if __name__ == "__main__":
    unittest.main()
