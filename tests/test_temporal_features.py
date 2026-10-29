import copy
import unittest

from baseline_features import SUMMARY_FEATURE_NAMES, TEMPORAL_NAMES, temporal_summary
from data_processing import FEATURE_NAMES, round_to_features
from test_data_processing import event, player, record


class TemporalFeatureTests(unittest.TestCase):
    def summarize(self, r):
        return dict(zip(SUMMARY_FEATURE_NAMES, temporal_summary(round_to_features(r, 30))))

    def test_timing_uses_bin_endpoints_and_presence_flags(self):
        r = record()
        r.events = [(2.2, event("player-killed-player", player("A"), player("D"))),
                    (10, event("player-completed-plantBomb")),
                    (27.1, event("player-killed-player", player("D"), player("A")))]
        values = self.summarize(r)
        self.assertEqual(values["first_kill_time_fraction"], 3 / 30)
        self.assertEqual(values["time_since_last_elimination_fraction"], 2 / 30)
        self.assertEqual(values["time_since_plant_fraction"], 19 / 30)
        self.assertEqual(values["elimination_observed"], 1)
        missing = self.summarize(record())
        self.assertTrue(all(missing[name] == 0 for name in TEMPORAL_NAMES))

    def test_recent_counts_include_left_boundary_exclude_cutoff(self):
        r = record()
        r.events = [(24.999, event("player-used-ability", player("A"))),
                    (25, event("player-used-ability", player("A"))),
                    (29.999, event("player-used-ability", player("A"))),
                    (30, event("player-used-ability", player("A")))]
        self.assertAlmostEqual(self.summarize(r)["atk_ability_uses_last_5s"], .1)

    def test_same_snapshot_different_event_history(self):
        early, late = record(), record()
        early.events = [(3, event("player-killed-player", player("A"), player("D")))]
        late.events = [(27, event("player-killed-player", player("A"), player("D")))]
        a, b = [temporal_summary(round_to_features(r, 30)) for r in (early, late)]
        self.assertEqual(a[:len(FEATURE_NAMES)], b[:len(FEATURE_NAMES)])
        self.assertNotEqual(a[len(FEATURE_NAMES):], b[len(FEATURE_NAMES):])

    def test_future_events_and_labels_do_not_change_summary(self):
        a = record()
        b = copy.deepcopy(a)
        b.atk_won = 0
        b.events = [(30, event("player-killed-player", player("A"), player("D"))),
                    (40, event("player-completed-plantBomb"))]
        self.assertEqual(self.summarize(a), self.summarize(b))

    def test_malformed_sequences_are_rejected(self):
        for window in ([], [[0] * 11], [[float("nan")] * 12]):
            with self.subTest(window=window), self.assertRaises(ValueError):
                temporal_summary(window)
