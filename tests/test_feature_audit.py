import unittest

from audit_rolling import check_record, event_ledger
from rolling_features import FEATURE_NAMES, observe_round
from test_data_processing import event, player
from test_rolling import located_record


class FeatureAuditTests(unittest.TestCase):
    def test_independent_event_counts_detect_corrupted_projection(self):
        record = located_record()
        ability = event("player-used-ability", player("A"), {"id": "recon-bolt"})
        ability["id"] = "cast"
        record.events = [(19.9, ability)]
        observed = observe_round(record)
        self.assertEqual(check_record(record, observed)["failures"], [])
        observed["rows"][19][FEATURE_NAMES.index("atk_utility_recon_uses")] = 0
        failed = check_record(record, observed)["failures"]
        self.assertEqual(failed[0]["feature"], "atk_utility_recon_uses")
        self.assertEqual(failed[0]["expected_from_events"], 1)

    def test_five_second_boundary_and_simultaneous_kills(self):
        for delay, expected in ((0, 0), (5, 1), (5.001, 0)):
            with self.subTest(delay=delay):
                record = located_record()
                original = event("player-killed-player", player("D", 0), player("A", 0))
                original["id"] = "original"
                returned = event("player-killed-player", player("A", 1), player("D", 0))
                returned["id"] = "returned"
                record.events = [(10, original), (10 + delay, returned)]
                observed = observe_round(record)
                candidates = sum(row[FEATURE_NAMES.index("atk_trade_candidates_1s")] for row in observed["rows"])
                self.assertEqual(candidates, expected)
                self.assertEqual(check_record(record, observed)["failures"], [])
                self.assertEqual(event_ledger(record)[1]["traded_event_id"], "original" if expected else None)
