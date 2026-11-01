import copy
import unittest

from review_windows import window_evidence
from rolling_features import observe_round
from rolling_inputs import derive_input_policy
from test_data_processing import event, player, record


class ReviewWindowTests(unittest.TestCase):
    def fixture(self):
        r = record()
        r.start_state["map"] = {"bounds": {"min": {"x": 0, "y": 0}, "max": {"x": 100, "y": 100}}}
        r.start_state["teams"][0]["players"][0]["position"] = {"x": 10, "y": 20}
        r.events = [(10, event("player-killed-player", player("A"), player("D"), id="past")),
                    (20, event("player-used-ability", player("A"), {"id": "flash-drive"}, id="boundary")),
                    (30, event("player-killed-player", player("D", 1), player("A"), id="future"))]
        policy = derive_input_policy([observe_round(r)["rows"][:20]])
        return r, policy

    def test_review_withholds_future_events_and_outcome(self):
        r, policy = self.fixture()
        original = window_evidence(r, 20, policy, ["A0"])
        self.assertEqual([e["id"] for e in original["events"]], ["past"])
        self.assertEqual(original["state"]["attacker_alive"], 5)
        self.assertEqual(original["state"]["defender_alive"], 4)
        self.assertEqual(original["focused_player_reports"][0]["position_reports"][0]["position"],
                         {"x": 10, "y": 20})
        changed = copy.deepcopy(r)
        changed.atk_won = 0
        changed.events = r.events[:1] + [(20, event("player-killed-player", player("D", 1),
                                                  player("A"), id="different-boundary")),
                                       (90, event("team-won-round", {"id": "D"},
                                                  {"id": "round-1"}, id="outcome"))]
        self.assertEqual(window_evidence(changed, 20, policy, ["A0"]), original)

    def test_nonzero_constant_observation_is_reported_as_suppressed(self):
        r, policy = self.fixture()
        evidence = window_evidence(r, 20, policy)
        suppressed = {item["name"]: item for item in evidence["input_support"]["nonzero_raw_channels_suppressed"]}
        self.assertEqual(suppressed["atk_alive"]["raw_last"], 1)
        self.assertEqual(suppressed["atk_alive"]["model_value"], 0)
        self.assertNotIn("def_alive", suppressed)
        with self.assertRaisesRegex(ValueError, "eligible"):
            window_evidence(r, 19, policy)


if __name__ == "__main__":
    unittest.main()
