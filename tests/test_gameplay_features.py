import copy
import json
import tempfile
import unittest
from pathlib import Path

from ability_taxonomy import ABILITY_TAGS, ability_tags
from rolling_features import ABILITIES, FEATURE_NAMES, schema, legacy_schema, observe_round, preview_round
from rolling_inputs import derive_input_policy, validate_input_policy, mask_window, selected_channels, support_report
from rolling_experiment import read_manifest, prediction_row, read_predictions
from test_data_processing import event, player
from test_rolling import located_record, rolling_manifest


class GameplayTests(unittest.TestCase):
    def value(self, observed, at, name):
        return observed["rows"][at - 1][FEATURE_NAMES.index(name)]

    def test_map_identity_does_not_change_nonspatial_inputs(self):
        a = located_record()
        for team in a.start_state["teams"]:
            for p in team["players"]:
                p.pop("position")
        b = copy.deepcopy(a)
        b.map_name = "breeze"
        self.assertEqual(observe_round(a)["rows"], observe_round(b)["rows"])
        self.assertFalse(any(n.startswith("map_") or "_ability_" in n for n in FEATURE_NAMES))

    def test_smokes_share_channels_and_multitag_casts_are_counted_once(self):
        a, b = located_record(), located_record()
        a.events = [(1, event("player-used-ability", player("A"), {"id": "cloudburst"}))]
        b.events = [(1, event("player-used-ability", player("A"), {"id": "dark-cover"}))]
        self.assertEqual(observe_round(a)["rows"], observe_round(b)["rows"])
        b.events = [(1, event("player-used-ability", player("A"), {"id": "zero-point"}))]
        observed = observe_round(b)
        for name in ("atk_abilities_1s", "atk_utility_recon_uses", "atk_utility_suppression_uses"):
            self.assertEqual(self.value(observed, 2, name), 1)
        self.assertEqual(set(ABILITY_TAGS), set(ABILITIES) - {"other"})
        self.assertEqual(ability_tags("future-ability"), ("other",))
        preview = preview_round(b)
        self.assertEqual(preview["ability_tags"]["zero-point"], ["recon", "suppression"])
        self.assertEqual(preview["windows"][0]["utility_categories"], [
            {"side": "atk", "category": "recon", "uses": 1},
            {"side": "atk", "category": "suppression", "uses": 1}])

    def test_movement_requires_same_player_and_fresh_post_cast_reports(self):
        r = located_record()
        first, second = player("A"), player("A")
        first["state"]["game"] = {"position": {"x": 0, "y": 0}}
        second["state"]["game"] = {"position": {"x": 300, "y": 500}}
        r.events = [(10.2, event("player-used-ability", first, {"id": "recon-bolt"})),
                    (11.2, event("player-used-ability", second, {"id": "shock-bolt"}))]
        observed = observe_round(r)
        self.assertEqual(self.value(observed, 11, "atk_movement_after_recon_pairs"), 0)
        self.assertEqual(self.value(observed, 12, "atk_movement_after_recon_pairs"), .2)
        self.assertGreater(self.value(observed, 12, "atk_movement_after_recon_mean"), 0)
        self.assertEqual(self.value(observed, 13, "atk_movement_pairs"), 0)  # no new report
        self.assertEqual(preview_round(r)["windows"][0]["movement"]["atk"],
                         {"report_pairs": 1, "after_recon_pairs": 1})
        stale = copy.deepcopy(r)
        stale.events[1] = (14.2, stale.events[1][1])
        self.assertEqual(self.value(observe_round(stale), 15, "atk_movement_pairs"), 0)
        different = copy.deepcopy(r)
        different.events[1][1]["actor"] = player("A", 1)
        different.events[1][1]["actor"]["state"]["game"] = second["state"]["game"]
        self.assertEqual(self.value(observe_round(different), 12, "atk_movement_pairs"), 0)

    def test_v1_replay_uses_frozen_contract_but_new_inference_rejects_it(self):
        r = located_record()
        r.source_sha256 = "a" * 64
        old = legacy_schema()
        self.assertEqual(len(old["feature_names"]), 365)
        self.assertIn("map_ascent", old["feature_names"])
        self.assertEqual(len(observe_round(r, old)["rows"][0]), 365)
        self.assertEqual(preview_round(r, old)["schema_id"], old["id"])
        manifest = rolling_manifest(r)
        manifest["schema"] = old
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "contract"):
                read_manifest(path)
            loaded, digest = read_manifest(path, allow_legacy=True)
            row = prediction_row(r, 20, {n: .5 for n in manifest["models"]}, loaded, digest)
            predictions = Path(directory) / "predictions.jsonl"
            predictions.write_text(json.dumps(row))
            _, public = read_predictions(predictions, path)
            self.assertEqual(public["schema"], old)
            manifest["schema"]["feature_names"][0] = "tampered"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "contract"):
                read_manifest(path, allow_legacy=True)


class InputSupportTests(unittest.TestCase):
    def windows(self):
        rows = [[0.0] * len(FEATURE_NAMES) for _ in range(20)]
        rows[0][0], rows[1][0] = 1.0, 2.0
        return [rows]

    def test_unseen_inputs_cannot_activate_unsupported_channels(self):
        training = self.windows()
        policy = derive_input_policy(training)
        evaluation = copy.deepcopy(training)
        evaluation[0][0][1] = 100
        masked = mask_window(evaluation[0], policy)
        self.assertEqual(masked, training[0])
        self.assertEqual(support_report(evaluation, policy)["constant_training_channels_changed"], [FEATURE_NAMES[1]])
        self.assertFalse(policy["keep"][1])
        self.assertTrue(policy["keep"][0])

    def test_constant_nonzero_values_are_also_masked(self):
        windows = self.windows()
        for row in windows[0]:
            row[1] = 1
        policy = derive_input_policy(windows)
        self.assertFalse(policy["keep"][1])
        self.assertTrue(all(row[1] == 0 for row in mask_window(windows[0], policy)))
        tampered = copy.deepcopy(policy)
        tampered["keep"][1] = True
        with self.assertRaisesRegex(ValueError, "mask"):
            validate_input_policy(tampered)

    def test_policy_is_schema_bound_and_inputs_must_be_finite(self):
        policy = derive_input_policy(self.windows())
        policy["schema_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "policy"):
            validate_input_policy(policy)
        windows = self.windows()
        windows[0][0][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "inputs"):
            derive_input_policy(windows)
        with self.assertRaises(ValueError):
            derive_input_policy([])

    def test_ablation_families_remove_derived_proxies_too(self):
        def names(preset):
            return {n for n, selected in zip(FEATURE_NAMES, selected_channels(preset)) if selected}
        self.assertNotIn("atk_movement_after_recon_mean", names("no_utility"))
        self.assertNotIn("atk_recon_cast_known", names("no_utility"))
        self.assertNotIn("atk_abilities_total", names("no_utility"))
        self.assertNotIn("atk_utility_recon_x", names("no_spatial"))
        self.assertNotIn("atk_kill_origin_unknown", names("no_spatial"))
        self.assertIn("atk_utility_recon_uses", names("no_spatial"))
        self.assertNotIn("atk_trade_candidates_1s", names("no_trades"))
        self.assertEqual(len(names("state_only")), 14)
        self.assertIn("atk_alive", names("state_only"))
        self.assertNotIn("atk_kills_total", names("state_only"))
