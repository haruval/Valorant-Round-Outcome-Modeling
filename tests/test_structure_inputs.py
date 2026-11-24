import copy
import unittest

from rolling_experiment import baseline_probabilities
from rolling_features import FEATURE_NAMES, observe_round, schema
from rolling_inputs import derive_input_policy
from structure_inputs import (ARMS, EXTENSION_NAMES, derive_extension_policy, extended_baseline_probabilities,
                              extension_contract, extension_support, extension_windows, living_attacker,
                              validate_baselines, validate_extension_block, validate_extension_policy)
from test_rolling import located_record
from test_utility_structure import cast
from utility_structure import FAMILIES


def fixture(events):
    r = located_record()
    r.events = sorted(events, key=lambda item: item[0])
    observed = observe_round(r)
    return r, observed


def column(window, name):
    return [row[EXTENSION_NAMES.index(name)] for row in window]


def coefficients(width):
    return {"mean": [.1] * width, "scale": [1 + i % 3 for i in range(width)],
            "coefficients": [((i * 37) % 11 - 5) / 10 for i in range(width)], "intercept": .25}


class StructureInputTests(unittest.TestCase):
    def test_extension_rows_align_with_v2_rows_in_every_window(self):
        r, observed = fixture([cast(10.2, "paranoia"), cast(24.5, "dark-cover", number=1)])
        endpoints = observed["endpoints"][:3]
        self.assertEqual(endpoints, [20, 25, 30])
        windows = extension_windows([r] * 3, endpoints)
        casts = FEATURE_NAMES.index("atk_abilities_1s")
        for at, window in zip(endpoints, windows):
            self.assertEqual((len(window), len(window[0])), (20, len(EXTENSION_NAMES)))
            first_cast = next(t for t, row in enumerate(observed["rows"][at - 20:at]) if row[casts])
            first_live = next(t for t, value in enumerate(column(window, "atk_combo_casters")) if value)
            # The second in which v2 counts the cast is the second its live span starts.
            self.assertEqual(first_live, first_cast, at)
        # Cast at 10.2s with a 3-second span: live through 13.2s, so it still shows in bin 13.
        self.assertEqual(column(windows[0], "atk_combo_casters")[9:15], [0, .2, .2, .2, .2, 0])
        self.assertEqual(column(windows[1], "atk_combo_casters")[4:10], [0, .2, .2, .2, .2, 0])
        self.assertEqual(column(windows[1], "atk_combo_casters")[19], .2)

    def test_placebo_counts_casts_where_structure_counts_players(self):
        one, observed = fixture([cast(10, "flash-drive", number=1), cast(10.2, "paranoia", number=1),
                                 cast(10.4, "recon-bolt", number=1)])
        window, = extension_windows([one], [20])
        self.assertEqual(column(window, "atk_combo_casters")[10], .2)
        self.assertEqual(column(window, "atk_support_casts_live")[10], .6)
        three, _ = fixture([cast(10, "flash-drive", number=1), cast(10.2, "paranoia", number=2),
                            cast(10.4, "recon-bolt", number=3)])
        window, = extension_windows([three], [20])
        self.assertEqual(column(window, "atk_combo_casters")[10], .6)
        self.assertEqual(column(window, "atk_support_casts_live")[10], .6)
        self.assertEqual(column(window, "def_support_casts_live"), [0] * 20)

    def test_future_events_and_outcome_cannot_change_extension_windows(self):
        r, _ = fixture([cast(10, "tailwind"), cast(10.5, "flash-drive", number=1)])
        expected = extension_windows([r], [20])
        changed = copy.deepcopy(r)
        changed.atk_won = 0
        changed.events.append(cast(20, "recon-bolt", number=2))
        self.assertEqual(extension_windows([changed, changed], [20, 25])[0], expected[0])

    def test_arms_are_disjoint_controls_over_named_channels(self):
        self.assertEqual(ARMS["base"], [])
        self.assertEqual(ARMS["structure"], FAMILIES["structure"])
        self.assertEqual(len(ARMS["structure"]), 6)
        self.assertFalse(set(ARMS["structure"]) & set(ARMS["placebo"]))
        self.assertTrue(all(set(names) <= set(EXTENSION_NAMES) for names in ARMS.values()))
        self.assertEqual(len(set(EXTENSION_NAMES)), len(EXTENSION_NAMES))
        self.assertFalse(any("id" in n or "map" in n or n.endswith(("_x", "_y")) for n in EXTENSION_NAMES))
        self.assertEqual(extension_contract()["feature_names"], EXTENSION_NAMES)

    def test_policy_keeps_only_arm_channels_that_vary_in_training(self):
        r, _ = fixture([cast(10, "tailwind"), cast(10.5, "flash-drive", number=1)])
        train = extension_windows([r], [20])
        policy = derive_extension_policy(train, "structure")
        kept = [n for n, keep in zip(EXTENSION_NAMES, policy["keep"]) if keep]
        self.assertEqual(kept, ["atk_combo_casters", "atk_combo_functions", "atk_supported_entry",
                                "atk_entry_support_casters"])  # Defender channels are constant here.
        self.assertFalse(any(derive_extension_policy(train, "base")["keep"]))
        validate_extension_policy(policy)
        other, _ = fixture([cast(10, "paranoia", team="D")])
        report = extension_support(extension_windows([other], [20]), policy)
        self.assertIn("def_combo_casters", report["constant_training_channels_changed"])
        for change in (lambda p: p["keep"].__setitem__(EXTENSION_NAMES.index("def_combo_casters"), True),
                       lambda p: p.update(arm="placebo"), lambda p: p.update(extension_sha256="0" * 64),
                       lambda p: p["minimum"].pop()):
            bad = copy.deepcopy(policy)
            change(bad)
            with self.assertRaises(ValueError):
                validate_extension_policy(bad)
        with self.assertRaises(ValueError):
            derive_extension_policy(train, "unknown")

    def bundles(self, arm):
        r, observed = fixture([cast(10, "tailwind"), cast(10.5, "flash-drive", number=1),
                               cast(12, "paranoia", team="D")])
        window, extension = observed["rows"][:20], extension_windows([r], [20])[0]
        policy = derive_input_policy([window], "no_spatial")
        extra = derive_extension_policy([extension], arm)
        width = len(FEATURE_NAMES) + sum(extra["keep"])
        frozen = {"schema": schema(), "input_policy": policy, "train_prior": .5,
                  "models": {"snapshot_logistic": coefficients(len(FEATURE_NAMES)),
                             "history_logistic": coefficients(2 * len(FEATURE_NAMES))}}
        extended = {**frozen, "extension_policy": extra,
                    "models": {"snapshot_logistic": coefficients(width), "history_logistic": coefficients(2 * width)}}
        return window, extension, frozen, extended

    def test_base_arm_baselines_equal_the_frozen_baselines_exactly(self):
        window, extension, frozen, extended = self.bundles("base")
        self.assertEqual(extended_baseline_probabilities(extended, [window], [extension]),
                         baseline_probabilities(frozen, [window]))

    def test_only_kept_extension_columns_reach_the_baselines(self):
        window, extension, frozen, extended = self.bundles("structure")
        validate_baselines(extended)
        before = extended_baseline_probabilities(extended, [window], [extension])
        self.assertNotEqual(before, baseline_probabilities(frozen, [window]))
        masked = copy.deepcopy(extension)
        for row in masked:
            row[EXTENSION_NAMES.index("atk_support_casts_live")] += 1000
        self.assertEqual(extended_baseline_probabilities(extended, [window], [masked]), before)
        kept = copy.deepcopy(extension)
        kept[-1][EXTENSION_NAMES.index("atk_combo_casters")] += 1
        self.assertNotEqual(extended_baseline_probabilities(extended, [window], [kept]), before)
        short = copy.deepcopy(extended)
        short["models"]["snapshot_logistic"]["coefficients"].pop()
        with self.assertRaises(ValueError):
            validate_baselines(short)

    def test_living_attacker_stratum_uses_the_final_bin(self):
        alive = FEATURE_NAMES.index("atk_alive")
        rows = lambda value: [[0.0] * len(FEATURE_NAMES) for _ in range(19)] + [[value if i == alive else 0.0
                                                                                 for i in range(len(FEATURE_NAMES))]]
        self.assertEqual(living_attacker([rows(.2), rows(0.0), rows(1.0)], [0, 1, 2]), [0, 2])

    def test_run_manifest_extension_block_is_checked(self):
        r, _ = fixture([cast(10, "tailwind"), cast(10.5, "flash-drive", number=1)])
        policy = derive_extension_policy(extension_windows([r], [20]), "structure")
        block = {"contract": extension_contract(), "policy": policy, "arm": "structure", "architecture": "x"}
        self.assertIs(validate_extension_block({"extension": block}), block)
        for change in (lambda b: b.update(arm="placebo"), lambda b: b.pop("architecture"),
                       lambda b: b["contract"].update(id="other")):
            bad = copy.deepcopy(block)
            change(bad)
            with self.assertRaises(ValueError):
                validate_extension_block({"extension": bad})
        with self.assertRaises(ValueError):
            validate_extension_block({})


if __name__ == "__main__":
    unittest.main()
