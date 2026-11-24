import copy
import json
import unittest
from pathlib import Path

from test_data_processing import event, player, record
from utility_structure import FEATURE_NAMES, check_annotations, extract_casts, preview_structure


def cast(at, ability, team="A", number=0, point=(0, 0)):
    actor = player(team, number)
    actor["state"]["game"] = {"position": dict(zip(("x", "y"), point)) if point is not None else None}
    return at, event("player-used-ability", actor, {"id": ability}, id=f"{team}{number}-{ability}-{at}")


def rows(events, at=20):
    r = record()
    r.events = sorted(events, key=lambda item: item[0])
    return preview_structure(r, [at])["windows"][0]["rows"]


def column(table, name):
    return [row[FEATURE_NAMES.index(name)] for row in table]


class UtilityStructureTests(unittest.TestCase):
    def test_counts_distinct_casters_not_cast_volume(self):
        one = rows([cast(10, "flash-drive", number=1), cast(10.2, "paranoia", number=1),
                    cast(10.4, "recon-bolt", number=1)])
        three = rows([cast(10, "flash-drive", number=1), cast(10.2, "paranoia", number=2),
                      cast(10.4, "recon-bolt", number=3)])
        self.assertEqual(column(one, "atk_combo_casters")[10], .2)
        self.assertEqual(column(three, "atk_combo_casters")[10], .6)
        # Flash and paranoia share one function; the dart adds a second.
        self.assertEqual(column(one, "atk_combo_functions")[10], .5)
        self.assertEqual(column(three, "def_combo_casters"), [0] * 20)

    def test_spans_follow_the_owner_decision_and_are_half_open(self):
        # Owner, 2026-11-23: flashes and recon count for 3 seconds, smokes and mollies for 5.
        for ability, bins in (("paranoia", [10, 11, 12]), ("recon-bolt", [10, 11, 12]),
                              ("dark-cover", [10, 11, 12, 13, 14]), ("nanoswarm", [10, 11, 12, 13, 14])):
            values = column(rows([cast(10, ability)]), "atk_combo_casters")
            self.assertEqual([t for t, v in enumerate(values) if v], bins, ability)

    def test_casts_within_a_span_meet_whatever_the_second_boundaries(self):
        # 2.9s apart, but no whole-second three-second window holds both.
        table = rows([cast(10.9, "flash-drive", number=1), cast(13.8, "paranoia", number=2)])
        self.assertEqual(column(table, "atk_combo_casters")[13], .4)
        apart = rows([cast(10.9, "flash-drive", number=1), cast(13.95, "paranoia", number=2)])
        self.assertEqual(max(column(apart, "atk_combo_casters")), .2)

    def test_supported_entry_needs_another_attacker_before_the_plant(self):
        alone = rows([cast(10, "tailwind"), cast(10.5, "cloudburst")])
        self.assertEqual(column(alone, "atk_supported_entry"), [0] * 20)
        helped = rows([cast(10, "tailwind"), cast(10.5, "flash-drive", number=1), cast(11, "recon-bolt", number=2)])
        # Shown for five seconds after the entry cast, growing only as supporting casts arrive.
        self.assertEqual(column(helped, "atk_supported_entry")[9:16], [0, 1, 1, 1, 1, 1, 0])
        self.assertEqual(column(helped, "atk_entry_support_casters")[10:15], [.25, .5, .5, .5, .5])
        plant = (9, event("player-completed-plantBomb", player("A", 3)))
        planted = rows([plant, cast(10, "tailwind"), cast(10.5, "flash-drive", number=1)])
        self.assertEqual(column(planted, "atk_supported_entry"), [0] * 20)
        self.assertEqual(column(planted, "atk_combo_casters")[10], .2)

    def test_entry_markers_are_tailwind_and_updraft_only(self):
        # Owner, 2026-11-23: Omen's teleports do not mark an entry.
        for ability, expected in (("tailwind", 1), ("updraft", 1), ("shrouded-step", 0), ("from-the-shadows", 0)):
            table = rows([cast(10, ability), cast(10.5, "flash-drive", number=1)])
            self.assertEqual(max(column(table, "atk_supported_entry")), expected, ability)

    def test_entry_support_uses_each_function_span_on_exact_times(self):
        # The owner-confirmed Ascent R13 pattern: a molly 2.846s before the Tailwind event.
        table = rows([cast(12.27, "nanoswarm", number=1), cast(15.116, "tailwind")])
        self.assertEqual(column(table, "atk_supported_entry")[14:20], [0, 1, 1, 1, 1, 1])
        for ability, gap, expected in (("flash-drive", 2.9, 1), ("flash-drive", 3.1, 0),
                                       ("dark-cover", 4.9, 1), ("dark-cover", 5.1, 0)):
            before = rows([cast(10, ability, number=1), cast(10 + gap, "tailwind")])
            after = rows([cast(10, "tailwind"), cast(10 + gap, ability, number=1)])
            self.assertEqual(max(column(before, "atk_supported_entry")), expected, (ability, gap))
            self.assertEqual(max(column(after, "atk_supported_entry")), expected, (ability, gap))

    def test_defensive_mobility_is_not_an_entry(self):
        # Owner, Ascent R1: a defender's dash beside allied utility is a hold, not a supported entry.
        table = rows([cast(10, "tailwind", team="D"), cast(10.5, "paranoia", team="D", number=1)])
        self.assertEqual(column(table, "atk_supported_entry"), [0] * 20)
        self.assertFalse(any(name.startswith("def_") and "entry" in name for name in FEATURE_NAMES))
        self.assertFalse(any(name.startswith("atk_") and "answer" in name for name in FEATURE_NAMES))

    def test_answer_requires_an_earlier_opposing_vision_block_within_ten_seconds(self):
        reply = cast(12, "paranoia", team="D")
        for smoke_at, expected in ((5, .2), (2, .2), (1.9, 0), (12, 0), (12.5, 0)):
            table = rows([cast(smoke_at, "dark-cover"), reply])
            self.assertEqual(column(table, "def_answer_casters")[12], expected, smoke_at)
        self.assertEqual(column(rows([reply]), "def_answer_casters"), [0] * 20)
        own = rows([cast(5, "dark-cover", team="D", number=1), reply])
        self.assertEqual(column(own, "def_answer_casters"), [0] * 20)

    def test_setup_deployables_and_mobility_are_not_combination_utility(self):
        table = rows([cast(10, "turret"), cast(10.2, "trapwire", number=1), cast(10.4, "updraft", number=2)])
        self.assertEqual(column(table, "atk_combo_casters"), [0] * 20)

    def test_caster_spacing_uses_fresh_positions_and_keeps_unlocated_casters(self):
        table = rows([cast(10, "paranoia", point=(0, 0)), cast(10.5, "recon-bolt", number=1, point=(3000, 4000))])
        self.assertEqual(column(table, "atk_combo_spacing")[10], .5)
        self.assertEqual(column(table, "atk_combo_spacing_known")[10], 1)
        table = rows([cast(10, "paranoia"), cast(10.5, "recon-bolt", number=1, point=None)])
        self.assertEqual(column(table, "atk_combo_casters")[10], .4)
        self.assertEqual(column(table, "atk_combo_spacing_known"), [0] * 20)

    def test_future_events_and_outcome_cannot_change_a_window(self):
        r = record()
        r.events = [cast(5, "dark-cover"), cast(10, "tailwind"), cast(10.5, "flash-drive", number=1),
                    cast(12, "paranoia", team="D")]
        expected = preview_structure(r, [20])
        changed = copy.deepcopy(r)
        changed.atk_won = 0
        changed.events += [cast(20, "recon-bolt", number=2), (21, event("team-won-round", {"id": "D"}))]
        self.assertEqual(preview_structure(changed, [20]), expected)
        self.assertEqual(preview_structure(changed, [20, 25])["windows"][0], expected["windows"][0])

    def test_identity_map_and_absolute_location_do_not_enter_channels(self):
        events = [cast(10, "tailwind", point=(30, 40)), cast(10.5, "flash-drive", number=1, point=(0, 0)),
                  cast(11, "recon-bolt", number=2, point=(90, 90)), cast(12, "paranoia", team="D", point=(5, 5))]
        r = record()
        r.events = events
        before = preview_structure(r, [20])["windows"][0]["rows"]
        moved = copy.deepcopy(r)
        moved.map_name = "different-map"
        for _, e in moved.events:
            p = e["actor"]["state"]["game"]["position"]
            p["x"], p["y"] = 1000 - p["y"], 2000 + p["x"]
        self.assertEqual(preview_structure(moved, [20])["windows"][0]["rows"], before)
        self.assertFalse(any("id" in n or "map" in n or n.endswith(("_x", "_y")) for n in FEATURE_NAMES))

    def test_owner_assessed_casts_must_match_recorded_casts(self):
        r = record()
        r.events = [cast(10, "paranoia", number=1)]
        found, = extract_casts(r, 20)
        item = {"event_id": found["event_id"], "ability": "paranoia", "caster_id": "A1", "at_cast_seconds": 10}
        annotations = {"annotations": [{"round": list(r.key), "event_assessments": [item]}]}
        self.assertEqual(check_annotations([r], annotations), 1)
        for key, wrong in (("ability", "flash-drive"), ("caster_id", "A2"), ("at_cast_seconds", 10.5),
                           ("event_id", "missing")):
            bad = copy.deepcopy(annotations)
            bad["annotations"][0]["event_assessments"][0][key] = wrong
            with self.assertRaises(ValueError):
                check_annotations([r], bad)

    def test_saved_owner_verdicts_quote_the_owner(self):
        path = Path(__file__).resolve().parents[1] / "docs" / "gameplay-annotations.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        keys = {(*a["round"], a["at_seconds"]) for a in saved["annotations"]}
        self.assertEqual(len(keys), len(saved["annotations"]))
        for annotation in saved["annotations"]:
            quotes = {r["owner_quote"] for r in annotation["owner_responses"]}
            verdict = annotation["window_assessment"]
            # A recorded verdict must carry one of the owner's own answers as its reason.
            self.assertIn(verdict["reason"], quotes)
            self.assertIn(verdict["useful_support"], (True, False, None))
            for item in annotation.get("event_assessments", []):
                self.assertTrue(item["owner_quote"] and item["evidence_source"])


if __name__ == "__main__":
    unittest.main()
