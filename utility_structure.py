"""Utility-structure channels (v1) from a hand review of replay cases; standard library only.

Six of the twelve channels (the `structure` family) were trained in the 36-run
experiment in docs/experiment-results.md. The answer and caster-spacing channels
have not been trained.

Revises `entry-support-prototype-v0` after the owner's seven-case review and design
decisions recorded in docs/gameplay-annotations.json. Report-pair links are replaced
by cast-to-cast structure: every cast is a timestamped event, so these channels do
not depend on how often the feed happens to report positions.

A cast is a recorded use. The feed gives the caster's position only: no landing
location, direction, duration, blind, reveal or damage. Player identifiers
separate casters inside a round and never enter the numeric channels.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path

from ability_taxonomy import ability_tags
from data_processing import discover_data_files, extract_rounds, game_state
from entry_support import position
from rolling_features import HISTORY_SECONDS, observe_round

# Owner decisions, 2026-11-23. Flashes and recon count for the 3 seconds the owner described in
# Ascent R13; smokes and mollies count for 5. The four functions are the ones named in that execute.
SPAN = {"vision_block": 5.0, "damage": 5.0, "vision_impair": 3.0, "recon": 3.0}
SUPPORT_TAGS = tuple(SPAN)
MAX_SPAN = max(SPAN.values())
# Owner decision: Tailwind and Updraft mark an entry; Omen's teleports do not. In this series both
# abilities belong to one player per team. The owner reports that the Tailwind event is the
# activation, so the dash follows it by a hold time the feed does not record.
ENTRY_ABILITIES = ("tailwind", "updraft")
ANSWER_LOOKBACK = 10.0  # Owner decision: keep at 10s. Answers in Ascent R1 fell 3.6-8.2s after the smoke.
POSITION_MAX_AGE = 2.0  # Same experimental freshness limit as the v2 contract.
SPACING_SCALE = 10000.0  # Fixed feed-unit divisor shared by every map; not a per-map minimap scale.
SETUP_TAG = "detection_deployable"  # Owner decision: sensors and traps stay out of combinations.
SHARED = ("combo_casters", "combo_functions", "combo_spacing", "combo_spacing_known")
SIDE_FIELDS = {"atk": SHARED + ("supported_entry", "entry_support_casters"),
               "def": SHARED + ("answer_casters", "answer_functions")}
# Each group peaks together; its first channel picks the instant.
GROUPS = {"atk_combo": tuple(f"atk_{field}" for field in SHARED),
          "def_combo": tuple(f"def_{field}" for field in SHARED),
          "atk_supported_entry": ("atk_entry_support_casters", "atk_supported_entry"),
          "def_answer": ("def_answer_casters", "def_answer_functions")}
FEATURE_NAMES = [f"{side}_{field}" for side in ("atk", "def") for field in SIDE_FIELDS[side]]
FAMILIES = {"structure": [n for n in FEATURE_NAMES if "_spacing" not in n and "_answer_" not in n],
            "caster_spacing_proxy": [n for n in FEATURE_NAMES if "_spacing" in n],
            "answer_after_opposing_block": [n for n in FEATURE_NAMES if "_answer_" in n]}


def contract():
    # Frozen: this dictionary is hashed into the completed runs, so its text is not updated.
    # The status string below was written before that experiment was approved and run.
    return {"id": "utility-structure-prototype-v1", "status": "untrained_experiment_not_approved",
            "revises": "entry-support-prototype-v0", "bin_seconds": 1,
            "owner_decisions": "docs/gameplay-annotations.json design_decisions, 2026-11-23",
            "evaluation": "Peak within each second, taken at the bin start and at each cast time in the bin",
            "span_seconds": dict(SPAN), "answer_lookback_seconds": ANSWER_LOOKBACK,
            "position_max_age_seconds": POSITION_MAX_AGE, "spacing_divisor_feed_units": SPACING_SCALE,
            "support_functions": list(SUPPORT_TAGS), "entry_abilities": list(ENTRY_ABILITIES),
            "entry_time": "Cast event time. For Tailwind the owner reports this is the activation, not the dash",
            "excluded_setup_function": SETUP_TAG, "feature_names": FEATURE_NAMES,
            "families": FAMILIES, "identity_inputs": False, "absolute_coordinates": False,
            "side_asymmetry": "Supported entry is attacker-only and pre-plant; answers are defender-only",
            "semantics": "Recorded casts by distinct players; no verified effect, landing spot or entry"}


def support_tags(cast):
    return () if SETUP_TAG in cast["tags"] else tuple(t for t in SUPPORT_TAGS if t in cast["tags"])


def span(cast):
    return max(SPAN[tag] for tag in support_tags(cast))


def extract_casts(record, until):
    """Read only events before until. Caster position is the caster's own fresh report."""
    if not until > 0:  # Rejects NaN too. An infinite endpoint reads the whole round for annotation checks.
        raise ValueError("Positive endpoint required")
    players, last, casts, planted_at = {}, {}, [], None
    for team in record.start_state["teams"]:
        team_id = str(team["id"])
        if team_id in (record.attacker_team_id, record.defender_team_id):
            for source in team["players"]:
                pid = str(source["id"])
                if pid in players:
                    raise ValueError("Duplicate player in utility structure")
                players[pid] = {"name": source.get("name", pid),
                                "side": "atk" if team_id == record.attacker_team_id else "def"}
                last[pid] = (0.0, position(source.get("position")))
    previous = -math.inf
    for at, event in record.events:
        if at >= until:
            break
        if at < previous:
            raise ValueError("Events must be time ordered")
        previous = at
        if at < 0:
            continue
        kind = event.get("type", "")
        actor, target = event.get("actor", {}), event.get("target", {})
        updates = [(str(u.get("id")), u) for team in (game_state(event, record.game_id) or {}).get("teams", [])
                   for u in team.get("players", [])]
        updates += [(str(e.get("id")), e.get("state", {}).get("game", e.get("state", {})))
                    for e in (actor, target) if e.get("type") == "player"]
        for pid, state in updates:
            if pid in players and "position" in state:
                last[pid] = (at, position(state["position"]))
        if kind.endswith("-completed-plantBomb") and planted_at is None:
            planted_at = at
        if kind == "player-used-ability":
            pid = str(actor.get("id"))
            if pid not in players:
                raise ValueError("Unknown ability caster")
            ability = str(target.get("id", "other"))
            seen, point = last[pid]
            casts.append({"t": at, "event_id": str(event.get("id", "unidentified-event")),
                          "caster_id": pid, "caster": players[pid]["name"], "side": players[pid]["side"],
                          "ability": ability, "tags": list(ability_tags(ability)),
                          "point": point if point is not None and at - seen <= POSITION_MAX_AGE else None})
    for item in casts:
        # A cast at the plant timestamp has no established order; treat it as post-plant.
        item["pre_plant"] = planted_at is None or item["t"] < planted_at
    return casts


def snapshot(casts, now):
    """Channel values at one instant, from casts at or before it."""
    past = [c for c in casts if c["t"] <= now]
    values, evidence = dict.fromkeys(FEATURE_NAMES, 0.0), {}
    live = {side: [c for c in past if c["side"] == side and support_tags(c) and now - c["t"] < span(c)]
            for side in ("atk", "def")}
    for side, selected in live.items():
        points = {}
        for cast in selected:  # Time ordered: keeps each caster's latest located support cast.
            if cast["point"] is not None:
                points[cast["caster_id"]] = cast["point"]
        gaps = [math.dist(a, b) for a, b in itertools.combinations(points.values(), 2)]
        values[f"{side}_combo_casters"] = len({c["caster_id"] for c in selected}) / 5
        values[f"{side}_combo_functions"] = len({t for c in selected for t in support_tags(c)}) / len(SUPPORT_TAGS)
        values[f"{side}_combo_spacing"] = sum(gaps) / len(gaps) / SPACING_SCALE if gaps else 0.0
        values[f"{side}_combo_spacing_known"] = float(bool(gaps))
        if selected:
            evidence[f"{side}_combo"] = [c["event_id"] for c in selected]
    # Owner, Ascent R1: the duelist-around-utility frame fits an attacking site execute, not a defensive hold.
    best = None
    for entry in past:
        if (entry["side"] != "atk" or entry["ability"] not in ENTRY_ABILITIES or not entry["pre_plant"]
                or now - entry["t"] >= MAX_SPAN):
            continue
        helpers = [c for c in past if c["side"] == "atk" and support_tags(c) and c["pre_plant"]
                   and c["caster_id"] != entry["caster_id"] and abs(c["t"] - entry["t"]) < span(c)]
        count = len({c["caster_id"] for c in helpers})
        if count and (best is None or count > best[0]):
            best = (count, [entry["event_id"]] + [c["event_id"] for c in helpers])
    if best:
        values["atk_supported_entry"], values["atk_entry_support_casters"] = 1.0, best[0] / 4
        evidence["atk_supported_entry"] = best[1]
    # Owner, Ascent R1: defenders answered an attacker smoke with their own utility to retake space.
    blocks = [c["t"] for c in past if c["side"] == "atk" and "vision_block" in c["tags"]]
    answers = [c for c in live["def"] if any(0 < c["t"] - at <= ANSWER_LOOKBACK for at in blocks)]
    values["def_answer_casters"] = len({c["caster_id"] for c in answers}) / 5
    values["def_answer_functions"] = len({t for c in answers for t in support_tags(c)}) / len(SUPPORT_TAGS)
    if answers:
        evidence["def_answer"] = [c["event_id"] for c in answers]
    return values, evidence


def structure_bin(casts, end):
    """Channels for the one-second bin ending at end, plus the casts that produced them.

    Each group reports its peak inside the second. Evaluating at exact cast times means two
    casts within a span of each other always meet, wherever the whole-second boundaries fall.
    """
    instants = sorted({end - 1.0} | {c["t"] for c in casts if end - 1 <= c["t"] < end})
    shots = [snapshot(casts, now) for now in instants]
    values, evidence = dict.fromkeys(FEATURE_NAMES, 0.0), {}
    for group, names in GROUPS.items():
        # Ties go to the second channel, then to the latest instant.
        peak = max(reversed(shots), key=lambda shot: (shot[0][names[0]], shot[0][names[1]]))
        values.update({name: peak[0][name] for name in names})
        if group in peak[1]:
            evidence[group] = peak[1][group]
    return [values[name] for name in FEATURE_NAMES], evidence


def preview_structure(record, endpoints=None):
    """Rows are per round second, like the v2 contract; a window is rows[at-20:at]."""
    endpoints = observe_round(record)["endpoints"] if endpoints is None else endpoints
    if any(type(at) is not int or at < HISTORY_SECONDS for at in endpoints):
        raise ValueError("Endpoints must be whole seconds with a full history window")
    casts = extract_casts(record, max(endpoints)) if endpoints else []
    bins = [structure_bin(casts, end) for end in range(1, max(endpoints, default=0) + 1)]
    windows = []
    for at in endpoints:
        start = at - HISTORY_SECONDS
        windows.append({"at_seconds": at, "window_start": start, "rows": [row for row, _ in bins[start:at]],
                        "evidence": [{"bin_start": t, **bins[t][1]} for t in range(start, at) if bins[t][1]],
                        "casts": [c for c in casts if start - MAX_SPAN <= c["t"] < at]})
    return {"contract": contract(), "windows": windows}


def check_annotations(records, annotations):
    """Every owner-assessed cast must be a recorded cast with the stated ability, caster and time."""
    index = {record.key: record for record in records}
    checked = 0
    for annotation in annotations["annotations"]:
        record = index.get(tuple(annotation["round"]))
        if record is None:
            raise ValueError("Owner annotation names an unknown round")
        casts = {c["event_id"]: c for c in extract_casts(record, math.inf)}
        for item in annotation.get("event_assessments", []):
            cast = casts.get(item["event_id"])
            if (cast is None or cast["ability"] != item["ability"] or cast["caster_id"] != item["caster_id"]
                    or ("at_cast_seconds" in item and abs(cast["t"] - item["at_cast_seconds"]) > 0.0015)):
                raise ValueError(f"Owner annotation does not match the recorded cast {item['event_id']}")
            checked += 1
    return checked


def support_audit(previews):
    """How often each channel is nonzero. Sparse channels cannot be fitted on 44 rounds."""
    audit = {}
    for map_name in sorted({m for m, _, _ in previews}):
        rows = Counter()
        rounds, windows, total_rounds, total_windows = Counter(), Counter(), 0, 0
        for name, _, preview in previews:
            if name != map_name:
                continue
            total_rounds += 1
            active_round = set()
            for window in preview["windows"]:
                total_windows += 1
                active = {n for row in window["rows"] for n, v in zip(FEATURE_NAMES, row) if v}
                windows.update(active)
                active_round |= active
            rounds.update(active_round)
        audit[map_name] = {"rounds": total_rounds, "windows": total_windows,
                           "nonzero": {n: {"rounds": rounds[n], "windows": windows[n]} for n in FEATURE_NAMES}}
    return audit


def build_review(data_dir, annotations):
    repo = Path(__file__).resolve().parent
    sources, previews, cases, records = [], [], [], []
    wanted = {(*a["round"], a["at_seconds"]): a for a in annotations["annotations"]}
    if len(wanted) != len(annotations["annotations"]):
        raise ValueError("Duplicate owner annotation")
    for path in discover_data_files(data_dir):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sources.append({"file": path.name, "sha256": digest})
        for record in extract_rounds(path)[0]:
            records.append(record)
            preview = preview_structure(record)
            previews.append((record.map_name, record.key, preview))
            for window in preview["windows"]:
                annotation = wanted.get((*record.key, window["at_seconds"]))
                if annotation:
                    if annotation.get("source_sha256", digest) != digest:
                        raise ValueError("Owner annotation source differs from review data")
                    cases.append({"round": list(record.key), "map": record.map_name, "source_sha256": digest,
                                  **window, "owner_case_id": annotation.get("case_id"),
                                  "owner_window_assessment": annotation.get("window_assessment")})
    if len(cases) != len(wanted):
        raise ValueError("An owner annotation does not match an eligible window")
    names = ("utility_structure.py", "entry_support.py", "data_processing.py", "rolling_features.py",
             "ability_taxonomy.py")
    return {"contract": contract(),
            "implementation_sha256": {n: hashlib.sha256((repo / n).read_bytes()).hexdigest() for n in names},
            "annotations_sha256": hashlib.sha256(json.dumps(annotations, sort_keys=True,
                                                            separators=(",", ":")).encode()).hexdigest(),
            "owner_assessed_casts_matched": check_annotations(records, annotations),
            "sources": sources, "window_count": sum(len(p["windows"]) for _, _, p in previews),
            "support": support_audit(previews),
            "selection": "The owner-annotated windows only; chosen before these channels existed",
            "limitations": ["Channels were designed from these seven cases; agreement with them is not validation.",
                            "Entry markers are two named abilities; in this series one player per team casts them.",
                            "A cast is a recorded use, not a confirmed effect, landing spot or duration.",
                            "Caster spacing is not the distance between utility effects.",
                            "No model consumes these prototype features.",
                            "These development rounds are not an independent evaluation set."],
            "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--annotations", type=Path, default=Path("docs/gameplay-annotations.json"))
    args = parser.parse_args()
    review = build_review(args.data_dir, json.loads(args.annotations.read_text(encoding="utf-8")))
    print(json.dumps(review, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
