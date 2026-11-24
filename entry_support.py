"""Untrained utility/movement relationship prototype; standard library only.

Player identifiers join observations within a round and appear in review evidence,
never in the numeric channels. Temporal proximity is not verified utility support,
an entry, contact, visibility, or an ability effect.
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
from rolling_features import DEATHS, HISTORY_SECONDS, observe_round

MAX_REPORT_GAP = 2.0
CAST_LOOKBACK = 5.0
FIELDS = (
    "observed_pairs", "moving_pairs", "unusable_reports", "displacement_sum",
    "preceding_ally_pairs", "overlapping_ally_pairs", "preceding_self_pairs",
    "moving_without_ally_cast_pairs", "ally_casters_sum", "ally_casts_sum",
    "latest_ally_delay_mean", "ally_delay_known_pairs",
    "nearest_support_spacing_mean", "support_spacing_known_pairs",
    "ally_vision_impair_pairs", "ally_vision_block_pairs", "ally_recon_pairs",
)
FEATURE_NAMES = [f"{side}_{field}" for side in ("atk", "def") for field in FIELDS]


def contract():
    return {"id": "entry-support-prototype-v0", "status": "untrained_owner_review",
            "history_seconds": HISTORY_SECONDS, "bin_seconds": 1, "interval": "[start,end)",
            "max_report_gap_seconds": MAX_REPORT_GAP, "cast_lookback_seconds": CAST_LOOKBACK,
            "feature_names": FEATURE_NAMES, "identity_inputs": False,
            "distance_unit": "Euclidean feed-coordinate units; not meters or path length",
            "semantics": "Same-player reported displacement near allied/self casts; no verified entry or effect",
            "tie_policy": "Merge equal-time reports; casts at the second report time cannot support that pair"}


def position(value):
    if value is None:
        return None
    point = [value.get(axis) for axis in ("x", "y")]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in point):
        raise ValueError("Malformed entry-review position")
    return point


def extract_relations(record, until):
    """Read only events before until; attach each pair to its second report time."""
    if not math.isfinite(until) or until <= 0:
        raise ValueError("Positive finite endpoint required")
    players, reports, casts, pairs, unusable, death_times = {}, {}, [], [], [], {}
    for team in record.start_state["teams"]:
        team_id = str(team["id"])
        if team_id not in (record.attacker_team_id, record.defender_team_id):
            continue
        side = "atk" if team_id == record.attacker_team_id else "def"
        for source in team["players"]:
            pid = str(source["id"])
            if pid in players:
                raise ValueError("Duplicate player in entry review")
            players[pid] = {"id": pid, "name": source.get("name", pid), "side": side,
                            "alive": source.get("alive", True)}
            initial = position(source.get("position"))
            reports[pid] = [{"t": 0.0, "point": initial, "source_ids": ["live-start"],
                             "alive": players[pid]["alive"]}]

    # Reading this prefix does not inspect subsequent event bodies or the label.
    prefix = []
    last = -math.inf
    for at, event in record.events:
        if at >= until:
            break
        if at < last:
            raise ValueError("Events must be time ordered")
        last = at
        if at >= 0:
            prefix.append((at, event))
    for at, group in itertools.groupby(prefix, key=lambda item: item[0]):
        updates, new_casts, deaths = {}, [], set()

        def merge(pid, state, event_id):
            if pid not in players:
                return
            if "alive" in state:
                players[pid]["alive"] = state["alive"]
            if "position" in state:
                point = position(state["position"])
                prior = updates.get(pid)
                ids = prior["source_ids"] + [event_id] if prior and prior["point"] == point else [event_id]
                updates[pid] = {"t": at, "point": point, "source_ids": sorted(set(ids))}

        for _, event in group:
            event_id = str(event.get("id", "unidentified-event"))
            kind = event.get("type", "")
            if "reviv" in kind.lower() or "respawn" in kind.lower():
                raise ValueError("Revive/respawn semantics require review")
            for team in (game_state(event, record.game_id) or {}).get("teams", []):
                for update in team.get("players", []):
                    merge(str(update.get("id")), update, event_id)
            actor, target = event.get("actor", {}), event.get("target", {})
            for entity in (actor, target):
                if entity.get("type") == "player":
                    state = entity.get("state", {})
                    merge(str(entity.get("id")), state.get("game", state), event_id)
            if kind in DEATHS:
                deaths.add(str(target.get("id")))
            if kind == "player-used-ability":
                pid = str(actor.get("id"))
                if pid not in players:
                    raise ValueError("Unknown ability caster")
                ability = str(target.get("id", "other"))
                new_casts.append({"t": at, "event_id": event_id, "caster_id": pid,
                                  "caster": players[pid]["name"], "side": players[pid]["side"],
                                  "ability": ability, "tags": list(ability_tags(ability))})
        for pid in deaths:
            if pid in players:
                players[pid]["alive"] = False
                death_times.setdefault(pid, at)
        # All equal-time reports are available together; no arbitrary event-order movement.
        for pid, report in updates.items():
            report["alive"] = players[pid]["alive"]
            if reports[pid][-1]["t"] == at:
                reports[pid][-1] = report
            else:
                reports[pid].append(report)
        for pid, second in updates.items():
            history = reports[pid]
            if len(history) < 2 or not players[pid]["alive"] or pid in death_times:
                continue
            first = history[-2]
            gap = at - first["t"]
            if not first["alive"] or first["point"] is None or second["point"] is None or not 0 < gap <= MAX_REPORT_GAP:
                unusable.append({"t": at, "side": players[pid]["side"], "player_id": pid,
                                 "reason": "missing position or report gap over 2s"})
                continue
            related = []
            for cast in casts:
                if cast["side"] != players[pid]["side"] or not 0 < at - cast["t"] <= CAST_LOOKBACK:
                    continue
                relation = "self" if cast["caster_id"] == pid else "ally"
                timing = "preceding" if cast["t"] <= first["t"] else "overlapping"
                # Measure spacing at the first movement report, never from a future frame.
                support_report = next((p for p in reversed(reports[cast["caster_id"]]) if p["t"] <= first["t"]), None)
                known = (relation == "ally" and support_report and support_report["point"] is not None
                         and support_report["alive"] and first["t"] - support_report["t"] <= MAX_REPORT_GAP
                         and death_times.get(cast["caster_id"], math.inf) > first["t"])
                related.append({**cast, "relation": relation, "timing": timing,
                                "delay_to_second_report": at - cast["t"],
                                "spacing_at_first_report": math.dist(first["point"], support_report["point"]) if known else None,
                                "support_position_age": first["t"] - support_report["t"] if known else None,
                                "support_position_source_ids": support_report["source_ids"] if known else []})
            pairs.append({"t": at, "start_t": first["t"], "report_gap": gap,
                          "player_id": pid, "player": players[pid]["name"], "side": players[pid]["side"],
                          "from": first["point"], "to": second["point"],
                          "source_ids": [first["source_ids"], second["source_ids"]],
                          "displacement": math.dist(first["point"], second["point"]), "casts": related})
        casts.extend(new_casts)
    return {"pairs": pairs, "unusable": unusable}


def numeric_bin(pairs, unusable):
    result = dict.fromkeys(FEATURE_NAMES, 0.0)
    for side in ("atk", "def"):
        selected = [p for p in pairs if p["side"] == side]
        values = Counter(observed_pairs=len(selected), unusable_reports=sum(p["side"] == side for p in unusable))
        delays, spacings = [], []
        for pair in selected:
            moving = pair["displacement"] > 0
            values["moving_pairs"] += moving
            values["displacement_sum"] += pair["displacement"]
            allies = [c for c in pair["casts"] if c["relation"] == "ally"]
            values["preceding_ally_pairs"] += moving and any(c["timing"] == "preceding" for c in allies)
            values["overlapping_ally_pairs"] += moving and any(c["timing"] == "overlapping" for c in allies)
            values["preceding_self_pairs"] += moving and any(c["relation"] == "self" and c["timing"] == "preceding" for c in pair["casts"])
            values["moving_without_ally_cast_pairs"] += moving and not allies
            values["ally_casters_sum"] += len({c["caster_id"] for c in allies}) if moving else 0
            values["ally_casts_sum"] += len(allies) if moving else 0
            if moving and allies:
                delays.append(min(c["delay_to_second_report"] for c in allies))
                known = [c["spacing_at_first_report"] for c in allies if c["spacing_at_first_report"] is not None]
                if known:
                    spacings.append(min(known))
            for tag in ("vision_impair", "vision_block", "recon"):
                values[f"ally_{tag}_pairs"] += moving and any(tag in c["tags"] for c in allies)
        values.update(latest_ally_delay_mean=sum(delays) / len(delays) if delays else 0,
                      ally_delay_known_pairs=len(delays),
                      nearest_support_spacing_mean=sum(spacings) / len(spacings) if spacings else 0,
                      support_spacing_known_pairs=len(spacings))
        result.update({f"{side}_{name}": float(values[name]) for name in FIELDS})
    return [result[name] for name in FEATURE_NAMES]


def window_evidence(observed, at):
    start = at - HISTORY_SECONDS
    # Both reports must be inside the window. No hidden pre-window movement.
    pairs = [p for p in observed["pairs"] if start <= p["start_t"] and p["t"] < at]
    pairs = [{**p, "casts": [c for c in p["casts"] if start <= c["t"]]} for p in pairs]
    unusable = [p for p in observed["unusable"] if start <= p["t"] < at]
    rows = [numeric_bin([p for p in pairs if t <= p["t"] < t + 1],
                        [p for p in unusable if t <= p["t"] < t + 1]) for t in range(start, at)]
    return {"at_seconds": at, "window_start": start, "rows": rows, "pairs": pairs,
            "unusable_report_count": len(unusable),
            "moving_pair_count": sum(p["displacement"] > 0 for p in pairs),
            "ally_linked_moving_pairs": sum(p["displacement"] > 0 and any(c["relation"] == "ally" for c in p["casts"]) for p in pairs)}


def preview_entry_support(record, endpoints=None):
    endpoints = observe_round(record)["endpoints"] if endpoints is None else endpoints
    observed = extract_relations(record, max(endpoints)) if endpoints else {"pairs": [], "unusable": []}
    return {"contract": contract(), "windows": [window_evidence(observed, at) for at in endpoints]}


def build_review(data_dir, annotations=None):
    annotations = annotations or {"annotations": []}
    annotation_index = {(*a["round"], a["at_seconds"]): a for a in annotations["annotations"]}
    if len(annotation_index) != len(annotations["annotations"]):
        raise ValueError("Duplicate owner annotation")
    candidates, sources = [], []
    for path in discover_data_files(data_dir):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sources.append({"file": path.name, "sha256": digest})
        for record in extract_rounds(path)[0]:
            for window in preview_entry_support(record)["windows"]:
                candidates.append({"round": list(record.key), "map": record.map_name, "source_sha256": digest, **window})
    cases, selected = [], set()

    def add(case, reason):
        key = (*case["round"], case["at_seconds"])
        if key not in selected:
            annotation = annotation_index.get(key)
            if annotation and annotation.get("source_sha256", case["source_sha256"]) != case["source_sha256"]:
                raise ValueError("Owner annotation source differs from review data")
            selected.add(key)
            # Only an assessment the owner recorded fills this field; it is never inferred.
            cases.append({**case, "selection_reason": reason,
                          "owner_annotation": annotation,
                          "owner_review": (annotation or {}).get("window_assessment")
                          or {"useful_support": None, "reason": None}})

    for case in candidates:
        if case["map"] == "ascent" and case["round"][-1] == "round-13" and case["at_seconds"] == 25:
            add(case, "Previously owner-identified B-main entry; no outcome-based selection")
    for map_name in sorted({c["map"] for c in candidates}):
        for reason, predicate in (
            ("Movement near allied casts", lambda c: c["ally_linked_moving_pairs"] > 0),
            ("Moving pairs without recent allied casts", lambda c: c["moving_pair_count"] > 0 and c["ally_linked_moving_pairs"] == 0),
            ("No usable moving pairs; missingness/stationarity review", lambda c: c["moving_pair_count"] == 0),
        ):
            match = next((c for c in candidates if c["map"] == map_name and predicate(c)), None)
            if match:
                add(match, reason + "; first eligible window in source order")
    repo = Path(__file__).resolve().parent
    implementation = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in
                      ("entry_support.py", "data_processing.py", "rolling_features.py", "ability_taxonomy.py")}
    return {"contract": contract(), "implementation_sha256": implementation,
            "annotations_sha256": hashlib.sha256(json.dumps(annotations, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "sources": sources, "window_count": len(candidates),
            "selection": "Owner-pinned case plus first eligible window per map/stratum; no probabilities or outcomes used",
            "limitations": ["Temporal association is not verified utility support or entry success.",
                            "Displacement is not a route; feed units and thresholds need domain review.",
                            "No verified ability effects, enemy visibility, contact, or map-region crossings.",
                            "These development examples are not an independent evaluation set.",
                            "No model consumes these prototype features."],
            "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--annotations", type=Path, default=Path("docs/gameplay-annotations.json"))
    args = parser.parse_args()
    print(json.dumps(build_review(args.data_dir, json.loads(args.annotations.read_text())), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
