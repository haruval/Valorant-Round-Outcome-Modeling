"""Causal one-second observations and 20s windows every 5s; standard library only."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from data_processing import discover_data_files, extract_rounds, game_state, team_of
from ability_taxonomy import CATEGORIES, ability_tags, taxonomy_contract

HISTORY_SECONDS = 20
STRIDE_SECONDS = 5
POSITION_MAX_AGE = 2.0
TRADE_MAX_SECONDS = 5.0  # Owner-approved initial proxy; not a provider trade label.
ABILITIES = sorted([
    "alarmbot", "blade-storm", "cloudburst", "cyber-cage", "dark-cover", "flash-drive",
    "frag-ment", "from-the-shadows", "hunter's-fury", "lockdown", "nanoswarm",
    "neural-theft", "null-cmd", "owl-drone", "paranoia", "poison-cloud", "recon-bolt",
    "shock-bolt", "shrouded-step", "snake-bite", "spycam", "tailwind", "toxic-screen",
    "trapwire", "turret", "updraft", "viper's-pit", "zero-point",
]) + ["other"]
DEATHS = {"player-killed-player", "game-killed-player", "player-selfkilled-player", "player-teamkilled-player"}
GLOBAL_NAMES = ["elapsed_120s", "spike_planted", "plant_age_45s", "defusing", "defuse_checkpoint",
                "clock_reported_120s", "clock_known", "clock_age_20s"]
SIDE_STATE = ["alive", "start_loadout", "alive_loadout", "reported_health", "health_known", "health_age_20s",
              "reported_armor", "armor_known", "armor_age_20s", "position_fresh", "position_unknown",
              "position_age_20s", "centroid_x", "centroid_y", "spread", "kills_total", "abilities_total",
              "recon_cast_known", "recon_cast_age_20s", "movement_pairs", "movement_mean",
              "movement_after_recon_pairs", "movement_after_recon_mean"]
SIDE_EVENTS = ["kills_1s", "deaths_1s", "trade_candidates_1s", "trade_delay_5s", "abilities_1s"]
SPATIAL_EVENTS = ["kill_origin", "kill_victim", "cast_origin"]


def feature_names():
    names = list(GLOBAL_NAMES)
    for side in ("atk", "def"):
        names += [f"{side}_{name}" for name in SIDE_STATE + SIDE_EVENTS]
        names += [f"{side}_presence_cell_{cell}" for cell in range(9)]
        for kind in SPATIAL_EVENTS:
            names += [f"{side}_{kind}_cell_{cell}" for cell in range(9)]
            names += [f"{side}_{kind}_unknown"]
        for category in CATEGORIES:
            names += [f"{side}_utility_{category}_{field}" for field in ("uses", "x", "y", "located")]
    return names


FEATURE_NAMES = feature_names()


def schema():
    calibration = json.loads((Path(__file__).parent / "viewer/maps/manifest.json").read_text())
    geometry = {name: {key: value[key] for key in
                       ("xMultiplier", "yMultiplier", "xScalarToAdd", "yScalarToAdd")} for name, value in calibration.items()}
    return {"id": "rolling-gameplay-v2", "feature_names": FEATURE_NAMES,
            "history_seconds": HISTORY_SECONDS, "stride_seconds": STRIDE_SECONDS,
            "bin_seconds": 1, "interval": "[start,end)", "position_max_age": POSITION_MAX_AGE,
            "trade_max_seconds": TRADE_MAX_SECONDS, "grid": [3, 3], "calibration": geometry,
            "map_identity_inputs": False, "ability_taxonomy": taxonomy_contract(),
            "movement": {"max_report_gap_seconds": 2, "after_recon_seconds": 5,
                         "unit": "normalized minimap displacement; not path length or confirmed rotation"}}


def legacy_schema():
    """Frozen v1 contract, supported only for replaying existing prediction JSON."""
    return json.loads((Path(__file__).parent / "schemas/rolling-multimodal-v1.json").read_text())


def validate_contract(contract, allow_legacy=False):
    if contract != schema() and not (allow_legacy and contract == legacy_schema()):
        raise ValueError("Unsupported rolling feature contract")


def xy(position, calibration):
    if not position or not calibration:
        return None
    values = (position.get("x"), position.get("y"))
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("Malformed recorded position")
    x = values[1] * calibration["xMultiplier"] + calibration["xScalarToAdd"]
    y = values[0] * calibration["yMultiplier"] + calibration["yScalarToAdd"]
    return (x, y) if 0 <= x <= 1 and 0 <= y <= 1 else None


def cell(point):
    return min(2, int(point[1] * 3)) * 3 + min(2, int(point[0] * 3))


def update_player(player, update, at):
    for key in ("position", "currentHealth", "currentArmor", "loadoutValue", "alive"):
        if key in update:
            player[key] = copy.deepcopy(update[key])
            player[key + "_at"] = at


def observe_round(record, contract=None):
    """Reconstruct observations without score, winner, IDs, or future state as inputs.

    All event state is applied strictly before each bin endpoint. Location channels
    use observed coordinates only. Timing and unknown masks retain source limits.
    """
    contract = contract or schema()
    validate_contract(contract, allow_legacy=True)
    legacy = contract["id"] == "rolling-multimodal-v1"
    if record.live_at is None or record.ended_at is None or record.paused:
        raise ValueError("Rolling observations require an unpaused completed round with a live start")
    end = min(record.ended_at, record.terminal_at or record.ended_at) - record.live_at
    if not 0 < end <= 600:
        raise ValueError("Round duration outside supported 0–600s range")
    calibration = contract["calibration"].get(record.map_name)
    players, start_loadout = {}, {}
    for side, team_id in (("atk", record.attacker_team_id), ("def", record.defender_team_id)):
        team = next(t for t in record.start_state["teams"] if str(t["id"]) == team_id)
        if len(team["players"]) != 5 or len({str(p["id"]) for p in team["players"]}) != 5:
            raise ValueError("Expected five distinct players per team")
        start_loadout[side] = float(team["loadoutValue"]) / 25000
        for source in team["players"]:
            player = {"side": side, "alive": True}
            update_player(player, source, 0.0)
            players[str(source["id"])] = player
    rows, evidence, coverage = [], [], Counter()
    kills, casts = Counter(), Counter()
    prior_kills = []
    planted_at, defusing, checkpoint = None, False, False
    clock_value, clock_at = None, 0.0
    cursor = 0
    decided_at = end
    ages = []
    last_recon = {"atk": None, "def": None}
    previous_positions = {}
    for endpoint in range(1, math.ceil(end)):
        counts = Counter()
        locations = {}
        while cursor < len(record.events) and record.events[cursor][0] < endpoint:
            at, event = record.events[cursor]
            cursor += 1
            if at < 0:
                continue
            kind = event.get("type", "")
            if "reviv" in kind.lower() or "respawn" in kind.lower():
                raise ValueError("Revive/respawn semantics require review")
            snapshot = game_state(event, record.game_id) or {}
            for team in snapshot.get("teams", []):
                for update in team.get("players", []):
                    if str(update.get("id")) in players:
                        update_player(players[str(update["id"])], update, at)
            clock = snapshot.get("clock", {})
            if type(clock.get("currentSeconds")) in (int, float):
                clock_value, clock_at = clock["currentSeconds"], at
            actor, target = event.get("actor", {}), event.get("target", {})
            for entity in (actor, target):
                if entity.get("type") == "player" and str(entity.get("id")) in players:
                    state = entity.get("state", {})
                    update_player(players[str(entity["id"])], state.get("game", state), at)

            def event_point(entity):
                player = players.get(str(entity.get("id")), {})
                if at - player.get("position_at", -1e9) > POSITION_MAX_AGE:
                    return None
                return xy(player.get("position"), calibration)

            def add_location(side, name, point):
                counts[f"{side}_{name}_cell_{cell(point)}" if point is not None else f"{side}_{name}_unknown"] += 1

            if kind == "player-used-ability":
                side = "atk" if team_of(actor, record) == record.attacker_team_id else "def"
                ability = str(target.get("id", "other"))
                ability = ability if ability in ABILITIES else "other"
                point = event_point(actor)
                casts[side] += 1
                counts[f"{side}_abilities_1s"] += 1
                counts[f"{side}_ability_{ability}_uses"] += 1
                if point is not None:
                    locations.setdefault((side, ability), []).append(point)
                add_location(side, "cast_origin", point)
                tags = ability_tags(str(target.get("id", "unknown")))
                for tag in tags:
                    counts[f"{side}_utility_{tag}_uses"] += 1
                    if point is not None:
                        locations.setdefault((side, "utility_" + tag), []).append(point)
                if "recon" in tags:
                    last_recon[side] = at
                evidence.append({"t": at, "kind": "ability", "side": side,
                                 "ability": str(target.get("id", "unknown")), "origin": point})
            elif kind in DEATHS:
                victim = players.get(str(target.get("id")))
                if victim is None:
                    raise ValueError("Unknown death victim")
                victim["alive"] = False
                victim["currentHealth"], victim["currentHealth_at"] = 0, at
                counts[f"{victim['side']}_deaths_1s"] += 1
                killer = players.get(str(actor.get("id")))
                if kind == "player-killed-player" and killer and killer["side"] != victim["side"]:
                    side = killer["side"]
                    kills[side] += 1
                    counts[f"{side}_kills_1s"] += 1
                    origin, destination = event_point(actor), event_point(target)
                    add_location(side, "kill_origin", origin)
                    add_location(side, "kill_victim", destination)
                    candidates = [prior for prior in prior_kills if prior["killer"] == str(target["id"])
                                  and prior["side"] != side and 0 < at - prior["t"] <= TRADE_MAX_SECONDS]
                    delay = at - candidates[-1]["t"] if candidates else None
                    if delay is not None:
                        counts[f"{side}_trade_candidates_1s"] += 1
                        counts[f"{side}_trade_delay_5s"] += delay / TRADE_MAX_SECONDS
                    prior_kills.append({"t": at, "killer": str(actor["id"]), "side": side})
                    evidence.append({"t": at, "kind": "kill", "side": side, "origin": origin,
                                     "victim_position": destination, "trade_delay": delay})
            elif kind.endswith("-completed-plantBomb"):
                planted_at = at
            elif kind.endswith("-completed-beginDefuseBomb"):
                defusing = True
            elif kind.endswith("-completed-stopDefuseBomb"):
                defusing = False
            elif kind.endswith("-completed-reachDefuseBombCheckpoint"):
                checkpoint = True
            # All defenders dead decides the round. Dead attackers do not decide a planted round.
            alive_sides = {p["side"] for p in players.values() if p["alive"]}
            if "def" not in alive_sides or ("atk" not in alive_sides and planted_at is None):
                decided_at = min(decided_at, at)
        values = {"elapsed_120s": endpoint / 120, "spike_planted": float(planted_at is not None),
                  "plant_age_45s": (endpoint - planted_at) / 45 if planted_at is not None else 0,
                  "defusing": float(defusing), "defuse_checkpoint": float(checkpoint),
                  "clock_reported_120s": clock_value / 120 if clock_value is not None else 0,
                  "clock_known": float(clock_value is not None),
                  "clock_age_20s": min((endpoint - clock_at) / 20, 1) if clock_value is not None else 1}
        if legacy:
            values.update(map_breeze=float(record.map_name == "breeze"),
                          map_ascent=float(record.map_name == "ascent"),
                          map_other=float(record.map_name not in ("breeze", "ascent")))
        values.update(counts)
        next_positions = {}
        for side in ("atk", "def"):
            living = [p for p in players.values() if p["side"] == side and p["alive"]]
            values.update({f"{side}_alive": len(living) / 5, f"{side}_start_loadout": start_loadout[side],
                           f"{side}_alive_loadout": sum(float(p.get("loadoutValue", 0)) for p in living) / 25000,
                           f"{side}_kills_total": kills[side] / 5, f"{side}_abilities_total": casts[side] / 20})
            for field, name, scale in (("currentHealth", "health", 500), ("currentArmor", "armor", 250)):
                known = [p for p in living if p.get(field) is not None]
                values[f"{side}_reported_{name}"] = sum(float(p[field]) for p in known) / scale
                values[f"{side}_{name}_known"] = len(known) / 5
                values[f"{side}_{name}_age_20s"] = sum(min((endpoint - p[field + '_at']) / 20, 1) for p in known) / len(known) if known else 1
            fresh, side_ages = [], []
            for player in living:
                point = xy(player.get("position"), calibration)
                age = endpoint - player.get("position_at", -1e9)
                coverage["alive_player_seconds"] += 1
                if point is None:
                    coverage["position_unknown_seconds"] += 1
                else:
                    ages.append(age)
                    side_ages.append(min(age / 20, 1))
                    if age <= POSITION_MAX_AGE:
                        fresh.append(point)
                        key = f"{side}_presence_cell_{cell(point)}"
                        values[key] = values.get(key, 0) + .2
                        coverage["position_fresh_seconds"] += 1
                    else:
                        coverage["position_stale_seconds"] += 1
            values[f"{side}_position_fresh"] = len(fresh) / 5
            values[f"{side}_position_unknown"] = (len(living) - len(side_ages)) / 5
            values[f"{side}_position_age_20s"] = sum(side_ages) / len(side_ages) if side_ages else 1
            if fresh:
                center = [sum(point[i] for point in fresh) / len(fresh) for i in (0, 1)]
                values[f"{side}_centroid_x"], values[f"{side}_centroid_y"] = center
                values[f"{side}_spread"] = sum(math.dist(point, center) for point in fresh) / len(fresh)
            # Match the same living player across bins. Changing roster membership
            # cannot turn a death into movement. Do not bridge stale report gaps.
            distances, after_recon = [], []
            for player_id, player in players.items():
                if player["side"] != side or not player["alive"]:
                    continue
                point = xy(player.get("position"), calibration)
                reported_at = player.get("position_at", -1e9)
                if point is None or endpoint - reported_at > POSITION_MAX_AGE:
                    continue
                next_positions[player_id] = (point, reported_at)
                previous = previous_positions.get(player_id)
                if previous and 0 < reported_at - previous[1] <= 2:
                    distance = math.dist(point, previous[0])
                    distances.append(distance)
                    if (last_recon[side] is not None and previous[1] >= last_recon[side]
                            and reported_at - last_recon[side] <= 5):
                        after_recon.append(distance)
            values[f"{side}_movement_pairs"] = len(distances) / 5
            values[f"{side}_movement_mean"] = sum(distances) / len(distances) if distances else 0
            values[f"{side}_movement_after_recon_pairs"] = len(after_recon) / 5
            values[f"{side}_movement_after_recon_mean"] = sum(after_recon) / len(after_recon) if after_recon else 0
            values[f"{side}_recon_cast_known"] = float(last_recon[side] is not None)
            values[f"{side}_recon_cast_age_20s"] = min((endpoint - last_recon[side]) / 20, 1) if last_recon[side] is not None else 1
            for ability in ABILITIES:
                points = locations.get((side, ability), [])
                prefix = f"{side}_ability_{ability}"
                values[prefix + "_located"] = len(points)
                if points:
                    values[prefix + "_x"] = sum(p[0] for p in points) / len(points)
                    values[prefix + "_y"] = sum(p[1] for p in points) / len(points)
            for tag in CATEGORIES:
                points = locations.get((side, "utility_" + tag), [])
                prefix = f"{side}_utility_{tag}"
                values[prefix + "_located"] = len(points)
                if points:
                    values[prefix + "_x"] = sum(p[0] for p in points) / len(points)
                    values[prefix + "_y"] = sum(p[1] for p in points) / len(points)
        previous_positions = next_positions
        row = [float(values.get(name, 0)) for name in contract["feature_names"]]
        if not all(math.isfinite(value) for value in row):
            raise ValueError("Nonfinite rolling feature")
        rows.append(row)
    endpoints = [t for t in range(HISTORY_SECONDS, len(rows) + 1, STRIDE_SECONDS) if t < decided_at]
    return {"rows": rows, "endpoints": endpoints, "events": evidence, "coverage": dict(coverage),
            "position_ages": ages, "decided_at": decided_at}


@dataclass
class RollingDataset:
    features: list
    labels: list
    records: list
    endpoints: list
    weights: list
    audit: dict


def preview_round(record, contract=None):
    """Compact feature-only viewer payload; never calculates a model probability."""
    contract = contract or schema()
    observed = observe_round(record, contract)
    names = ["atk_alive", "def_alive", "atk_start_loadout", "def_start_loadout", "spike_planted",
             "plant_age_45s", "defusing", "defuse_checkpoint", "atk_position_fresh", "def_position_fresh",
             "atk_position_age_20s", "def_position_age_20s"]
    columns = {name: i for i, name in enumerate(contract["feature_names"])}
    windows = []
    for at in observed["endpoints"]:
        events = [e for e in observed["events"] if at - HISTORY_SECONDS <= e["t"] < at]
        uses = Counter((e["side"], e["ability"]) for e in events if e["kind"] == "ability")
        counts = {f"{side}_{plural}": sum(e["kind"] == kind and e["side"] == side for e in events)
                  for side in ("atk", "def") for kind, plural in (("ability", "abilities"), ("kill", "kills"))}
        counts["trade_candidates"] = sum(e["kind"] == "kill" and e["trade_delay"] is not None for e in events)
        rows = observed["rows"][at - HISTORY_SECONDS:at]
        alive = sum(row[columns[f"{side}_alive"]] for row in rows for side in ("atk", "def"))
        fresh = sum(row[columns[f"{side}_position_fresh"]] for row in rows for side in ("atk", "def"))
        window = {"at_seconds": at, "window_start": at - HISTORY_SECONDS, "counts": counts,
                        "fresh_position_fraction": fresh / alive if alive else None,
                        "abilities": [{"side": s, "ability": a, "uses": n} for (s, a), n in sorted(uses.items())],
                        "kills": [e for e in events if e["kind"] == "kill"],
                        "features": [observed["rows"][at - 1][columns[name]] for name in names]}
        if contract["id"] == "rolling-gameplay-v2":
            tagged = Counter((e["side"], tag) for e in events if e["kind"] == "ability"
                             for tag in ability_tags(e["ability"]))
            window["utility_categories"] = [{"side": side, "category": tag, "uses": count}
                                             for (side, tag), count in sorted(tagged.items())]
            window["movement"] = {
                side: {"report_pairs": round(5 * sum(row[columns[f"{side}_movement_pairs"]] for row in rows)),
                       "after_recon_pairs": round(5 * sum(row[columns[f"{side}_movement_after_recon_pairs"]] for row in rows))}
                for side in ("atk", "def")}
        windows.append(window)
    return {"schema_id": contract["id"], "history_seconds": HISTORY_SECONDS, "stride_seconds": STRIDE_SECONDS,
            "feature_count": len(contract["feature_names"]), "display_names": names, "windows": windows,
            "ability_tags": contract.get("ability_taxonomy", {}).get("abilities", {}),
            "position_max_age": POSITION_MAX_AGE}


def load_rolling_dataset(paths):
    features, labels, records_out, endpoints, weights = [], [], [], [], []
    sources, seen, ages = [], set(), []
    counts, coverage = Counter(), Counter()
    contract = schema()
    for path in sorted(paths):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sources.append({"file": path.name, "sha256": digest})
        records, audit = extract_rounds(path)
        counts.update(audit)
        for record in records:
            if record.key in seen:
                raise ValueError("Duplicate round across rolling data sources")
            seen.add(record.key)
            record.source_sha256 = digest
            if record.live_at is None or record.paused:
                counts["missing_live_start_or_paused"] += 1
                continue
            observations = observe_round(record, contract)
            coverage.update(observations["coverage"])
            ages.extend(observations["position_ages"])
            times = observations["endpoints"]
            if not times:
                counts["no_eligible_window"] += 1
                continue
            counts["eligible_rounds"] += 1
            for at in times:
                features.append(observations["rows"][at - HISTORY_SECONDS:at])
                labels.append(record.atk_won)
                records_out.append(record)
                endpoints.append(at)
                weights.append(1 / len(times))
    counts["windows"] = len(features)
    counts["series"] = len({r.series_id for r in records_out})
    counts["maps"] = len({(r.series_id, r.game_id) for r in records_out})
    ages.sort()
    audit = {"counts": dict(counts), "sources": sources, "schema": contract,
             "coverage": dict(coverage), "position_age_seconds": {
                 name: ages[int((len(ages) - 1) * q)] if ages else None
                 for name, q in (("median", .5), ("p90", .9), ("p99", .99), ("max", 1))},
             "windows_by_time": dict(sorted(Counter(endpoints).items())),
             "weighting": "Each round has total sample weight 1 across its windows"}
    return RollingDataset(features, labels, records_out, endpoints, weights, audit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    print(json.dumps(load_rolling_dataset(discover_data_files(args.data)).audit, indent=2, allow_nan=False))
