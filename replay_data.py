"""Project event exports into compact, causal replay frames. No ML dependencies."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from data_processing import (
    DEFAULT_OBSERVATION_SECONDS, FEATURE_NAMES, discover_data_files,
    extract_rounds, game_state, round_to_features,
)
from baseline_features import TEMPORAL_NAMES, temporal_summary
from experiment import public_run, read_manifest, round_key, validate_prediction
from rolling_features import preview_round
from rolling_experiment import read_predictions as read_rolling_predictions

PLAYER_FIELDS = ("name", "alive", "currentHealth", "currentArmor", "loadoutValue")
DEATH_EVENTS = {"player-killed-player", "game-killed-player",
                "player-selfkilled-player", "player-teamkilled-player"}


def round_token(key) -> str:
    return hashlib.sha256(json.dumps(list(key)).encode()).hexdigest()[:24]


def merge_player(player: dict, update: dict, timestamp: float) -> None:
    """Merge only reported fields; absent fields do not reset prior state."""
    for field in PLAYER_FIELDS:
        if field in update:
            player[field] = update[field]
    if "character" in update:
        player["agent"] = update["character"].get("name", "unknown")
    if "position" in update:
        position = update["position"]
        if position is None:
            player["position"] = None
        elif all(isinstance(position.get(axis), (float, int)) and
                 math.isfinite(position[axis]) for axis in ("x", "y")):
            player["position"] = {axis: position[axis] for axis in ("x", "y")}
        else:
            raise ValueError("Nonfinite or malformed player position")
        player["position_at"] = timestamp
    if "abilities" in update:
        abilities = player.setdefault("abilities", {})
        for ability in update["abilities"]:
            previous = abilities.setdefault(str(ability["id"]), {})
            previous.update({k: v for k, v in ability.items() if k in {"id", "name", "ready"}})


def read_predictions(path: Path | None, manifest=None, manifest_hash=None) -> dict:
    """Read prediction JSONL written by predict.py; never open a checkpoint."""
    if path is None:
        return {}
    if manifest is None or manifest_hash is None:
        raise ValueError("Predictions require their run manifest")
    if path.stat().st_size > 10_000_000:
        raise ValueError("Prediction JSONL exceeds 10 MB")
    predictions = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        validate_prediction(row, manifest, manifest_hash)
        key = round_key(row)
        if key in predictions:
            raise ValueError(f"Duplicate prediction for {key}")
        predictions[key] = {**row, "run": public_run(manifest, manifest_hash)}
    return predictions


def build_round_replay(record, prediction=None) -> dict:
    if record.live_at is None or record.ended_at is None:
        raise ValueError("Replay requires a completed round with a live start")
    players, teams = {}, []
    for team in record.start_state.get("teams", []):
        team_id = str(team["id"])
        if team_id not in {record.attacker_team_id, record.defender_team_id}:
            continue
        side = "attacker" if team_id == record.attacker_team_id else "defender"
        teams.append({"id": team_id, "name": team.get("name", team_id), "side": side,
                      "score": team.get("score", 0), "loadout": team.get("loadoutValue", 0)})
        for number, source in enumerate(team.get("players", []), 1):
            player_id = str(source["id"])
            player = {"id": player_id, "team_id": team_id, "side": side, "number": number,
                      "name": source.get("name", player_id), "alive": True, "agent": "unknown",
                      "position": None, "position_at": 0, "abilities": {}}
            merge_player(player, source, 0)
            players[player_id] = player
    teams.sort(key=lambda t: t["side"] != "attacker")
    bounds = record.start_state.get("map", {}).get("bounds")
    if not bounds or any(bounds["max"][a] <= bounds["min"][a] for a in ("x", "y")):
        raise ValueError(f"Missing map bounds for {record.key}")
    duration = record.ended_at - record.live_at
    state = {"spike": "unplanted", "winner": None, "kills": {"attacker": 0, "defender": 0},
             "ability_uses": {"attacker": 0, "defender": 0}}
    frames = [{"t": 0.0, "players": copy.deepcopy(list(players.values())), **copy.deepcopy(state)}]
    timeline = []
    for relative, event in record.events:
        if relative < 0:
            continue
        kind = event.get("type", "")
        current_state = game_state(event, record.game_id) or {}
        for team in current_state.get("teams", []):
            for update in team.get("players", []):
                if str(update.get("id")) in players:
                    merge_player(players[str(update["id"])], update, relative)
        actor, target = event.get("actor", {}), event.get("target", {})
        for entity in (actor, target):
            if entity.get("type") == "player" and str(entity.get("id")) in players:
                source = entity.get("state", {})
                merge_player(players[str(entity["id"])], source.get("game", source), relative)
        actor_player = players.get(str(actor.get("id")))
        target_player = players.get(str(target.get("id")))
        item = {"t": round(relative, 6), "id": event.get("id"), "kind": kind,
                "actor": actor_player["name"] if actor_player else str(actor.get("id", "")),
                "actor_id": actor_player["id"] if actor_player else None,
                "side": actor_player["side"] if actor_player else None}
        if kind in DEATH_EVENTS:
            if target_player:
                target_player["alive"] = False
                target_player["currentHealth"] = 0
            item.update(category="kill", target=target_player["name"] if target_player else "player")
            if kind == "player-killed-player" and actor_player and target_player:
                if actor_player["side"] != target_player["side"]:
                    state["kills"][actor_player["side"]] += 1
        elif kind == "player-used-ability":
            item.update(category="ability", ability=target.get("state", {}).get("name", target.get("id", "ability")),
                        position=copy.deepcopy(actor_player.get("position")) if actor_player else None)
            if actor_player:
                state["ability_uses"][actor_player["side"]] += 1
        elif kind.endswith("-completed-plantBomb"):
            state["spike"] = "planted"
            item.update(category="objective", label="Spike planted")
        elif kind.endswith("-completed-defuseBomb"):
            state["spike"] = "defused"
            item.update(category="objective", label="Spike defused")
        elif kind.endswith("-completed-explodeBomb"):
            state["spike"] = "exploded"
            item.update(category="objective", label="Spike detonated")
        elif kind == "team-won-round":
            state["winner"] = "attacker" if record.atk_won else "defender"
            item.update(category="outcome", label=f"{state['winner'].title()} win")
        else:
            item = None
        if item:
            timeline.append(item)
        frame = {"t": round(relative, 6), "players": copy.deepcopy(list(players.values())), **copy.deepcopy(state)}
        if frames[-1]["t"] == frame["t"]:
            frames[-1] = frame
        else:
            frames.append(frame)
    cutoff = prediction["observation_seconds"] if prediction else DEFAULT_OBSERVATION_SECONDS
    try:
        features = round_to_features(record, cutoff)
        eligibility = "eligible"
    except ValueError as exc:
        features, eligibility = [], str(exc)
    if prediction and (prediction["observation_seconds"] >= duration or
                       (record.terminal_at is not None and
                        prediction["observation_seconds"] >= record.terminal_at - record.live_at)):
        raise ValueError(f"Prediction cutoff occurs after round outcome: {record.key}")
    if prediction and not features:
        raise ValueError(f"Prediction supplied for an ineligible round: {record.key}")
    return {"id": round_token(record.key), "series_id": record.series_id,
            "game_id": record.game_id, "round_id": record.round_id, "map": record.map_name,
            "duration": duration, "bounds": bounds, "teams": teams,
            "frames": frames, "events": timeline,
            "ml": {"feature_names": FEATURE_NAMES, "features": features, "cutoff": cutoff,
                   "temporal_feature_names": TEMPORAL_NAMES,
                   "temporal_features": temporal_summary(features)[len(FEATURE_NAMES):] if features else [],
                   "eligibility": eligibility, "prediction": prediction},
            "position_mode": "last_observed", "time_origin": "end_of_buy_time"}


def build_catalog(data: Path, predictions: Path | None = None,
                  run_manifest: Path | None = None, rolling_predictions: Path | None = None,
                  rolling_run_manifest: Path | None = None) -> tuple[dict, dict]:
    files = discover_data_files(data)
    if (predictions is None) != (run_manifest is None):
        raise ValueError("Supply both --predictions and --run-manifest")
    manifest, manifest_hash = read_manifest(run_manifest) if run_manifest else (None, None)
    imported = read_predictions(predictions, manifest, manifest_hash)
    if predictions and rolling_predictions:
        raise ValueError("Choose fixed-cutoff or rolling predictions for this viewer")
    rolling_imported, rolling_run = read_rolling_predictions(rolling_predictions, rolling_run_manifest)
    metadata_path = (data if data.is_dir() else data.parent) / "series.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    series = {}
    summaries, payloads, sources = [], {}, []
    counts, seen = Counter(), set()
    for path in files:
        records, audit = extract_rounds(path)
        counts.update(audit)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        source_id = digest.hexdigest()
        sources.append({"file": path.name, "sha256": source_id})
        for record in records:
            record.source_sha256 = source_id
            if record.key in seen:
                raise ValueError(f"Duplicate round across exports: {record.key}")
            seen.add(record.key)
            if record.live_at is None:
                counts["missing_live_start"] += 1
                continue
            prediction = imported.get(record.key)
            if prediction and prediction["source_sha256"] != source_id:
                raise ValueError(f"Prediction source hash differs from replay data: {record.key}")
            payload = build_round_replay(record, prediction)
            if predictions is None:
                try:
                    preview = preview_round(record)
                except ValueError as exc:
                    # Ineligible rounds remain replayable but never accept predictions.
                    preview = {"history_seconds": 20, "stride_seconds": 5, "feature_count": 0,
                               "display_names": [], "windows": [], "eligibility": str(exc)}
                rows = rolling_imported.get(record.key, [])
                eligible_times = {w["at_seconds"] for w in preview["windows"]}
                for row in rows:
                    if row["source_sha256"] != source_id or row["at_seconds"] not in eligible_times:
                        raise ValueError("Rolling prediction source or observation time differs from replay data")
                payload["ml"].update(mode="rolling", rolling=preview, predictions=rows, run=rolling_run)
            payload["source_sha256"] = source_id
            observed_date = datetime.fromtimestamp(record.live_at, timezone.utc).date().isoformat()
            if record.series_id not in series:
                series[record.series_id] = {
                    "id": record.series_id,
                    "teams": sorted(t["name"] for t in payload["teams"]),
                    "date": observed_date, "timezone": "UTC", "tournament": None, "stage": None,
                }
            info = series[record.series_id]
            known = metadata.get(record.series_id, {})
            if known.get("source_sha256") == source_id:
                info.update({key: known[key] for key in
                             ("date", "timezone", "tournament", "stage", "tournament_id", "reference") if key in known})
            elif not info["tournament"]:
                info["date"] = min(info["date"], observed_date)
            token = payload["id"]
            payloads[token] = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
            summaries.append({k: payload[k] for k in ("id", "series_id", "game_id", "round_id", "map", "duration", "teams")})
    unknown = set(imported) - seen
    if unknown:
        raise ValueError("Predictions include rounds absent from the selected dataset")
    if set(rolling_imported) - seen:
        raise ValueError("Rolling predictions include rounds absent from the selected dataset")
    if not summaries:
        raise ValueError("No completed rounds with live-start snapshots were found")
    counts["replay_rounds"] = len(summaries)
    return {"schema_version": 1, "rounds": summaries, "sources": sources, "series": list(series.values()),
            "series_count": len({r["series_id"] for r in summaries}),
            "map_count": len({(r["series_id"], r["game_id"]) for r in summaries}),
            "counts": dict(counts), "prediction_count": len(imported) + sum(len(rows) for rows in rolling_imported.values()),
            "prediction_run": public_run(manifest, manifest_hash) if manifest else rolling_run}, payloads
