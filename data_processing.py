"""Read local GRID JSONL into fixed, pre-outcome observation windows.

This module uses only the Python standard library. Times are feed timestamps,
relative to round-ended-freezetime, not the in-game clock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

SCHEMA_VERSION = 1
DEFAULT_OBSERVATION_SECONDS = 30
FEATURE_NAMES = [
    "elapsed_fraction", "atk_alive", "def_alive", "atk_kills", "def_kills",
    "spike_planted", "atk_loadout", "def_loadout", "atk_ability_uses",
    "def_ability_uses", "first_kill_known", "first_kill_atk",
]


@dataclass
class RoundRecord:
    series_id: str
    game_id: str
    round_id: str
    map_name: str
    attacker_team_id: str
    defender_team_id: str
    start_state: dict
    events: list[tuple[float, dict]] = field(default_factory=list)
    live_at: float | None = None
    ended_at: float | None = None
    atk_won: int | None = None
    terminal_at: float | None = None
    paused: bool = False
    source_sha256: str = ""

    @property
    def key(self) -> tuple[str, str, str]:
        return self.series_id, self.game_id, self.round_id


@dataclass
class Dataset:
    features: list[list[list[float]]]
    labels: list[int]
    rounds: list[RoundRecord]
    audit: dict


def discover_data_files(path: str | Path) -> list[Path]:
    """Find complete event exports in a file or nested dataset directory."""
    path = Path(path).resolve()
    if path.is_file() and path.suffix.lower() == ".jsonl":
        return [path]
    if not path.is_dir():
        raise ValueError(f"Expected a JSONL file or data directory: {path}")
    files = sorted(path.rglob("*.jsonl"))
    if any(not candidate.resolve().is_relative_to(path) for candidate in files):
        raise ValueError("Dataset symlinks must stay inside the selected directory")
    if not files:
        raise ValueError(f"No JSONL exports found in {path}")
    return files


def parse_timestamp(value: str) -> float:
    value = value[:-1] + "+00:00" if value.endswith("Z") else value
    # Python 3.9 accepts microseconds, but GRID may emit nanoseconds. Preserve
    # the timezone offset while normalizing only the fractional seconds.
    value = re.sub(r"\.(\d+)", lambda match: "." + match[1][:6].ljust(6, "0"), value, count=1)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("Event timestamps must include a timezone")
    return parsed.timestamp()


def iter_events(path: Path):
    previous_time = float("-inf")
    previous_sequence = -1
    seen_ids = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                envelope = json.loads(line)
                ts = parse_timestamp(envelope["occurredAt"])
                sequence = int(envelope["sequenceNumber"])
                series_id = str(envelope["seriesId"])
                if not series_id or ts < previous_time or sequence <= previous_sequence:
                    raise ValueError("Missing series ID or out-of-order envelope")
                previous_time, previous_sequence = ts, sequence
                for event in envelope["events"]:
                    event_id = event["id"]
                    if event_id in seen_ids:
                        raise ValueError(f"Duplicate event ID: {event_id}")
                    seen_ids.add(event_id)
                    yield ts, series_id, event
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path.name}:{line_number}: {exc}") from exc


def game_state(event: dict, game_id: str) -> dict | None:
    actor = event.get("actor", {})
    if actor.get("type") == "game" and str(actor.get("id")) == game_id:
        return actor.get("state")
    return next((g for g in event.get("seriesState", {}).get("games", [])
                 if str(g.get("id")) == game_id), None)


def extract_rounds(path: str | Path) -> tuple[list[RoundRecord], Counter]:
    rounds = []
    current = None
    excluded_games = set()
    audit = Counter()
    for ts, series_id, event in iter_events(Path(path)):
        kind = event.get("type", "")
        actor, target = event.get("actor", {}), event.get("target", {})
        if "!" in kind:
            # No speculative undo: exclude the affected game, including earlier rounds.
            ids = {str(g["id"]) for g in event.get("seriesState", {}).get("games", [])}
            ids.update(str(entity["id"]) for entity in (actor, target)
                       if entity.get("type") == "game")
            if current is not None:
                ids.add(current.game_id)
            if not ids:
                raise ValueError(f"Cannot associate correction {event['id']} with a game")
            excluded_games.update((series_id, game_id) for game_id in ids)
            audit["correction_events"] += 1
            continue
        if kind == "grid-invalidated-series":
            if any(r.series_id == series_id for r in rounds):
                raise ValueError("Series invalidated after completed rounds; review source data")
            if current is not None:
                excluded_games.add((series_id, current.game_id))
            audit["series_invalidations"] += 1
        if kind == "game-started-round":
            if current is not None:
                audit["unfinished_rounds"] += 1
            state = actor.get("state", {})
            teams = state.get("teams", [])
            attackers = [str(t["id"]) for t in teams if t.get("side") == "attacker"]
            defenders = [str(t["id"]) for t in teams if t.get("side") == "defender"]
            if len(attackers) != 1 or len(defenders) != 1 or attackers == defenders:
                raise ValueError(f"Missing or ambiguous round sides: {event['id']}")
            current = RoundRecord(series_id, str(actor["id"]), str(target["id"]),
                                  state.get("map", {}).get("name", "unknown"),
                                  attackers[0], defenders[0], state)
            audit["started_rounds"] += 1
        if current is None:
            continue
        if current.series_id != series_id:
            raise ValueError("Interleaved series are unsupported; use one series per file")
        if kind == "round-ended-freezetime":
            if str(actor.get("id")) != current.round_id or current.live_at is not None:
                raise ValueError("Mismatched or duplicate end of buy time")
            current.live_at = ts
            state = game_state(event, current.game_id)
            if state and state.get("teams"):
                current.start_state = state
        if current.live_at is not None:
            current.events.append((ts - current.live_at, event))
            state = game_state(event, current.game_id) or {}
            if state.get("paused") or kind in {"game-paused", "game-paused-clock"}:
                current.paused = True
            if kind.endswith(("-completed-defuseBomb", "-completed-explodeBomb")):
                current.terminal_at = ts if current.terminal_at is None else min(ts, current.terminal_at)
        if kind == "team-won-round":
            if str(target.get("id")) != current.round_id:
                raise ValueError("Winner event refers to a different round")
            winner = str(actor.get("id", ""))
            if winner not in {current.attacker_team_id, current.defender_team_id}:
                raise ValueError("Winner does not match either round team")
            current.atk_won = int(winner == current.attacker_team_id)
            current.ended_at = ts
            rounds.append(current)
            current = None
    if current is not None:
        audit["unfinished_rounds"] += 1
    audit["completed_rounds"] = len(rounds)
    clean = [r for r in rounds if (r.series_id, r.game_id) not in excluded_games]
    audit["rounds_excluded_for_corrections"] = len(rounds) - len(clean)
    audit["games_with_corrections_or_invalidation"] = len(excluded_games)
    return clean, audit


def team_of(entity: dict, record: RoundRecord) -> str:
    valid = {record.attacker_team_id, record.defender_team_id}
    candidates = [entity.get("id") if entity.get("type") == "team" else None,
                  entity.get("state", {}).get("teamId")]
    candidates.extend(n.get("id") for n in entity.get("statePath", []))
    for candidate in candidates:
        if candidate is not None and str(candidate) in valid:
            return str(candidate)
    for team in record.start_state.get("teams", []):
        if any(str(p.get("id")) == str(entity.get("id")) for p in team.get("players", [])):
            return str(team["id"])
    raise ValueError(f"Unknown team for entity {entity.get('id')}, round {record.key}")


def round_to_features(record: RoundRecord, observation_seconds: int) -> list[list[float]]:
    if not 1 <= observation_seconds <= 120:
        raise ValueError("observation_seconds must be between 1 and 120")
    if record.live_at is None or record.ended_at is None:
        raise ValueError("Round lacks live start or outcome")
    ending = min(record.ended_at, record.terminal_at or record.ended_at)
    if ending - record.live_at <= observation_seconds:
        raise ValueError("Round already ended by the observation cutoff")
    if record.paused:
        raise ValueError("Paused rounds are excluded from wall-clock windows")
    team_ids = [record.attacker_team_id, record.defender_team_id]
    loadout = [0.0, 0.0]
    alive = []
    for index, team_id in enumerate(team_ids):
        team = next(t for t in record.start_state["teams"] if str(t["id"]) == team_id)
        players = team.get("players", [])
        if len(players) != 5 or len({str(p["id"]) for p in players}) != 5:
            raise ValueError(f"Expected five distinct players for team {team_id}")
        alive.append({str(p["id"]) for p in players if p.get("alive", True)})
        loadout[index] = float(team["loadoutValue"]) / 25000.0
    kills, abilities = [0, 0], [0, 0]
    planted, first_known, first_atk = 0, 0, 0
    cursor, features = 0, []
    for second in range(observation_seconds):
        while cursor < len(record.events) and record.events[cursor][0] < second + 1:
            relative, event = record.events[cursor]
            cursor += 1
            if relative < 0:
                continue
            kind = event.get("type", "")
            actor, target = event.get("actor", {}), event.get("target", {})
            if kind in {"player-killed-player", "game-killed-player",
                        "player-selfkilled-player", "player-teamkilled-player"}:
                victim_index = team_ids.index(team_of(target, record))
                alive[victim_index].discard(str(target["id"]))
                if kind == "player-killed-player":
                    killer_index = team_ids.index(team_of(actor, record))
                    if killer_index != victim_index:
                        kills[killer_index] += 1
                        if not first_known:
                            first_known, first_atk = 1, int(killer_index == 0)
            elif kind == "player-used-ability":
                abilities[team_ids.index(team_of(actor, record))] += 1
            elif kind.endswith("-completed-plantBomb"):
                planted = 1
            elif "reviv" in kind.lower() or "respawn" in kind.lower():
                raise ValueError("Revive/respawn semantics need validation before using this round")
        row = [(second + 1) / observation_seconds, len(alive[0]) / 5,
               len(alive[1]) / 5, kills[0] / 5, kills[1] / 5, float(planted),
               *loadout, min(abilities[0] / 20, 2), min(abilities[1] / 20, 2),
               float(first_known), float(first_atk)]
        if not all(math.isfinite(value) for value in row):
            raise ValueError(f"Nonfinite feature in round {record.key}")
        features.append(row)
    return features


def load_dataset(paths: list[Path], observation_seconds: int = DEFAULT_OBSERVATION_SECONDS) -> Dataset:
    if not paths:
        raise ValueError("No JSONL files found")
    if not 1 <= observation_seconds <= 120:
        raise ValueError("observation_seconds must be between 1 and 120")
    features, labels, retained = [], [], []
    audit, seen, sources = Counter(), set(), []
    for path in sorted(paths):
        records, counts = extract_rounds(path)
        audit.update(counts)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        source_hash = digest.hexdigest()
        sources.append({"file": path.name, "sha256": source_hash})
        for record in records:
            record.source_sha256 = source_hash
            if record.key in seen:
                raise ValueError(f"Duplicate round across source files: {record.key}")
            seen.add(record.key)
            if record.live_at is None:
                audit["missing_live_start"] += 1
                continue
            if record.paused:
                audit["paused_rounds"] += 1
                continue
            ending = min(record.ended_at, record.terminal_at or record.ended_at)
            if ending - record.live_at <= observation_seconds:
                audit["ended_by_cutoff"] += 1
                continue
            features.append(round_to_features(record, observation_seconds))
            labels.append(record.atk_won)
            retained.append(record)
    audit["eligible_rounds"] = len(retained)
    audit["series"] = len({r.series_id for r in retained})
    audit["games"] = len({(r.series_id, r.game_id) for r in retained})
    return Dataset(features, labels, retained, {
        "counts": dict(audit), "sources": sources, "feature_names": FEATURE_NAMES,
        "observation_seconds": observation_seconds, "schema_version": SCHEMA_VERSION,
        "label_counts": dict(Counter(str(y) for y in labels)),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Local JSONL file or directory")
    parser.add_argument("--observation-seconds", type=int, default=DEFAULT_OBSERVATION_SECONDS)
    args = parser.parse_args()
    try:
        dataset = load_dataset(discover_data_files(args.data), args.observation_seconds)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Data error: {exc}\n")
    print(json.dumps(dataset.audit, indent=2))


if __name__ == "__main__":
    main()
