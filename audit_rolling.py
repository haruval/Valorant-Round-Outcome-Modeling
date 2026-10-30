"""Cross-check rolling event channels against source events; no ML runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from data_processing import discover_data_files, extract_rounds
from rolling_features import FEATURE_NAMES, HISTORY_SECONDS, TRADE_MAX_SECONDS, observe_round, schema
from ability_taxonomy import CATEGORIES, ABILITY_TAGS


def event_ledger(record):
    """Use roster membership and raw event identities, not projected evidence."""
    roster = {str(p["id"]): {"name": p.get("name", str(p["id"])),
                             "side": "atk" if str(team["id"]) == record.attacker_team_id else "def"}
              for team in record.start_state["teams"] for p in team["players"]}
    ledger = []
    for at, event in record.events:
        if at < 0:
            continue
        kind = event["type"]
        actor = str(event.get("actor", {}).get("id"))
        target = str(event.get("target", {}).get("id"))
        item = {"at_seconds": at, "event_id": event["id"], "actor_id": actor,
                "actor": roster.get(actor, {}).get("name", actor), "target_id": target}
        if kind == "player-used-ability":
            if actor not in roster:
                raise ValueError("Ability actor is absent from the round roster")
            item.update(kind="ability", side=roster[actor]["side"], ability=target)
        elif kind in {"player-killed-player", "game-killed-player", "player-selfkilled-player", "player-teamkilled-player"}:
            if target not in roster:
                raise ValueError("Death victim is absent from the round roster")
            enemy_kill = kind == "player-killed-player" and actor in roster and roster[actor]["side"] != roster[target]["side"]
            item.update(kind="kill" if enemy_kill else "death", target=roster[target]["name"],
                        victim_side=roster[target]["side"], side=roster[actor]["side"] if enemy_kill else None)
        else:
            continue
        ledger.append(item)
    kills = [item for item in ledger if item["kind"] == "kill"]
    for item in kills:
        # Independently search raw kill pairs. Exactly simultaneous events are excluded.
        previous = [other for other in kills if other["actor_id"] == item["target_id"]
                    and other["victim_side"] == item["side"]
                    and 0 < item["at_seconds"] - other["at_seconds"] <= TRADE_MAX_SECONDS]
        original = max(previous, key=lambda other: other["at_seconds"]) if previous else None
        item["trade_delay"] = item["at_seconds"] - original["at_seconds"] if original else None
        item["traded_event_id"] = original["event_id"] if original else None
    return ledger


def count_channels(events):
    expected = Counter()
    for item in events:
        side = item["side"]
        if item["kind"] == "ability":
            expected[f"{side}_abilities_1s"] += 1
            for category in ABILITY_TAGS.get(item["ability"], ("other",)):
                expected[f"{side}_utility_{category}_uses"] += 1
        else:
            expected[f"{item['victim_side']}_deaths_1s"] += 1
            if item["kind"] == "kill":
                expected[f"{side}_kills_1s"] += 1
                if item["trade_delay"] is not None:
                    expected[f"{side}_trade_candidates_1s"] += 1
                    expected[f"{side}_trade_delay_5s"] += item["trade_delay"] / TRADE_MAX_SECONDS
    return expected


def check_record(record, observed=None):
    observed = observe_round(record) if observed is None else observed
    ledger = event_ledger(record)
    channels = [f"{side}_{field}" for side in ("atk", "def")
                for field in ["abilities_1s", "kills_1s", "deaths_1s", "trade_candidates_1s", "trade_delay_5s"]
                + [f"utility_{category}_uses" for category in CATEGORIES]]
    columns = {name: FEATURE_NAMES.index(name) for name in channels}
    failures, windows = [], []
    for at in observed["endpoints"]:
        selected = [item for item in ledger if at - HISTORY_SECONDS <= item["at_seconds"] < at]
        expected = count_channels(selected)
        for name, column in columns.items():
            actual = sum(row[column] for row in observed["rows"][at - HISTORY_SECONDS:at])
            if not math.isclose(actual, expected[name], rel_tol=1e-9, abs_tol=1e-9):
                failures.append({"round": list(record.key), "at_seconds": at, "feature": name,
                                 "expected_from_events": expected[name], "actual": actual})
        windows.append({"at_seconds": at, "window_start": at - HISTORY_SECONDS,
                        "ability_uses": sum(i["kind"] == "ability" for i in selected),
                        "enemy_kills": sum(i["kind"] == "kill" for i in selected),
                        "trade_candidates": sum(i.get("trade_delay") is not None for i in selected)})
    return {"ledger": ledger, "windows": windows, "failures": failures,
            "checks": len(columns) * len(windows)}


def audit(paths):
    totals, failures, sources, examples, seen = Counter(), [], [], [], set()
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sources.append({"file": path.name, "sha256": digest})
        records, _ = extract_rounds(path)
        for record in records:
            if record.key in seen:
                raise ValueError("Duplicate round across audit sources")
            seen.add(record.key)
            if record.paused or record.live_at is None:
                totals["skipped_paused_or_missing_start"] += 1
                continue
            checked = check_record(record)
            totals["rounds"] += 1
            totals["windows"] += len(checked["windows"])
            totals["channel_comparisons"] += checked["checks"]
            totals["source_ability_events"] += sum(i["kind"] == "ability" for i in checked["ledger"])
            totals["source_enemy_kills"] += sum(i["kind"] == "kill" for i in checked["ledger"])
            totals["source_trade_candidates"] += sum(i.get("trade_delay") is not None for i in checked["ledger"])
            failures.extend(checked["failures"])
            example_times = ({20, 40} if record.round_id == "round-1" and record.map_name in {"breeze", "ascent"}
                             else {105} if record.map_name == "breeze" and record.round_id == "round-23" else set())
            if example_times:
                for window in checked["windows"]:
                    if window["at_seconds"] not in example_times:
                        continue
                    events = [i for i in checked["ledger"] if window["window_start"] <= i["at_seconds"] < window["at_seconds"]]
                    examples.append({"round": list(record.key), "map": record.map_name,
                                     "source_sha256": digest, **window, "events": events})
    return {"audit_version": 2, "feature_schema": schema()["id"], "status": "pass" if not failures else "fail", "sources": sources,
            "counts": dict(totals), "trade_max_seconds": TRADE_MAX_SECONDS,
            "scope": "Event counts, shared ability tags and return-kill timing in every eligible window. "
                     "Does not validate tactical labels, position calibration, effect success or ML performance.",
            "failures": failures, "review_examples": examples}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    result = audit(discover_data_files(args.data))
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result["status"] == "pass" else 1)
