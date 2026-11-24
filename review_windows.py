"""Reconstruct study disagreement cases using JSON and the standard library only.

Default output withholds outcomes and probabilities for owner annotation. This
does not make the cases an unbiased sample: selection uses model disagreement.
No model artifacts are deserialized and no inference is performed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from data_processing import discover_data_files, extract_rounds
from replay_data import build_round_replay
from rolling_experiment import read_manifest, read_predictions, schema_hash
from rolling_features import FEATURE_NAMES, HISTORY_SECONDS, observe_round, preview_round, schema, xy
from rolling_inputs import mask_window, validate_input_policy


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def window_evidence(record, at, policy, focus_player_ids=()):
    """Expose only source events strictly before the requested observation end."""
    validate_input_policy(policy)
    observed = observe_round(record)
    if at not in observed["endpoints"]:
        raise ValueError("Review endpoint is not an eligible prediction window")
    start = at - HISTORY_SECONDS
    prefix = copy.deepcopy(record)
    prefix.events = [(t, e) for t, e in record.events if t < at]
    replay = build_round_replay(prefix)
    frame = replay["frames"][-1]
    window_ids = {e["id"] for t, e in prefix.events if start <= t < at}
    earlier_ids = {e["id"] for t, e in prefix.events if 0 <= t < start}
    preview = next(w for w in preview_round(prefix)["windows"] if w["at_seconds"] == at)
    rows = observed["rows"][start:at]
    masked = mask_window(rows, policy)
    last = dict(zip(FEATURE_NAMES, rows[-1]))
    state = {"attacker_alive": round(last["atk_alive"] * 5),
             "defender_alive": round(last["def_alive"] * 5),
             "spike_planted": bool(last["spike_planted"]),
             "plant_age_seconds": last["plant_age_45s"] * 45 if last["spike_planted"] else None,
             "defusing_reported": bool(last["defusing"]),
             "defuse_checkpoint_reported": bool(last["defuse_checkpoint"])}
    state["start_loadout"] = {side: round(last[f"{side}_start_loadout"] * 25000)
                              for side in ("atk", "def")}
    suppressed = []
    for i, (name, keep) in enumerate(zip(FEATURE_NAMES, policy["keep"])):
        values = [row[i] for row in rows]
        if not keep and any(values):
            suppressed.append({"name": name, "raw_minimum": min(values), "raw_maximum": max(values),
                               "raw_last": values[-1], "model_value": 0})
    focused = []
    for player_id in focus_player_ids:
        identity = next((p for p in frame["players"] if p["id"] == player_id), None)
        if identity is None:
            raise ValueError("Annotated player is absent from the round")
        reports, seen = [], set()
        for snapshot in replay["frames"]:
            p = next(p for p in snapshot["players"] if p["id"] == player_id)
            t = p["position_at"]
            if p["alive"] and p["position"] and start <= t < at and t not in seen:
                seen.add(t)
                reports.append({"at_seconds": t, "position": p["position"],
                                "normalized_minimap": xy(p["position"], schema()["calibration"].get(record.map_name)),
                                "gap_from_previous_report": t - reports[-1]["at_seconds"] if reports else None})
        focused.append({"player_id": player_id, "name": identity["name"], "side": identity["side"],
                        "position_reports": reports})
    return {"round": list(record.key), "map": record.map_name, "source_sha256": record.source_sha256,
            "at_seconds": at, "window_start": start, "interval": "[start,end)",
            "teams": [{k: team[k] for k in ("name", "side")} for team in replay["teams"]],
            "state": state, "counts": preview["counts"],
            "fresh_position_fraction": preview["fresh_position_fraction"],
            "movement": preview["movement"], "utility_categories": preview["utility_categories"],
            "events": [e for e in replay["events"] if e["id"] in window_ids],
            "earlier_kills_and_objectives": [e for e in replay["events"] if e["id"] in earlier_ids
                                             and e["category"] in ("kill", "objective")],
            "trade_evidence": [e for e in preview["kills"] if e["trade_delay"] is not None],
            "focused_player_reports": focused,
            "living_players": [{"name": p["name"], "side": p["side"], "agent": p["agent"],
                                "last_reported_position": p["position"],
                                "position_age_seconds": at - p["position_at"] if p["position"] else None}
                               for p in frame["players"] if p["alive"]],
            "input_support": {"retained_channels": sum(policy["keep"]),
                              "masked_channels": len(FEATURE_NAMES) - sum(policy["keep"]),
                              "nonzero_raw_channels_suppressed": suppressed,
                              "policy_sha256": digest(policy), "raw_window_sha256": digest(rows),
                              "model_window_sha256": digest(masked)}}


def build_review(data_dir, study_dir, study_result, reveal=False, annotations=None):
    result = json.loads(Path(study_result).read_text())
    protocol = result["protocol"]
    repo = Path(__file__).resolve().parent
    for name, expected in protocol["source_sha256"].items():
        # UI/server revisions do not change reconstructed evidence. Retain checks
        # for all study computation and map calibration used by this reviewer.
        # The map manifest's Ascent A/B display labels were corrected on 2026-11-23,
        # so its whole-file hash differs from the study's. Its calibration numbers
        # stay bound: read_manifest() below rejects any run whose contract, which
        # embeds them, differs from the current one.
        if name in {"serve.py", "viewer/app.js", "viewer/index.html", "viewer/maps/manifest.json"}:
            continue
        path = (repo / name).resolve()
        if not path.is_relative_to(repo) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Study implementation changed: {name}")
    records = {}
    for source in discover_data_files(data_dir):
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        for record in extract_rounds(source)[0]:
            if record.key in records:
                raise ValueError("Duplicate source round")
            record.source_sha256 = source_hash
            records[record.key] = record
    summaries = {r["name"]: r for r in result["runs"]}
    annotations = annotations or {"annotations": []}
    annotation_index = {(*a["round"], a["at_seconds"]): a for a in annotations["annotations"]}
    if len(annotation_index) != len(annotations["annotations"]):
        raise ValueError("Duplicate owner annotation")
    runs = {}
    for map_name in protocol["validation_maps"]:
        pair = []
        for architecture in protocol["architectures"]:
            name = f"{map_name}-all-{architecture}-seed42"
            directory = Path(study_dir) / name
            manifest, manifest_hash = read_manifest(directory / "run.json")
            if (manifest_hash != summaries[name]["manifest_sha256"]
                    or manifest["run_id"] != summaries[name]["run_id"]):
                raise ValueError("Run does not match the completed study")
            grouped, _ = read_predictions(directory / "predictions.jsonl", directory / "run.json")
            rows = {(*key, row["at_seconds"]): row for key, items in grouped.items() for row in items}
            runs[map_name, architecture] = (manifest, rows, {
                "run_name": name, "run_id": manifest["run_id"], "manifest_sha256": manifest_hash,
                "predictions_sha256": hashlib.sha256((directory / "predictions.jsonl").read_bytes()).hexdigest()})
            pair.append(manifest)
        for field in ("input_policy", "schema", "partitions", "implementation_sha256"):
            if pair[0][field] != pair[1][field]:
                raise ValueError(f"Comparison runs disagree on {field}")
    cases = []
    for candidate in result["inspection_candidates"]:
        record = records[tuple(candidate["round"])]
        at = candidate["at_seconds"]
        if record.map_name != candidate["map"] or record.source_sha256 != protocol["data_sha256"]:
            raise ValueError("Candidate source mismatch")
        manifest = runs[record.map_name, "temporal_cnn"][0]
        annotation = annotation_index.get((*record.key, at))
        case = window_evidence(record, at, manifest["input_policy"],
                               annotation.get("focus_player_ids", ()) if annotation else ())
        case["provenance"] = {}
        for architecture, probability_key in (("temporal_cnn", "cnn_probability"),
                                               ("pooled_mlp", "pooled_probability")):
            _, rows, provenance = runs[record.map_name, architecture]
            row = rows[(*record.key, at)]
            if (row["partition"] != "validation" or row["source_sha256"] != record.source_sha256
                    or row["probabilities"][architecture] != candidate[probability_key]):
                raise ValueError("Candidate must match a saved validation prediction")
            case["provenance"][architecture] = provenance
        case["partition"] = "validation"
        case["owner_annotation"] = annotation or {"favored_side": None, "reason": None, "missing_evidence": None}
        if reveal:
            case["retrospective"] = {"cnn_probability": candidate["cnn_probability"],
                                     "pooled_probability": candidate["pooled_probability"],
                                     "attacker_won": record.atk_won}
        cases.append(case)
    return {"schema_id": "disagreement-review-v1", "feature_schema_sha256": schema_hash(schema()),
            "annotations_sha256": digest(annotations),
            "study_result_sha256": hashlib.sha256(Path(study_result).read_bytes()).hexdigest(),
            "selection": "Seed 42, largest CNN/control disagreement per round, top three rounds per validation map; no winner filter.",
            "outcomes_and_probabilities_shown": reveal,
            "limitations": ["Development validation cases selected by disagreement, not representative or independent test data.",
                            "Names and agents identify review evidence; they are not model input channels.",
                            "Cast locations are caster positions, not verified effect locations or successful utility.",
                            "Spectator positions do not establish player knowledge, visibility, or a rotation's cause.",
                            "Position coverage measures fresh reported observations, not continuous tracking.",
                            "Counts and state show raw observations; training-constant channels are suppressed as listed."],
            "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--study-dir", type=Path, default=Path("runs/gameplay-v2-study"))
    parser.add_argument("--study-result", type=Path, default=Path("docs/study-12runs.json"))
    parser.add_argument("--annotations", type=Path, default=Path("docs/gameplay-annotations.json"))
    parser.add_argument("--reveal", action="store_true", help="Include saved probabilities and eventual winners")
    args = parser.parse_args()
    print(json.dumps(build_review(args.data_dir, args.study_dir, args.study_result, args.reveal,
                                  json.loads(args.annotations.read_text())),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
