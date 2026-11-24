"""Utility-structure channels beside the frozen v2 inputs; standard library only.

The 222-channel `rolling-gameplay-v2` contract, its loader and its input policy are
used unchanged. Extension channels travel in a second array aligned row for row.
With every extension channel masked (the `base` arm) an extended run is built to
reproduce a `train_rolling.py` run exactly.

Fitting baselines still requires approval to execute ML dependencies.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import utility_structure
from data_processing import discover_data_files
from rolling_experiment import schema_hash, summary
from rolling_features import FEATURE_NAMES, HISTORY_SECONDS, load_rolling_dataset, schema
from rolling_inputs import derive_input_policy, mask_window, validate_input_policy
from utility_structure import FAMILIES, extract_casts, span, structure_bin, support_tags

EXTENSION_ID = "utility-structure-extension-v1"
PLACEBO_NAMES = ["atk_support_casts_live", "def_support_casts_live"]
EXTENSION_NAMES = utility_structure.FEATURE_NAMES + PLACEBO_NAMES
ARMS = {"base": [], "structure": FAMILIES["structure"], "placebo": PLACEBO_NAMES,
        "structure_answer": FAMILIES["structure"] + FAMILIES["answer_after_opposing_block"],
        "structure_spacing": FAMILIES["structure"] + FAMILIES["caster_spacing_proxy"]}
BASELINES = ("snapshot_logistic", "history_logistic")
ATTACKER_ALIVE = FEATURE_NAMES.index("atk_alive")


def extension_contract():
    return {"id": EXTENSION_ID, "base_schema_sha256": schema_hash(schema()),
            "feature_names": list(EXTENSION_NAMES), "structure": utility_structure.contract(),
            "placebo": {"names": PLACEBO_NAMES,
                        "definition": "Peak count of live support casts per side within the second, divided by 5. "
                                      "Same spans as the structure channels, without who cast or which function."},
            "arms": {name: list(channels) for name, channels in ARMS.items()},
            "alignment": "Row t of a window pairs with v2 row t: the one-second bin ending at t+1"}


def placebo_bin(casts, end):
    """How many support casts are live, ignoring caster and function. A volume control."""
    instants = sorted({end - 1.0} | {c["t"] for c in casts if end - 1 <= c["t"] < end})
    peaks = []
    for side in ("atk", "def"):
        mine = [c for c in casts if c["side"] == side and support_tags(c)]
        peaks.append(max(sum(c["t"] <= now and now - c["t"] < span(c) for c in mine) for now in instants) / 5)
    return peaks


def extension_windows(records, endpoints):
    """One 20-row extension window per v2 window, in the same order."""
    latest, bins, windows = {}, {}, []
    for record, at in zip(records, endpoints):
        latest[record.key] = max(at, latest.get(record.key, 0))
    for record, at in zip(records, endpoints):
        if type(at) is not int or at < HISTORY_SECONDS:
            raise ValueError("Extension windows need whole-second endpoints with a full history")
        if record.key not in bins:
            casts = extract_casts(record, latest[record.key])
            bins[record.key] = [structure_bin(casts, end)[0] + placebo_bin(casts, end)
                                for end in range(1, latest[record.key] + 1)]
        windows.append(bins[record.key][at - HISTORY_SECONDS:at])
    return windows


def checked_extension(windows):
    for window in windows:
        if len(window) != HISTORY_SECONDS:
            raise ValueError("Expected a 20-bin extension window")
        for row in window:
            if len(row) != len(EXTENSION_NAMES) or any(type(v) not in (int, float) or not math.isfinite(v) for v in row):
                raise ValueError("Malformed extension inputs")
            yield row


def derive_extension_policy(training_windows, arm):
    """Do not pass validation windows. Keep only this arm's channels that vary in training."""
    if arm not in ARMS:
        raise ValueError("Unknown extension arm")
    low, high, count = [math.inf] * len(EXTENSION_NAMES), [-math.inf] * len(EXTENSION_NAMES), 0
    for row in checked_extension(training_windows):
        count += 1
        for i, value in enumerate(row):
            low[i], high[i] = min(low[i], value), max(high[i], value)
    if not count:
        raise ValueError("Cannot derive extension support without training observations")
    return {"id": "extension-support-v1", "extension_sha256": schema_hash(extension_contract()), "arm": arm,
            "feature_names": list(EXTENSION_NAMES), "training_bins": count, "minimum": low, "maximum": high,
            "keep": [name in ARMS[arm] and a < b for name, a, b in zip(EXTENSION_NAMES, low, high)],
            "constant_policy": "replace with zero in all partitions"}


def validate_extension_policy(policy):
    if (not isinstance(policy, dict) or policy.get("id") != "extension-support-v1"
            or policy.get("extension_sha256") != schema_hash(extension_contract())
            or policy.get("feature_names") != EXTENSION_NAMES or policy.get("arm") not in ARMS
            or policy.get("constant_policy") != "replace with zero in all partitions"
            or type(policy.get("training_bins")) is not int or policy["training_bins"] <= 0
            or policy["training_bins"] % HISTORY_SECONDS):
        raise ValueError("Incompatible extension policy")
    for key in ("minimum", "maximum", "keep"):
        if not isinstance(policy.get(key), list) or len(policy[key]) != len(EXTENSION_NAMES):
            raise ValueError("Malformed extension policy width")
    for name, low, high, keep in zip(EXTENSION_NAMES, policy["minimum"], policy["maximum"], policy["keep"]):
        if (type(low) not in (int, float) or type(high) not in (int, float)
                or not math.isfinite(low) or not math.isfinite(high) or low > high
                or type(keep) is not bool or keep != (name in ARMS[policy["arm"]] and low < high)):
            raise ValueError("Malformed extension support mask")


def extension_support(windows, policy):
    """Report shift without changing the training-fitted policy."""
    validate_extension_policy(policy)
    active = set()
    for row in checked_extension(windows):
        active.update(i for i, value in enumerate(row) if policy["minimum"][i] == policy["maximum"][i]
                      and value != policy["minimum"][i])
    return {"kept_channels": [n for n, keep in zip(EXTENSION_NAMES, policy["keep"]) if keep],
            "constant_training_channels_changed": [EXTENSION_NAMES[i] for i in sorted(active)],
            "handling": "always masked; never expand support using evaluation data"}


def baseline_rows(window, extension, bundle):
    """Frozen v2 baseline inputs, followed by the kept extension columns only.

    With no extension column kept these rows equal the frozen baselines' inputs.
    """
    base = mask_window(window, bundle["input_policy"])
    keep = bundle["extension_policy"]["keep"]
    extra = [[value for value, kept in zip(row, keep) if kept] for row in checked_extension([extension])]
    means = [sum(row[i] for row in extra) / len(extra) for i in range(sum(keep))]
    return {"snapshot_logistic": base[-1] + extra[-1], "history_logistic": summary(base) + extra[-1] + means}


def validate_baselines(bundle):
    validate_input_policy(bundle.get("input_policy"))
    validate_extension_policy(bundle.get("extension_policy"))
    if (bundle.get("schema") != schema() or set(bundle.get("models", {})) != set(BASELINES)
            or type(bundle.get("train_prior")) not in (int, float) or not 0 <= bundle["train_prior"] <= 1):
        raise ValueError("Incompatible extended baseline schema")
    kept = sum(bundle["extension_policy"]["keep"])
    for name, model in bundle["models"].items():
        width = (len(FEATURE_NAMES) + kept) * (2 if name == "history_logistic" else 1)
        for key in ("mean", "scale", "coefficients"):
            if (not isinstance(model.get(key), list) or len(model[key]) != width
                    or any(type(v) not in (float, int) or not math.isfinite(v) for v in model[key])):
                raise ValueError("Malformed extended coefficients")
        if (any(v <= 0 for v in model["scale"]) or type(model.get("intercept")) not in (float, int)
                or not math.isfinite(model["intercept"])):
            raise ValueError("Invalid extended scaling/intercept")


def fit_extended_baselines(windows, extensions, labels, weights, seed, input_policy, extension_policy):
    """Same estimators and settings as rolling_experiment.fit_baselines."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    validate_input_policy(input_policy)
    validate_extension_policy(extension_policy)
    bundle = {"schema": schema(), "input_policy": input_policy, "extension_policy": extension_policy,
              "train_prior": sum(y * w for y, w in zip(labels, weights)) / sum(weights), "models": {}}
    rows = [baseline_rows(window, extension, bundle) for window, extension in zip(windows, extensions)]
    for name in BASELINES:
        matrix = np.asarray([row[name] for row in rows], dtype=float)
        scaler = StandardScaler().fit(matrix, sample_weight=weights)
        estimator = LogisticRegression(C=1, max_iter=3000, random_state=seed)
        estimator.fit(scaler.transform(matrix), labels, sample_weight=weights)
        if estimator.classes_.tolist() != [0, 1]:
            raise ValueError("Training requires both outcome classes")
        bundle["models"][name] = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                                  "coefficients": estimator.coef_[0].tolist(), "intercept": float(estimator.intercept_[0])}
    return bundle


def load_extended_baselines(path):
    path = Path(path)
    if path.stat().st_size > 500_000:
        raise ValueError("Extended baseline file exceeds size limit")
    bundle = json.loads(path.read_text())
    validate_baselines(bundle)
    return bundle


def extended_baseline_probabilities(bundle, windows, extensions):
    """Pure-Python inference; the arithmetic mirrors rolling_experiment.baseline_probabilities."""
    validate_baselines(bundle)
    if len(windows) != len(extensions):
        raise ValueError("Windows and extension windows must align")
    probabilities = {name: [] for name in ("train_prior",) + BASELINES}
    for window, extension in zip(windows, extensions):
        rows = baseline_rows(window, extension, bundle)
        probabilities["train_prior"].append(bundle["train_prior"])
        for name in BASELINES:
            model = bundle["models"][name]
            z = model["intercept"] + sum((x - mu) / scale * coef for x, mu, scale, coef in
                                         zip(rows[name], model["mean"], model["scale"], model["coefficients"]))
            if not math.isfinite(z):
                raise ValueError("Nonfinite extended prediction")
            probabilities[name].append(1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z)))
    return probabilities


def living_attacker(features, indices):
    """Windows whose final bin still has an attacker alive (owner decision: report both strata)."""
    return [i for i in indices if features[i][-1][ATTACKER_ALIVE] > 0]


def validate_extension_block(manifest):
    block = manifest.get("extension")
    if (not isinstance(block, dict) or set(block) != {"contract", "policy", "arm", "architecture"}
            or block["contract"] != extension_contract() or block["arm"] not in ARMS):
        raise ValueError("Run manifest lacks a compatible extension block")
    validate_extension_policy(block["policy"])
    if block["policy"]["arm"] != block["arm"]:
        raise ValueError("Extension arm differs from its policy")
    return block


def audit(data, feature_set="no_spatial"):
    """Training-only support of every arm in both map directions; nothing is fitted."""
    dataset = load_rolling_dataset(discover_data_files(data))
    if {r.map_name for r in dataset.records} != {"breeze", "ascent"}:
        raise ValueError("This diagnostic requires exactly the two development maps")
    extensions = extension_windows(dataset.records, dataset.endpoints)
    directions = {}
    for validation_map in ("ascent", "breeze"):
        train = [i for i, r in enumerate(dataset.records) if r.map_name != validation_map]
        validation = [i for i, r in enumerate(dataset.records) if r.map_name == validation_map]
        base = derive_input_policy([dataset.features[i] for i in train], feature_set)
        arms = {}
        for arm in ARMS:
            policy = derive_extension_policy([extensions[i] for i in train], arm)
            arms[arm] = {"retained_extension_channels": sum(policy["keep"]),
                         **extension_support([extensions[i] for i in validation], policy)}
        directions[validation_map] = {
            "training_windows": len(train), "validation_windows": len(validation),
            "validation_windows_living_attacker": len(living_attacker(dataset.features, validation)),
            "validation_rounds": len({dataset.records[i].key for i in validation}),
            "retained_base_channels": sum(base["keep"]), "arms": arms}
    return {"extension": extension_contract(), "extension_sha256": schema_hash(extension_contract()),
            "base_feature_set": feature_set, "sources": dataset.audit["sources"],
            "counts": dataset.audit["counts"], "status": "extension support audited; no models fitted",
            "validation_map": directions}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    print(json.dumps(audit(parser.parse_args().data), indent=2, allow_nan=False))
