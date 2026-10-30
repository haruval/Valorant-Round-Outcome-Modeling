"""Training-only input support and feature-family controls; standard library only."""
import hashlib
import json
import math

from rolling_features import FEATURE_NAMES, GLOBAL_NAMES, HISTORY_SECONDS, schema

FEATURE_SETS = ("all", "state_only", "no_utility", "no_spatial", "no_trades")


def contract_hash():
    return hashlib.sha256(json.dumps(schema(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def selected_channels(feature_set):
    if feature_set not in FEATURE_SETS:
        raise ValueError("Unknown feature set")
    minimal = set(GLOBAL_NAMES) | {f"{side}_{name}" for side in ("atk", "def")
                                  for name in ("alive", "start_loadout", "alive_loadout")}
    selected = []
    for name in FEATURE_NAMES:
        utility = any(part in name for part in ("_utility_", "_abilities_", "_cast_", "_after_recon_"))
        spatial = (any(part in name for part in ("_cell_", "_position_", "_centroid_", "_spread",
                                                "_movement_", "_origin_", "_victim_"))
                   or name.endswith(("_x", "_y", "_located")))
        selected.append(feature_set == "all" or
                        (feature_set == "state_only" and name in minimal) or
                        (feature_set == "no_utility" and not utility) or
                        (feature_set == "no_spatial" and not spatial) or
                        (feature_set == "no_trades" and "_trade_" not in name))
    return selected


def checked_rows(windows):
    for window in windows:
        if len(window) != HISTORY_SECONDS:
            raise ValueError("Expected a 20-bin rolling window")
        for row in window:
            if len(row) != len(FEATURE_NAMES) or any(type(v) not in (int, float) or not math.isfinite(v) for v in row):
                raise ValueError("Malformed rolling inputs")
            yield row


def derive_input_policy(training_windows, feature_set="all"):
    """Do not pass validation/test windows. Drop every training-constant channel.

    A zero input at inference prevents unseen values from activating untrained
    CNN weights. The same support mask and ablations apply to every model.
    """
    selected = selected_channels(feature_set)
    low, high, count = [math.inf] * len(FEATURE_NAMES), [-math.inf] * len(FEATURE_NAMES), 0
    for row in checked_rows(training_windows):
        count += 1
        for i, value in enumerate(row):
            low[i], high[i] = min(low[i], value), max(high[i], value)
    if not count:
        raise ValueError("Cannot derive input support without training observations")
    return {"id": "training-support-v1", "schema_sha256": contract_hash(),
            "feature_set": feature_set, "feature_names": list(FEATURE_NAMES),
            "training_bins": count, "minimum": low, "maximum": high,
            "keep": [allowed and a < b for allowed, a, b in zip(selected, low, high)],
            "constant_policy": "replace with zero in all partitions"}


def validate_input_policy(policy):
    if (not isinstance(policy, dict) or policy.get("id") != "training-support-v1"
            or policy.get("schema_sha256") != contract_hash() or policy.get("feature_names") != FEATURE_NAMES
            or policy.get("constant_policy") != "replace with zero in all partitions"
            or type(policy.get("training_bins")) is not int or policy["training_bins"] <= 0
            or policy["training_bins"] % HISTORY_SECONDS):
        raise ValueError("Incompatible input policy")
    selected = selected_channels(policy.get("feature_set"))
    for key in ("minimum", "maximum", "keep"):
        if not isinstance(policy.get(key), list) or len(policy[key]) != len(FEATURE_NAMES):
            raise ValueError("Malformed input policy width")
    for allowed, low, high, keep in zip(selected, policy["minimum"], policy["maximum"], policy["keep"]):
        if (type(low) not in (int, float) or type(high) not in (int, float)
                or not math.isfinite(low) or not math.isfinite(high) or low > high
                or type(keep) is not bool or keep != (allowed and low < high)):
            raise ValueError("Malformed input support mask")


def mask_window(window, policy):
    # The caller validates the policy once; checked_rows still checks every input.
    return [[value if keep else 0.0 for value, keep in zip(row, policy["keep"])]
            for row in checked_rows([window])]


def support_report(windows, policy):
    """Report shift without modifying the training-fitted policy."""
    validate_input_policy(policy)
    active = set()
    for row in checked_rows(windows):
        active.update(i for i, value in enumerate(row) if policy["minimum"][i] == policy["maximum"][i]
                      and value != policy["minimum"][i])
    return {"masked_channels": [n for n, keep in zip(FEATURE_NAMES, policy["keep"]) if not keep],
            "constant_training_channels_changed": [FEATURE_NAMES[i] for i in sorted(active)],
            "handling": "always masked; never expand support using evaluation data"}
