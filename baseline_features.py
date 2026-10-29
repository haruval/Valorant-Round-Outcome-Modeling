"""Fixed temporal summaries of the same binned observations used by the CNN."""
from __future__ import annotations

import math

from data_processing import FEATURE_NAMES

TEMPORAL_NAMES = [
    "first_kill_time_fraction", "elimination_observed",
    "time_since_last_elimination_fraction", "time_since_plant_fraction",
    "atk_kills_last_5s", "def_kills_last_5s",
    "atk_ability_uses_last_5s", "def_ability_uses_last_5s",
]
SUMMARY_FEATURE_NAMES = FEATURE_NAMES + TEMPORAL_NAMES


def temporal_summary(window: list[list[float]]) -> list[float]:
    """Append eight timing features to the cutoff snapshot; no outcome is read.

    Event times are represented by the END of their one-second bin. Unknown
    times use zero, disambiguated by first_kill_known, elimination_observed,
    and spike_planted. Recent counts retain the sequence's fixed normalizations.
    """
    if not window or any(len(row) != len(FEATURE_NAMES) or
                         any(not math.isfinite(v) for v in row) for row in window):
        raise ValueError("Expected a nonempty, finite sequence with the current feature schema")
    columns = {name: i for i, name in enumerate(FEATURE_NAMES)}
    cutoff = len(window)
    first_kill, last_elimination, plant = None, None, None
    previous_alive = 2.0  # Each side's roster is normalized by five.
    for endpoint, row in enumerate(window, 1):
        if first_kill is None and row[columns["first_kill_known"]]:
            first_kill = endpoint
        alive = row[columns["atk_alive"]] + row[columns["def_alive"]]
        if alive < previous_alive:
            last_elimination = endpoint
        previous_alive = alive
        if plant is None and row[columns["spike_planted"]]:
            plant = endpoint
    before_recent = window[cutoff - 6] if cutoff > 5 else [0.0] * len(FEATURE_NAMES)
    recent = [window[-1][columns[name]] - before_recent[columns[name]] for name in
              ("atk_kills", "def_kills", "atk_ability_uses", "def_ability_uses")]
    return list(window[-1]) + [
        first_kill / cutoff if first_kill is not None else 0.0,
        float(last_elimination is not None),
        (cutoff - last_elimination) / cutoff if last_elimination is not None else 0.0,
        (cutoff - plant) / cutoff if plant is not None else 0.0,
        *recent,
    ]
