"""Audit training-only support in both map directions; no ML imports or fitting."""
import argparse
import json
from pathlib import Path

from data_processing import discover_data_files
from rolling_features import load_rolling_dataset, schema
from rolling_inputs import derive_input_policy, support_report


def audit(data):
    dataset = load_rolling_dataset(discover_data_files(data))
    if {r.map_name for r in dataset.records} != {"breeze", "ascent"} or dataset.audit["counts"]["maps"] != 2:
        raise ValueError("This diagnostic requires exactly the two development maps")
    directions = {}
    for validation_map in ("ascent", "breeze"):
        train = [w for r, w in zip(dataset.records, dataset.features) if r.map_name != validation_map]
        validation = [w for r, w in zip(dataset.records, dataset.features) if r.map_name == validation_map]
        policy = derive_input_policy(train)
        directions[validation_map] = {"training_windows": len(train), "validation_windows": len(validation),
                                      "retained_channels": sum(policy["keep"]),
                                      **support_report(validation, policy)}
    return {"schema": schema(), "sources": dataset.audit["sources"], "counts": dataset.audit["counts"],
            "status": "feature support audited; no models fitted", "validation_map": directions}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    print(json.dumps(audit(parser.parse_args().data), indent=2, allow_nan=False))
