"""Inventory information/visibility signals in local exports; standard library only."""
import argparse
import hashlib
import json
import re
from collections import Counter

from data_processing import discover_data_files

SIGNALS = re.compile(r"visib|spott|reveal|line.?of.?sight|sight|heard|sound|audib|blind|suppress|detain|orient|rotation|angle|facing", re.I)


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        yield from value
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


def audit(data):
    fields, events = Counter(), Counter()
    declared, paths = set(), set()
    sources, envelopes = [], 0

    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                fields[key] += 1
                if key == "supportedEventTypes":
                    declared.update(strings(child))
                if key == "statePath":
                    paths.update(strings(child))
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for source in discover_data_files(data):
        sources.append({"file": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
        with source.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                value = json.loads(line)
                envelopes += 1
                walk(value)
                events.update(event["type"] for event in value["events"])
    return {"schema_id": "visibility-inventory-v1", "sources": sources,
            "scope": "All source envelopes, including corrected/invalidated segments; not eligible training-event totals.",
            "envelopes": envelopes, "events": sum(events.values()), "search_pattern": SIGNALS.pattern,
            "matching_field_counts": {k: fields[k] for k in sorted(fields) if SIGNALS.search(k)},
            "matching_state_paths": sorted(p for p in paths if SIGNALS.search(p)),
            "matching_observed_event_types": sorted(k for k in events if SIGNALS.search(k)),
            "matching_declared_event_types": sorted(k for k in declared if SIGNALS.search(k)),
            "field_names": sorted(fields), "event_type_counts": dict(sorted(events.items())),
            "declared_event_types": sorted(declared),
            "interpretation_limit": "A field-name inventory cannot establish gameplay visibility or the semantics of undocumented provider fields."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", default="data")
    print(json.dumps(audit(parser.parse_args().data), indent=2, allow_nan=False))
