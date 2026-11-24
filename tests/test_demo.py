"""The saved-prediction demo shipped with the repository must load as pinned; JSON only."""
import json
import unittest
from pathlib import Path

from comparison_data import load_comparisons

REPO = Path(__file__).resolve().parents[1]


class DemoTests(unittest.TestCase):
    def test_shipped_demo_loads_with_pinned_hashes_and_only_json(self):
        config, loaded = load_comparisons(REPO / "demo" / "comparison.json", REPO)
        self.assertEqual(len(loaded), 8)
        self.assertEqual({run["run"]["validation_map"] for run in loaded}, {"ascent", "breeze"})
        self.assertEqual({(run["run"]["feature_set"], run["architecture"]) for run in loaded},
                         {(f, a) for f in ("all", "no_spatial") for a in ("temporal_cnn", "pooled_mlp")})
        self.assertTrue(all(run["run"]["model_seed"] == 42 for run in loaded))
        self.assertTrue(all(entry["directory"].startswith("demo/") for entry in config["runs"]))
        for run in loaded:  # Every prediction shown is a validation prediction for that run's map.
            rows = [row for group in run["rows"].values() for row in group]
            self.assertEqual(len(rows), 555)
            self.assertIn(sum(row["partition"] == "validation" for row in rows), (269, 286))
        files = sorted(p.relative_to(REPO / "demo").as_posix() for p in (REPO / "demo").rglob("*")
                       if p.is_file() and not p.name.startswith("."))
        expected = ["README.md", "comparison.json"] + [f"{entry['id']}/{name}" for entry in config["runs"]
                                                       for name in ("predictions.jsonl", "run.json")]
        self.assertEqual(files, sorted(expected))


if __name__ == "__main__":
    unittest.main()
