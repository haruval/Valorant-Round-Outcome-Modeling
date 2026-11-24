import hashlib
import json
import unittest
from pathlib import Path

from structure_inputs import ARMS, extension_contract
from rolling_experiment import schema_hash

REPO = Path(__file__).resolve().parents[1]
PLAN = json.loads((REPO / "docs" / "utility-structure-plan.json").read_text())
PYTHON = "runs/ml-validation-20261029/venv/bin/python"


class StructurePlanTests(unittest.TestCase):
    def test_execution_needs_a_recorded_owner_approval(self):
        approved = PLAN["execution_authorized"]
        self.assertIs(type(approved), bool)
        self.assertEqual(approved, PLAN["status"] != "draft_not_approved")
        self.assertEqual(approved, bool(PLAN["approval_date"]) and bool(PLAN["authorization"]))

    def test_run_matrix_and_commands_are_exactly_the_planned_ones(self):
        expected = {("arms" if arm != "base" else "parity", arm, m, a, seed)
                    for arm in ("base", "structure", "placebo") for m in ("ascent", "breeze")
                    for a in ("temporal_cnn", "pooled_mlp") for seed in (42, 43, 44)}
        runs = PLAN["runs"]
        self.assertEqual({(r["stage"], r["arm"], r["validation_map"], r["architecture"], r["model_seed"]) for r in runs},
                         expected)
        self.assertEqual((len(runs), PLAN["run_count"], PLAN["total_epochs"]), (36, 36, 720))
        controls = {c["name"] for c in PLAN["controls"]}
        for r in runs:
            name = f"{r['validation_map']}-{r['arm']}-{r['architecture']}-seed{r['model_seed']}"
            output = f"runs/utility-structure-v1/{name}"
            self.assertEqual((r["name"], r["output_dir"], r["epochs"]), (name, output, 20))
            self.assertEqual(r["train_argv"], [
                PYTHON, "-E", "-s", "-B", "train_structure.py", "--data-dir", "data", "--output-dir", output,
                "--smoke-test", "--validation-map", r["validation_map"], "--split-seed", "42",
                "--model-seed", str(r["model_seed"]), "--feature-set", "no_spatial", "--arm", r["arm"],
                "--neural-model", r["architecture"], "--epochs", "20"])
            self.assertEqual(r["reexport_argv"], [PYTHON, "-E", "-s", "-B", "predict_structure.py", output, "data"])
            self.assertEqual(r["control"], f"{r['validation_map']}-no_spatial-{r['architecture']}-seed{r['model_seed']}")
            self.assertIn(r["control"], controls)
            self.assertEqual(r["must_reproduce_control"], r["arm"] == "base")
            self.assertEqual(r["expected_retained_extension_channels"], ARMS[r["arm"]])
        self.assertEqual(len(controls), 12)

    def test_pending_plan_is_bound_to_the_current_sources(self):
        # Once a plan has been executed it describes that snapshot and is no longer rechecked.
        if PLAN["status"] not in ("draft_not_approved", "approved_not_executed"):
            self.skipTest("Plan already executed")
        for name, expected in PLAN["source_sha256"].items():
            path = (REPO / name).resolve()
            self.assertTrue(path.is_relative_to(REPO))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected, name)
        self.assertEqual(PLAN["extension_sha256"], schema_hash(extension_contract()))
        audit = REPO / PLAN["support_audit"]["file"]
        self.assertEqual(hashlib.sha256(audit.read_bytes()).hexdigest(), PLAN["support_audit"]["sha256"])


if __name__ == "__main__":
    unittest.main()
