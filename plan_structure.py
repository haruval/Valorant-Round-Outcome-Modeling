"""Draft the hash-bound utility-structure run plan; standard library only.

Rewrites docs/structure-support.json and docs/utility-structure-plan.json from the
current sources and the saved control runs. Nothing here executes ML code, and an
approved plan is never overwritten. Drafting a plan does not authorize running it.
"""
import hashlib
import json
from pathlib import Path

from structure_inputs import audit

repo = Path(__file__).resolve().parent
target = repo / "docs/utility-structure-plan.json"
if target.exists() and json.loads(target.read_text())["status"] != "draft_not_approved":
    raise SystemExit("Refusing to overwrite a plan that is no longer a draft")
(repo / "docs/structure-support.json").write_text(json.dumps(audit(repo / "data"), indent=2, allow_nan=False) + "\n")
support = json.loads((repo / "docs/structure-support.json").read_text())
PY = "runs/ml-validation-20261029/venv/bin/python"
ROOT = "runs/utility-structure-v1"
SOURCES = [
    "ability_taxonomy.py", "data_processing.py", "evaluation.py", "experiment.py", "rolling_features.py",
    "rolling_inputs.py", "rolling_experiment.py", "rolling_model.py", "train_rolling.py", "predict_rolling.py",
    "entry_support.py", "utility_structure.py", "structure_inputs.py", "structure_model.py", "train_structure.py",
    "predict_structure.py", "plan_structure.py", "viewer/maps/manifest.json", "docs/gameplay-annotations.json",
    "tests/test_data_processing.py", "tests/test_evaluation.py", "tests/test_rolling.py",
    "tests/test_utility_structure.py", "tests/test_structure_inputs.py", "tests/test_structure_plan.py",
    "tests_ml/test_rolling_models.py", "tests_ml/test_structure_models.py"]


def sha(path):
    return hashlib.sha256((repo / path).read_bytes()).hexdigest()


def saved(map_name, architecture, seed):
    root = "runs/gameplay-v2-ablations" if seed == 42 else "runs/gameplay-v2-seed-check"
    name = f"{map_name}-no_spatial-{architecture}-seed{seed}"
    directory = f"{root}/{name}"
    manifest = json.loads((repo / directory / "run.json").read_text())
    arguments = manifest["arguments"]
    assert (arguments["feature_set"], arguments["model_seed"], arguments["epochs"], arguments["split_seed"],
            arguments["validation_map"], arguments["neural_model"]) == ("no_spatial", seed, 20, 42, map_name, architecture)
    return {"name": name, "directory": directory, "run_id": manifest["run_id"],
            "manifest_sha256": sha(f"{directory}/run.json"), "predictions_sha256": sha(f"{directory}/predictions.jsonl"),
            "environment": manifest["environment"]}


def run(stage, arm, map_name, architecture, seed):
    name = f"{map_name}-{arm}-{architecture}-seed{seed}"
    output = f"{ROOT}/{name}"
    direction = support["validation_map"][map_name]
    control = saved(map_name, architecture, seed)
    return {"name": name, "stage": stage, "arm": arm, "validation_map": map_name, "architecture": architecture,
            "model_seed": seed, "epochs": 20,
            "expected_retained_base_channels": direction["retained_base_channels"],
            "expected_retained_extension_channels": direction["arms"][arm]["kept_channels"],
            "output_dir": output,
            "train_argv": [PY, "-E", "-s", "-B", "train_structure.py", "--data-dir", "data", "--output-dir", output,
                           "--smoke-test", "--validation-map", map_name, "--split-seed", "42", "--model-seed", str(seed),
                           "--feature-set", "no_spatial", "--arm", arm, "--neural-model", architecture, "--epochs", "20"],
            "reexport_argv": [PY, "-E", "-s", "-B", "predict_structure.py", output, "data"],
            "reexport_stdout": output + "/reexport.jsonl",
            "control": control["name"], "must_reproduce_control": arm == "base"}


maps, architectures, seeds = ["ascent", "breeze"], ["temporal_cnn", "pooled_mlp"], [42, 43, 44]
controls = [saved(m, a, s) for s in seeds for m in maps for a in architectures]
environments = {json.dumps(c.pop("environment"), sort_keys=True) for c in controls}
assert len(environments) == 1
runs = [run("parity", "base", m, a, s) for s in seeds for m in maps for a in architectures]
runs += [run("arms", arm, m, a, s) for arm in ("structure", "placebo") for s in seeds for m in maps for a in architectures]
tests = [[PY, "-E", "-s", "-B", "-m", "unittest", "discover", "-s", "tests_ml", "-p", pattern, "-v"]
         for pattern in ("test_rolling_models.py", "test_structure_models.py")]

plan = {
    "protocol_id": "gameplay-v2-utility-structure-v1",
    "status": "draft_not_approved", "execution_authorized": False, "approval_date": None, "authorization": None,
    "drafted": "2026-11-23",
    "question": "Do channels describing who cast which support utility together, and attacker entries supported by "
                "teammates, lower validation log loss beyond a count of live support casts?",
    "design_record": "Owner verdicts and decisions: docs/gameplay-annotations.json. Channel definitions: "
                     "utility_structure.contract() and structure_inputs.extension_contract().",
    "owner_decisions": "design_decisions in docs/gameplay-annotations.json, recorded 2026-11-23",
    "source_sha256": {name: sha(name) for name in SOURCES},
    "data_sha256": sha("data/events_2629390_grid.jsonl"),
    "extension_sha256": support["extension_sha256"],
    "support_audit": {"file": "docs/structure-support.json", "sha256": sha("docs/structure-support.json")},
    "new_dependencies": [],
    "environment": {
        "python": PY, "expected_packages": json.loads(environments.pop()),
        "protocol": "Same runner protocol as runs/gameplay-v2-seed-check: verify these hashes and the installed "
                    "files' RECORD hashes before and after, run sequentially on CPU with one thread and no "
                    "data-loader workers, and keep temp files and caches inside the output root.",
        "output_root": ROOT},
    "split_seed": 42, "test_partition": False, "feature_set": "no_spatial", "epochs_per_run": 20,
    "validation_maps": maps, "architectures": architectures, "model_seeds": seeds,
    "wiring": {
        "frozen_files_unchanged": ["rolling_features.py", "rolling_inputs.py", "rolling_experiment.py",
                                   "rolling_model.py", "train_rolling.py", "predict_rolling.py"],
        "method": "Extension channels enter through a second input array and an added, bias-free projection. The "
                  "frozen model is built first and draws from the global random stream as before; extension "
                  "weights use a separate generator. With every extension channel masked the added term is zero.",
        "executed_so_far": "Standard-library code only: channel extraction, input policies, pure-Python baseline "
                           "inference, and the --preflight path of train_structure.py. structure_model.py, the "
                           "training loop, baseline fitting and tests_ml/test_structure_models.py have been "
                           "syntax-checked and never run."},
    "stages": [
        {"id": "ml-tests", "order": 1, "argv": tests,
         "pass": "Every test passes, including the step-by-step equality of the base arm and the frozen models.",
         "on_failure": "Stop. Nothing is trained."},
        {"id": "parity", "order": 2, "runs": 12,
         "pass": "For each run, every probability in predictions.jsonl equals the named control's, for all 555 "
                 "windows and all four models, and the re-export equals predictions.jsonl.",
         "on_failure": "Stop before the arms. Rerun the frozen train_rolling.py once for one failing configuration "
                       "to separate environment drift from a wiring fault, and report both to the owner."},
        {"id": "arms", "order": 3, "runs": 24,
         "pass": "All runs complete with the expected retained channels; every re-export equals its predictions."}],
    "run_count": len(runs), "total_epochs": 20 * len(runs),
    "controls": controls, "runs": runs,
    "readout": {
        "status": "Proposed; the thresholds are the assistant's and are for the owner to confirm or change at approval.",
        "metric": "Round-mean validation log loss at the selected epoch (lowest all-window validation log loss "
                  "of 20) and at epoch 20, each for all windows and for windows with a living attacker.",
        "pairs": "Within validation map, architecture and seed: structure against the base-arm run that "
                 "reproduced the saved no_spatial control, placebo against that run, and structure against "
                 "placebo. Twelve neural pairs each.",
        "structure_helps": "Only if structure has the lower selected-epoch log loss than both the control and "
                           "the placebo in at least 9 of 12 pairs, in both strata.",
        "order_matters": "Only if structure helps as defined, lowers the temporal CNN's log loss in at least 5 "
                         "of its 6 pairs, and the CNN's mean gain exceeds the pooled control's on both maps.",
        "baselines": "Report both logistic baselines for control, placebo and structure in each map direction. "
                     "They do not depend on the neural seed; identical values across a direction's runs are "
                     "checked as an integrity test.",
        "reporting": "Every run is reported, including weaker ones. No arm, seed or map is dropped. A null "
                     "result is recorded as no demonstrated benefit."},
    "deferred_arms": ["structure_answer", "structure_spacing"],
    "limitations": [
        "One series, two maps, 44 rounds, 555 overlapping windows, no independent test set.",
        "The channels were designed from seven windows in these same rounds by a reviewer who had seen the outcomes.",
        "Both maps have been reused for feature development and model selection; results are exploratory.",
        "The entry channels follow two named abilities cast by one player per team in this series.",
        "Seeds vary initialization only. They add no independent matches.",
        "A venv integrity check is not a host malware scan or an OS sandbox."],
    "not_covered": ["Showing extension runs in the visualizer; it would label them as plain no_spatial runs.",
                    "The answer and caster-spacing arms, the retake channel and the rotation experiment."]}
assert plan["run_count"] == 36 and len({r["name"] for r in runs}) == 36
target.write_text(json.dumps(plan, indent=1) + "\n")
print(json.dumps({"plan": str(target.relative_to(repo)), "status": plan["status"], "runs": plan["run_count"],
                  "epochs": plan["total_epochs"], "sources": len(SOURCES)}))
