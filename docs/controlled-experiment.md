# V2 validation and controlled experiment

**Bounded validation completed October 31, 2026.** No new packages were installed.
The retained macOS arm64 /
CPython 3.12 environment contains NumPy 2.5.4, scikit-learn 1.9.1, PyTorch 2.15.0,
safetensors 0.8.0 and their reviewed dependencies. These are the installed pins,
not a recommendation to upgrade. Full hashes and installation history are in
[the existing validation record](ml-validation.md).

The owner authorized a new safety review and execution if it passed on October 31.
The limited run below passed that review and completed. A later approval covered
the 12-run full-input study, which also completed. A separately approved
16-run feature-ablation study is now complete; results appear below. No package downloads, remote
models, public endpoints, or global environment changes were needed.

## Completed run

All 58 standard-library tests and four optional ML tests passed. Both architectures
trained for three epochs with seed 42 on 286 windows from 23 Breeze rounds and
validated on 269 windows from 21 Ascent rounds. Both selected epoch 3. There is no
test partition. Runtime checks confirmed 2,602 CNN and 2,666 pooled-control
parameters. Each uses 174 varying training channels and masks the other 48.

| Model | Round-mean log loss ↓ | Round-mean Brier score ↓ |
| --- | ---: | ---: |
| Training prior | 0.6648 | 0.2359 |
| Snapshot logistic | 0.4972 | 0.1467 |
| History-summary logistic | 0.4715 | 0.1565 |
| Pooled neural control | 0.6456 | 0.2268 |
| Temporal CNN | 0.7094 | 0.2583 |

The history baseline has the lowest log loss; snapshot logistic has the lowest
Brier score. The CNN's epoch log losses improved from 0.7392 to 0.7232 to 0.7094,
but it performed worse than the training prior in this run. The pooled control
improved from 0.6621 to 0.6537 to 0.6456. Three epochs are a bounded pipeline
check, not enough evidence to choose or reject temporal convolution as an
architecture. No generalization or causal gameplay claim follows from these scores.

Each run's 555 re-exported prediction rows matched the original probabilities
exactly (maximum absolute difference 0), including provenance. The two runs have
identical schemas, source implementations, input policies, partitions, and
baseline artifact bytes. Both 44-round viewer catalogs passed validation.
The CNN predictions are served on [localhost:8768](http://127.0.0.1:8768/).
Live HTTP checks are recorded with the run; interactive browser checks could not
be repeated because browser automation was unavailable in this session.

The safety review verified all 19 installed package versions against the retained
installation record and all 20,032 hashed files against their installed RECORDs.
No mismatches or unexpected custom startup modules were found. The sole .pth
startup hook was the reviewed setuptools distutils shim. A fresh
[OSV query](https://google.github.io/osv.dev/post-v1-querybatch/) for the exact
19 versions returned no matching advisories on October 31. RECORD integrity and
advisory checks are not a malware audit or guarantee of safety.

Source commit at execution: `f520120`. Detailed hashes, review evidence, logs,
artifacts, and metrics are retained in the ignored `runs/gameplay-v2-validation/`
directory. Its `safety-review.json` records the pre-execution review;
`validation.json` records export and viewer verification. The original v1 run and
environment remain intact. Expected new files can be removed using the scoped
cleanup below; no cleanup has been performed.

## Completed 12-run study

The owner subsequently approved the full-input comparison on October 31, 2026.
All **12 runs / 240 epochs** completed with unchanged feature, training, and
inference code. Both architectures used seeds 42/43/44, 20 epochs, both map
directions, fixed optimizer/settings, and validation checkpoint selection.
The environment was reused without installation. Feature ablations were not part
of that batch; the separate completed study is documented below.

All **6,660 prediction rows** matched exactly after checkpoint re-export.
Verification recomputed round-balanced metrics from exported probabilities and
round labels, checked artifact/source hashes and checkpoint selection, and
confirmed identical policies, partitions, and baseline artifacts within each
direction. Logs contained no warning or error lines. The previous 58 standard-library
and four ML tests still apply to the unchanged execution code; they were not
unnecessarily rerun for this study.

The [machine-readable results](study-12runs.json) include every seed, epoch,
training/validation curve, run ID, and manifest hash. Neural entries below show
mean **(minimum–maximum)** over three seeds. Baselines are deterministic here;
they are not six independent estimates. All scores are development validation
scores, and lower is better.

| Model | Breeze → Ascent log loss | Ascent → Breeze log loss |
| --- | ---: | ---: |
| Training prior | 0.6648 | 0.6696 |
| Snapshot logistic | 0.4972 | 0.5687 |
| History-summary logistic | 0.4715 | 1.1063 |
| Temporal CNN | 0.3459 (0.3235–0.3655) | 0.4476 (0.4398–0.4589) |
| Pooled neural control | 0.3517 (0.3368–0.3701) | 0.4324 (0.4140–0.4467) |

The CNN has a slightly lower mean log loss on Ascent, but beats the paired pooled
control in only one of three seeds there. On Breeze, the pooled control wins all
three seeds. Both neural architectures improve on the logistic baselines at their
selected checkpoints in both directions. These results support investigating
nonlinear history modeling; they do **not** establish a consistent advantage from
temporal convolutions. Time-valued features remain available in the pooled control.

The history-logistic baseline changes from 0.4715 on Ascent to 1.1063 on Breeze.
That instability matters: one favorable map direction would have hidden it.
No configuration was adjusted to rescue that result.

### Every selected checkpoint

| Validation map | Model | Seed | Selected epoch | Validation log loss | Validation Brier |
| --- | --- | ---: | ---: | ---: | ---: |
| ascent | Temporal CNN | 42 | 18 | 0.3655 | 0.1180 |
| ascent | Pooled neural control | 42 | 14 | 0.3368 | 0.1097 |
| ascent | Temporal CNN | 43 | 18 | 0.3487 | 0.1132 |
| ascent | Pooled neural control | 43 | 15 | 0.3482 | 0.1141 |
| ascent | Temporal CNN | 44 | 19 | 0.3235 | 0.1033 |
| ascent | Pooled neural control | 44 | 16 | 0.3701 | 0.1205 |
| breeze | Temporal CNN | 42 | 11 | 0.4589 | 0.1468 |
| breeze | Pooled neural control | 42 | 10 | 0.4467 | 0.1435 |
| breeze | Temporal CNN | 43 | 12 | 0.4398 | 0.1391 |
| breeze | Pooled neural control | 43 | 10 | 0.4140 | 0.1319 |
| breeze | Temporal CNN | 44 | 10 | 0.4442 | 0.1450 |
| breeze | Pooled neural control | 44 | 11 | 0.4366 | 0.1400 |

### Training and late validation behavior

The following are means over the same three seeds. Selected validation scores
benefit from choosing an epoch on these validation rounds; epoch 20 is also shown
to expose that selection and the late degradation.

| Validation map | Model | Selected validation loss | Epoch-20 validation loss | Epoch-20 training loss |
| --- | --- | ---: | ---: | ---: |
| ascent | Temporal CNN | 0.3459 | 0.4116 | 0.2211 |
| ascent | Pooled neural control | 0.3517 | 0.4141 | 0.2011 |
| breeze | Temporal CNN | 0.4476 | 0.5352 | 0.1450 |
| breeze | Pooled neural control | 0.4324 | 0.5881 | 0.1299 |

Best epochs were 18/18/19 for the Ascent CNN, 14/15/16 for the Ascent pooled
control, 11/12/10 for the Breeze CNN, and 10/10/11 for the Breeze pooled control.
Validation loss worsened again by epoch 20 while training loss remained low,
consistent with overfitting. More epochs alone are not an established remedy.

Seed ranges measure initialization sensitivity, not uncertainty over independent
matches. The two map directions reuse one series, and validation already informed
feature design and checkpoint selection. They are not independent test folds.
The feature ablations below are reported separately. Independent-series data and
future test results remain necessary for assessing generalization.

### Replay inspection candidates

The study viewer at [localhost:8769](http://127.0.0.1:8769/) uses the Ascent-validation
CNN with **seed 42**, chosen as the fixed demonstration seed, not the best-scoring
seed. Its Breeze predictions are training predictions. The old three-epoch
viewer on 8768 remains separate.

These candidates use seed 42 in each validation direction: the largest
CNN/control probability disagreement per round, then the three largest rounds
per map. They were selected without filtering for which model was right.
Observed winners are included only for retrospective inspection, never as inputs.

| Validation map / round | Window | CNN P(ATK win) | Pooled P(ATK win) | Observed winner |
| --- | --- | ---: | ---: | --- |
| ascent / round-13 | 5–25s | 85.0% | 29.9% | Attacker |
| ascent / round-1 | 30–50s | 86.3% | 41.4% | Attacker |
| ascent / round-7 | 35–55s | 40.0% | 6.6% | Attacker |
| breeze / round-23 | 95–115s | 72.7% | 35.5% | Attacker |
| breeze / round-13 | 0–20s | 83.3% | 52.9% | Defender |
| breeze / round-5 | 5–25s | 47.7% | 21.3% | Defender |

Review the cast timeline, fresh-position coverage, and kills/trades around these
windows. Differences between model probabilities do not by themselves identify
which event caused the difference or prove a tactical explanation.
Breeze candidates require the Breeze-validation run; the default study viewer
uses the opposite direction.

### Reproduction and artifacts

The frozen local protocol and all artifacts are in ignored
`runs/gameplay-v2-study/` (about 22 MiB). `protocol.json` specifies the matrix and
source hashes; `execute.sh` records the exact execution commands;
`summarize.py` preserves the standard-library verification; `summary.json`
contains the full results. Existing directories are intentionally refused.

The command matrix used the environment and cache settings from the bounded run,
redirected to `runs/gameplay-v2-study/`:

```bash
for validation_map in ascent breeze; do
  for model_seed in 42 43 44; do
    for architecture in temporal_cnn pooled_mlp; do
      run_name="$validation_map-all-$architecture-seed$model_seed"
      "$VALORANT_ENV_ROOT/venv/bin/python" -E -s -B train_rolling.py \
        --data-dir data --output-dir "$VALORANT_STUDY_ROOT/$run_name" \
        --smoke-test --validation-map "$validation_map" --split-seed 42 \
        --model-seed "$model_seed" --feature-set all \
        --neural-model "$architecture" --epochs 20
      "$VALORANT_ENV_ROOT/venv/bin/python" -E -s -B predict_rolling.py \
        "$VALORANT_STUDY_ROOT/$run_name" data \
        > "$VALORANT_STUDY_ROOT/$run_name/reexport.jsonl"
    done
  done
done
```

After stopping study previews, deleting only the dedicated study directory removes
its expected outputs while preserving the environment and previous validation.
No cleanup was performed. The security/removal limitations below still apply.

## First execution: bounded validation

The following commands document the completed execution. The output directory
now exists, so setup deliberately refuses to overwrite it. Tests/training logs
were also captured in that directory. Do not rerun or expand this plan silently.

Run in order in a separate shell. Stop on any failure; do not modify dependencies
or overwrite an existing run to continue. This fits the three baselines twice
and trains one CNN and one pooled control for **three epochs each**, seed 42,
training on Breeze and validating on Ascent. It is a runtime check.

Setup (no installation):

```bash
/bin/zsh -f
set -e
umask 077
cd /Users/arigladstone/Documents/Valorant-Round-Outcome-Modeling
export VALORANT_ENV_ROOT="$PWD/runs/ml-validation-20261029"
export VALORANT_V2_ROOT="$PWD/runs/gameplay-v2-validation"
test ! -L runs
test ! -L "$VALORANT_ENV_ROOT"
test -x "$VALORANT_ENV_ROOT/venv/bin/python"
mkdir "$VALORANT_V2_ROOT"
mkdir "$VALORANT_V2_ROOT/tmp" "$VALORANT_V2_ROOT/cache"
export TMPDIR="$VALORANT_V2_ROOT/tmp"
export XDG_CACHE_HOME="$VALORANT_V2_ROOT/cache"
export TORCH_HOME="$VALORANT_V2_ROOT/cache/torch"
export TORCH_EXTENSIONS_DIR="$VALORANT_V2_ROOT/cache/torch-extensions"
export TORCHINDUCTOR_CACHE_DIR="$VALORANT_V2_ROOT/cache/torchinductor"
export JOBLIB_TEMP_FOLDER="$VALORANT_V2_ROOT/tmp"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
```

Recheck standard-library tests, then run the **four optional rolling ML tests**:

```bash
"$VALORANT_ENV_ROOT/venv/bin/python" -E -S -B -m unittest discover -s tests -v
"$VALORANT_ENV_ROOT/venv/bin/python" -E -s -B -m unittest discover \
  -s tests_ml -p test_rolling_models.py -v
```

The optional tests import ML/native libraries, fit synthetic baselines, compute
gradients, and save/reload temporary local safetensors. They test CNN causality,
input-mask enforcement, pooled-control permutation invariance, parameter counts,
checkpoint policy preservation, and exported baseline probabilities.

Then train and export the two runs:

```bash
for architecture in temporal_cnn pooled_mlp; do
  "$VALORANT_ENV_ROOT/venv/bin/python" -E -s -B train_rolling.py \
    --data-dir data --output-dir "$VALORANT_V2_ROOT/$architecture" \
    --smoke-test --validation-map ascent --split-seed 42 --model-seed 42 \
    --feature-set all --neural-model "$architecture" --epochs 3
  "$VALORANT_ENV_ROOT/venv/bin/python" -E -s -B predict_rolling.py \
    "$VALORANT_V2_ROOT/$architecture" data \
    > "$VALORANT_V2_ROOT/$architecture/reexport.jsonl"
done
```

Verify re-exported probabilities and provenance, plus identical partitions and
input policy for the comparison. This verification uses only the standard library:

```bash
/usr/bin/python3 -I -S -B - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ['VALORANT_V2_ROOT'])
manifests = []
for name in ('temporal_cnn', 'pooled_mlp'):
    run = root / name
    manifest = json.loads((run / 'run.json').read_text())
    assert manifest['schema']['id'] == 'rolling-gameplay-v2'
    assert manifest['arguments']['epochs'] == 3
    assert manifest['arguments']['validation_map'] == 'ascent'
    manifests.append(manifest)
    original = [json.loads(s) for s in (run / 'predictions.jsonl').read_text().splitlines()]
    exported = [json.loads(s) for s in (run / 'reexport.jsonl').read_text().splitlines()]
    assert len(original) == len(exported) == 555
    for a, b in zip(original, exported):
        p, q = a.pop('probabilities'), b.pop('probabilities')
        assert a == b and p.keys() == q.keys()
        assert all(abs(p[k] - q[k]) <= 1e-6 for k in p)
    print(name, '555 predictions and provenance verified')
a, b = manifests
assert a['schema'] == b['schema']
assert a['partitions'] == b['partitions']
assert a['input_policy'] == b['input_policy']
assert a['implementation_sha256'] == b['implementation_sha256']
assert a['neural_parameter_count'] == 2602
assert b['neural_parameter_count'] == 2666
print('Comparison contract, implementation, splits and parameter counts verified.')
PY
```

Only after those checks pass, serve the new CNN predictions in another loopback
viewer. The server itself uses the standard library and reads JSON, not models:

```bash
/usr/bin/python3 -E -S -B serve.py --data-dir data --port 8768 \
  --rolling-predictions "$VALORANT_V2_ROOT/temporal_cnn/predictions.jsonl" \
  --rolling-run-manifest "$VALORANT_V2_ROOT/temporal_cnn/run.json"
```

Open [the proposed v2 prediction preview](http://127.0.0.1:8768/). Check warm-up,
20/25-second boundaries, backward seeking, both maps, model switching, partition
labels, v2 schema, feature set, and masked-channel count. After Ctrl+C, replace
both `temporal_cnn` path components with `pooled_mlp` to inspect the control.
The existing 8766 v1 viewer and 8767 feature preview can remain available.

## Completed feature ablations

**Completed October 31, 2026: 16 runs / 320 epochs**, using the existing reviewed
environment without installation. Both architectures used seed 42, both map
directions and four input subsets. Features, optimizer, weighting, epoch budget,
partitions and checkpoint selection were unchanged. Each run also refit all three
baselines with the same input mask. The four full-input seed-42 runs are the
matched controls; the three-seed averages from the earlier study are not used as
ablation controls. See the [executed plan](ablation-plan.json).

All **8,880 new prediction rows** matched exactly after checkpoint re-export;
the four controls' 2,220 rows were rechecked. Independent arithmetic reproduced
round-balanced log loss and Brier scores. Verification also checked source and
artifact hashes, identical window membership/partitions, the intended masks,
selected checkpoints, and identical baseline artifacts within each architecture
pair. Logs contained no warning/error lines. Pre- and post-execution checks found
no changes to the 20,032 hashed installed package files. These integrity checks
are not a host malware scan. The previously passing 60 standard-library and four
ML tests cover unchanged execution code; they were not rerun for this batch.

The [machine-readable results](ablation-results.json) preserve every result,
20-epoch training/validation curve, run ID, manifest hash and masked-channel list.
There is still **one series, 44 rounds, 555 overlapping windows and no independent
test set**. Ascent validation has 21 rounds / 269 windows; Breeze has 23 / 286.

### Selected-checkpoint round-mean log loss

Lower is better. These are development validation scores from **one seed**.

| Input set | Ascent CNN | Ascent pooled | Breeze CNN | Breeze pooled |
| --- | ---: | ---: | ---: | ---: |
| All inputs (existing control) | 0.3655 | 0.3368 | 0.4589 | 0.4467 |
| State only | 0.3663 | 0.3559 | 0.5473 | 0.5438 |
| Without utility | 0.3178 | 0.3288 | 0.4506 | 0.4426 |
| Without spatial features | 0.2991 | 0.3028 | 0.4255 | 0.4294 |
| Without trade features | 0.3570 | 0.3368 | 0.4466 | 0.4466 |

### Selected-checkpoint round-mean Brier score

| Input set | Ascent CNN | Ascent pooled | Breeze CNN | Breeze pooled |
| --- | ---: | ---: | ---: | ---: |
| All inputs (existing control) | 0.1180 | 0.1097 | 0.1468 | 0.1435 |
| State only | 0.1152 | 0.1104 | 0.1801 | 0.1724 |
| Without utility | 0.1022 | 0.1039 | 0.1432 | 0.1437 |
| Without spatial features | 0.0950 | 0.0977 | 0.1400 | 0.1405 |
| Without trade features | 0.1155 | 0.1063 | 0.1422 | 0.1435 |

### Checkpoint selection and final epoch

Each cell is **selected epoch / epoch-20 validation log loss**. This exposes
late degradation as well as the benefit of validation-based checkpoint selection.

| Input set | Ascent CNN | Ascent pooled | Breeze CNN | Breeze pooled |
| --- | ---: | ---: | ---: | ---: |
| All inputs (existing control) | 18 / 0.4517 | 14 / 0.3767 | 11 / 0.5384 | 10 / 0.5607 |
| State only | 18 / 0.3676 | 20 / 0.3559 | 12 / 0.6662 | 17 / 0.5509 |
| Without utility | 19 / 0.3618 | 15 / 0.3581 | 12 / 0.5108 | 11 / 0.5041 |
| Without spatial features | 20 / 0.2991 | 20 / 0.3028 | 14 / 0.4643 | 12 / 0.4766 |
| Without trade features | 18 / 0.4334 | 13 / 0.3815 | 11 / 0.5461 | 10 / 0.5614 |

### Logistic baselines under the same masks

The training-prior log losses remain 0.6648 on Ascent and 0.6696 on Breeze.
Other baseline log losses are shown below; all Brier scores are in the result JSON.

| Input set | Ascent snapshot | Ascent history | Breeze snapshot | Breeze history |
| --- | ---: | ---: | ---: | ---: |
| All inputs (existing control) | 0.4972 | 0.4715 | 0.5687 | 1.1063 |
| State only | 0.3591 | 0.3355 | 0.4330 | 0.4503 |
| Without utility | 0.3879 | 0.7186 | 0.5385 | 1.3818 |
| Without spatial features | 0.4164 | 0.3734 | 0.4081 | 0.5420 |
| Without trade features | 0.4450 | 0.4522 | 0.5699 | 1.1284 |

### Interpretation

- Removing spatial channels improved both neural models' log loss and Brier score
  in both directions, including at epoch 20. For the CNN, selected log loss fell
  from 0.3655 to 0.2991 on Ascent and 0.4589 to 0.4255 on Breeze. This is evidence
  against the current spatial representation in this sample, not against the
  importance of positioning in the game. Map-dependent geometry, sparse reporting
  and overfitting are possible explanations; this experiment does not isolate them.
- Removing utility channels also improved selected log loss in all four neural
  comparisons, with much smaller changes on Breeze. The Breeze pooled model's
  Brier score worsened slightly. Current aggregate utility features have not
  demonstrated a reliable incremental benefit.
- Removing trade channels barely changed pooled-model log loss (less than 0.00005
  in either direction). The CNN improved modestly. Kill/death timing remains in
  these inputs, so useful trade information may still be available indirectly.
- State-only neural inputs nearly matched the full CNN on Ascent but were worse
  on Breeze for both architectures. The state-only logistic baselines improved
  substantially over their full-input versions on both maps and remain serious
  benchmarks for any more complex representation.
- The strongest Breeze result was **snapshot logistic without spatial features**:
  0.4081 log loss, compared with the best CNN's 0.4255. On Ascent, the reduced CNN's
  0.2991 beat the best logistic result (state-only history, 0.3355). These results
  do not establish a general neural-model advantage.
- Without spatial channels the CNN beat the pooled control by about 0.0037/0.0039
  log loss. One seed and small, repeatedly inspected validation sets do not
  establish a stable advantage from temporal convolution.

The follow-up is to test player-level relationships identified in
[owner review](feature-review.md#owner-guidance-and-representation-follow-up):
entry movement timed with allied supporting utility, and movement after teammate
deaths with location and observation-coverage context. Keep the reduced-input
CNN and pooled model as additional controls. Confirmed sightings require a
visibility signal that the current export does not provide. These follow-up
features have not been added or trained in this frozen comparison.

### Artifacts and execution

Ignored `runs/gameplay-v2-ablations/` contains the approved plan snapshot,
`protocol.json`, `safety-review.json`, `post-execution-integrity.json`, exact
`execute.py` orchestration, cache environment, logs, checkpoints, predictions,
`summarize.py` verification and `summary.json`. Existing output roots are refused.
The plan records the exact train/re-export argument lists for all 16 runs.
No additional seed, feature revision or training run was silently added.

The existing localhost viewers continue to show their full-input controls.
To inspect an ablated run with the existing viewer, pass its `predictions.jsonl`
and matching `run.json` using `--rolling-predictions` and
`--rolling-run-manifest`. Observe the displayed feature-set and partition labels.

After stopping any preview that uses these artifacts, removing only the dedicated
ablation directory removes its expected files while preserving the environment
and earlier runs. No cleanup was performed. The limits below still apply.

## Completed reduced-input seed check

**Completed October 31, 2026: eight new runs / 160 epochs.** The owner approved
the proposed next steps. The [exact command plan](reduced-seed-plan.json) repeats
`no_spatial` for seeds 43 and 44, both architectures and both validation maps,
using the existing environment. Training code, data, optimizer, input policy,
partitions and 20-epoch budget were frozen. Seed 42 comes from the completed
ablation study; all three seeds have matched full-input controls from the earlier
study. No new prototype channels were passed to a model.

The [complete results](reduced-seed-results.json) include per-seed selected and
epoch-20 scores, all histories, baselines, provenance and paired differences.
Each cell below is mean round-balanced validation log loss over seeds 42–44.
Lower is better; these remain development-validation results.

| Validation map | Model | Full inputs | No spatial | Reduced-input wins |
| --- | --- | ---: | ---: | ---: |
| Ascent | Temporal CNN | 0.3459 | 0.3289 | 2 / 3 |
| Ascent | Pooled control | 0.3517 | 0.3170 | 3 / 3 |
| Breeze | Temporal CNN | 0.4476 | 0.4242 | 3 / 3 |
| Breeze | Pooled control | 0.4324 | 0.4121 | 3 / 3 |

Spatial removal improved 11 of 12 matched selected-checkpoint log losses. The
exception was Ascent CNN seed 43 (0.3717 reduced versus 0.3487 full). The reduced
CNN beat the reduced pooled control in three of six paired runs; the pooled
control had better mean log loss on both maps. The original one-seed CNN lead
therefore does not establish a reliable temporal-convolution benefit. Additional
seeds test initialization sensitivity; they add no independent matches and do
not turn the repeatedly inspected validation rounds into a final test set.

All **4,440 new prediction rows** matched exactly after artifact reload. The
standard-library verifier independently reproduced scores, checkpoint selection,
partitions, feature masks and matched-baseline artifacts, and rechecked **8,880
existing rows** across the four reduced seed-42 runs and 12 full-input controls.
Pre/post integrity checks found the same 20,032 hashed installed files
(735,491,911 bytes); no packages were installed and no logs contained warnings
or errors. This is an integrity check, not a host malware scan.

`runs/gameplay-v2-seed-check/` retains the execution protocol, preflight, exact
runner, logs, checkpoints, re-exports, verifier, numerical summary and post-run
integrity record (about 15 MB). The verifier completed before the subsequent
viewer edits; historical source hashes describe that execution snapshot. The
training implementation itself is unchanged. Removing this dedicated directory
after its processes have stopped removes its expected outputs, independently
of earlier studies and the reviewed environment. No cleanup was performed.

The [comparison viewer configuration](visualizer-comparison.json) pins eight
seed-42 manifests and prediction files: full/reduced inputs, both architectures,
both map directions. Each replay map receives only its validation predictions;
switching experiments preserves separate manifests and model hashes. Seed 42
is an inspection convention, not a best-seed selection. The additional
[utility/movement prototype](feature-review.md#utilitymovement-prototype) is
untrained and displayed separately from prediction inputs.

## Security and removal

The reviewed environment is reused without installation. It is still third-party
Python and native code executing with the account's permissions. A venv isolates
package selection, not filesystem or network access. The code has no download or
upload path, uses CPU with one worker/thread configuration, and saves bounded
JSON/safetensors artifacts instead of pickle; this reduces specific risks but
does not guarantee safety. Schema checks and hashes detect mismatches, not malicious
code that was already approved or introduced later.

Bounded validation files live under `runs/gameplay-v2-validation/`; full-input
study files under `runs/gameplay-v2-study/`; and the new ablation files, temporary
checkpoints and configured caches under `runs/gameplay-v2-ablations/` (about
29 MiB). Python bytecode writes are disabled by `-B`.
Expected new outputs can be removed without changing the original v1 environment
or run. Stop the preview and any ML process first. The following cleanup is also
**not executed**:

```bash
/usr/bin/python3 -I -S -B - <<'PY'
from pathlib import Path
import shutil
repo = Path('/Users/arigladstone/Documents/Valorant-Round-Outcome-Modeling')
root = repo / 'runs' / 'gameplay-v2-validation'
if Path.cwd().resolve() != repo.resolve():
    raise SystemExit('Run from the repository directory.')
if root.is_symlink() or root.parent.is_symlink() or not root.is_dir():
    raise SystemExit('Expected the dedicated nonsymlink v2 directory.')
shutil.rmtree(root)
assert not root.exists()
print('Removed v2 validation outputs; original environment and v1 run retained.')
PY
```

Deleting expected files cannot undo arbitrary dependency behavior or guarantee
erasure of OS logs, swap, backups, or browser/shell history. The full-input study
and ablation directories have separate lifecycles and are not removed by this
bounded-validation cleanup command.
