# Pending v2 validation and controlled experiment

**Prepared, not executed.** No new packages are needed. The retained macOS arm64 /
CPython 3.12 environment contains NumPy 2.5.4, scikit-learn 1.9.1, PyTorch 2.15.0,
safetensors 0.8.0 and their reviewed dependencies. These are the installed pins,
not a recommendation to upgrade. Full hashes and installation history are in
[the existing validation record](ml-validation.md).

The prior approval covered the completed v1 run. Under the owner's execution
policy, executing ML libraries, fitting these revised models, or opening their
new checkpoints waits for approval. Standard-library checks and the feature
preview have already run. The plan below introduces no downloads, new dependency,
remote model, public endpoint, or global environment change.

## First execution: bounded validation

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

## Subsequent study: separate scope

Do not treat the three-epoch check as a result about temporal learning. Before a
larger run, freeze the reviewed ability taxonomy, source bytes, feature contract,
and this protocol. The following study is proposed separately:

- Full inputs: both neural architectures, seeds 42/43/44, both map directions,
  20 epochs each: 12 runs, each with the three baselines.
- Feature-family checks: both architectures, seed 42, both directions, four
  ablations (`state_only`, `no_utility`, `no_spatial`, `no_trades`),
  20 epochs each: 16 additional runs.
- Keep optimizer, weighting, learning rate, epoch budget, and selection criterion
  fixed. Save every run, not just favorable results. Record input support because
  different training maps retain different channels.
- Report round-mean log loss and Brier score for each direction and seed, training
  curves, sample counts, and selected epochs. Inspect sequences where the CNN
  and controls differ. Do not use window-level confidence intervals as if
  overlapping windows were independent.
- A performance change after adding/removing a feature is predictive evidence
  within this sample, not a causal estimate of that gameplay action's benefit.

An example command for a **later approved** full-input study cell is:

```bash
"$VALORANT_ENV_ROOT/venv/bin/python" -E -s -B train_rolling.py \
  --data-dir data --output-dir runs/gameplay-v2-study/ascent-all-temporal_cnn-seed42 \
  --smoke-test --validation-map ascent --split-seed 42 --model-seed 42 \
  --feature-set all --neural-model temporal_cnn --epochs 20
```

The two directions reuse one series, with map/composition/team dependence.
Validation already influenced the redesign. Neither direction is an untouched
test set, and trying multiple configurations increases selection bias. A future
evaluation should reserve independent series before tuning. The present study
can demonstrate implementation quality and reasoned experimentation, not establish
general competitive-match accuracy.

## Security and removal

The reviewed environment is reused without installation. It is still third-party
Python and native code executing with the account's permissions. A venv isolates
package selection, not filesystem or network access. The code has no download or
upload path, uses CPU with one worker/thread configuration, and saves bounded
JSON/safetensors artifacts instead of pickle; this reduces specific risks but
does not guarantee safety. Schema checks and hashes detect mismatches, not malicious
code that was already approved or introduced later.

New run files, temporary checkpoints, and configured caches go under
`runs/gameplay-v2-validation/`. Python bytecode writes are disabled by `-B`.
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
erasure of OS logs, swap, backups, or browser/shell history. The optional later
study directory has a separate lifecycle and is not removed by this command.
