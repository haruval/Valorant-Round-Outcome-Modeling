# Earlier fixed-cutoff experiment

This is a historical comparison pipeline. The current project uses rolling inputs;
start with [the main README](../README.md). Execution of ML code still requires
owner approval. These commands do not reuse the retained rolling v1 artifacts.


The original revised 30-second experiment remains available for comparison.
The commands in the fixed-cutoff sections below use `train.py` and `predict.py`;
their checkpoints and prediction files are distinct from rolling runs.

For a round still in progress, estimate the probability of an attacker win after
30 seconds of observation. Time starts at `round-ended-freezetime` and uses the
feed's timestamps. Each one-second bin contains the state after events strictly
before the end of that bin. Events at or after the cutoff never enter features.

The experiment starts with a training-prior baseline, snapshot logistic regression,
and temporal-summary logistic regression. The temporal baseline adds eight timing
and recent-activity summaries to the last snapshot. An optional 730-parameter CNN
receives `(batch, seconds, 12 features)`: two temporal convolutions (8 and 16 channels,
kernel width 3), ReLU and dropout 0.1, global average pooling, and a two-class output.
It has no batch normalization or max pooling. The CNN is a comparison model;
whether temporal convolutions help is an open experimental question.
Features include players alive, cumulative kills, spike plant status, loadout
value at the end of buy time, ability uses, and first-kill indicators. Team IDs,
player names, map names, and the outcome are not input features.

Rounds that have ended by the cutoff are excluded. This means the task is
conditional on the round still being active; it does not cover every round.
See [methodology](methodology.md) for the exact feature definitions and
limitations.

## Local use

The data audit and `tests/` use only the Python standard library (Python 3.9+):

```bash
python3 -S -B -m unittest discover -s tests -v
python3 -S -B data_processing.py data
```

Baseline training requires Python 3.11+ and `requirements-baseline.txt` (NumPy and
scikit-learn). `requirements.txt` additionally includes PyTorch and safetensors
for the CNN. The
version ranges are proposed compatibility bounds, not a tested lockfile. Review
and approve dependencies before installing them in an isolated environment:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: -r requirements-baseline.txt
.venv/bin/python -m unittest discover -s tests_ml -p test_baselines.py -v
```

To exercise training with the included single-series sample:

```bash
.venv/bin/python train.py --data-dir data --output-dir runs/dev-30s \
  --observation-seconds 30 --split-seed 42 --model-seed 42 --smoke-test
python3 -S -B serve.py --data-dir data --port 8765 \
  --predictions runs/dev-30s/predictions.jsonl --run-manifest runs/dev-30s/run.json
```

That run holds out one map from the same series. Its validation numbers are
development diagnostics, not evidence of generalization. `predict.py` replays
completed rounds and prints JSON lines; it is not a live game integration and
does not present file-level accuracy as a test score. Training writes predictions
for every eligible round, including training rounds. The viewer labels the
partition explicitly and shows the run, cutoff, schema, and hashes. It compares
saved model probabilities only after the cutoff; it performs no live inference.

To compare the CNN after separate approval for its dependencies and execution:

```bash
.venv/bin/python -m pip install --only-binary=:all: -r requirements.txt
.venv/bin/python -m unittest discover -s tests_ml -v
.venv/bin/python train.py --data-dir data --output-dir runs/dev-30s-cnn \
  --observation-seconds 30 --split-seed 42 --model-seed 42 --smoke-test \
  --include-cnn --epochs 3
```

These commands are instructions for a future approved run, not completed validation.
Installing wheels still downloads and executes third-party software, including
native libraries and transitive dependencies. Binary-only installation avoids
source builds; it does not establish trust. Fitted baselines use bounded JSON
coefficients, and the CNN uses safetensors. Model loading and ML execution still
require approval; no pickle or remote model loading is used.

For an experiment with independent series, add data from at least three series
and omit `--smoke-test`:

```bash
.venv/bin/python train.py --data-dir data --output-dir runs/series-30s \
  --observation-seconds 30 --split-seed 42 --model-seed 42
```

Three series is only the mechanical minimum. A credible evaluation needs more
independent matches and a test set kept separate while choosing features,
cutoffs, and hyperparameters. Runs use the CPU, a fixed seed, and deterministic
PyTorch operations when the CNN is enabled. Keep `--split-seed` fixed while varying
`--model-seed` (`--seed` remains an alias) to compare initialization sensitivity.
Adding data can change the split even with the same seed; preserve existing run
manifests and establish a new evaluation protocol before expanding the dataset.
Exact agreement across library versions is not guaranteed.

Each new run directory contains data hashes, round assignments, installed
package versions, fitted baseline coefficients, baseline comparisons, and
per-round predictions. CNN runs also contain epoch history and the selected
checkpoint. A completed `run.json` binds these artifacts and partitions. Existing run directories are never
overwritten. Run outputs are ignored by Git.
