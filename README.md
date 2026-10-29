# VALORANT Round-Outcome Modeling

An experiment in predicting whether the attacking team will win a Valorant round
from the recent history of utility, combat, positioning, economy, and objective state.
The current direction is a temporal CNN with a 20-second rolling history, updated
every five seconds throughout a round. The project began as an undergraduate
machine learning final project and is being developed independently, with an
emphasis on data correctness and reproducible evaluation.

**Status:** the pipeline has been revised; no validated performance result is
claimed for this version. The included data has one series and two completed
maps, which is enough for development but not an independent match-level test.
The 48 standard-library regression tests pass on Python 3.9.6. The fixed-cutoff data audit
retains 43 rounds at 30 seconds (17 attacker wins, 26 defender wins). ML runtime
tests and training have not been run; they require separate approval.

The rolling pipeline produces 555 windows across 44 rounds. These overlap and
are not independent samples. No rolling performance result is available.

## Rolling experiment

The main research question is whether learned temporal patterns improve outcome
prediction beyond current state and explicit history summaries. The initial
gameplay priorities are utility combinations before contact, rotations after
information-related ability use, and trades. Regional presence is one input family
alongside ability identity/origin, kill origin/victim location, economy, reported
health, and spike/defuse state.

The model is a 3,746-parameter causal temporal CNN with four dilated residual
blocks. Baselines use the same observation channels. Coordinates older than two
seconds are masked; reveal success, sightlines, and ability-effect geometry are
not inferred. See [the rolling feature design](docs/rolling-design.md) for the
contract, model, weighting, limits, and gameplay decisions to review.

Standard-library audit (no model runtime):

```bash
python3 -E -S -B rolling_features.py data
```

After separate approval for dependencies, training, and model loading:

```bash
.venv/bin/python -m unittest discover -s tests_ml -p test_rolling_models.py -v
.venv/bin/python train_rolling.py --data-dir data --output-dir runs/rolling-dev \
  --smoke-test --split-seed 42 --model-seed 42 --include-cnn --epochs 30
```

Omitting `--include-cnn` fits the three rolling baselines only. The CNN requires
`requirements.txt`; baselines require `requirements-baseline.txt`. Both are
proposed ranges, not approved or tested installations. Existing run directories
are rejected. A completed run writes fitted artifacts, metrics, a manifest, and
time-indexed prediction JSONL. To display that approved output:

```bash
python3 -E -S -B serve.py --data-dir data --port 8766 \
  --rolling-predictions runs/rolling-dev/predictions.jsonl \
  --rolling-run-manifest runs/rolling-dev/run.json
```

The viewer defaults to rolling feature inspection even without model output. It
shows the selected 20-second window, utility uses, kills, trade candidates, and
position freshness. Window kill markers locate victims and are colored by the
killer's side. Fitted probabilities remain unavailable until an approved run.

Additional exports can be processed with `predict_rolling.py RUN_DIRECTORY DATA_PATH`
after approval to load that run's fitted artifacts. This is recorded-match
analysis from the spectator state, not a live competitive-game integration.

## Earlier fixed-cutoff experiment

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
See [methodology](docs/methodology.md) for the exact feature definitions and
limitations.

## Local replay viewer

Explore the supplied Riot hackathon event data without installing packages:

```bash
python3 -S -B serve.py --data-dir data --port 8765
```

Open [the local viewer](http://127.0.0.1:8765). Select a map and round, play or scrub
the timeline, and inspect player coordinates, health, ability readiness, casts,
kills, and the model's input features. Space toggles playback; left/right arrows
step one second when a control is not focused.

The minimaps use archived Breeze and Ascent artwork with calibrated world coordinates
and last reported player positions. The header identifies the tournament and match date.
Terrain artwork and ability-effect geometry are not present in the export.
Probabilities stay unavailable until saved predictions from an approved model run
are supplied. The server never loads models. See [the pipeline](docs/pipeline.md)
for additional data, prediction integration, replay semantics, and proposed work.

## What changed from the class project

- Replaced whole-round inputs with observation windows that exclude future events
  and post-outcome information. The current rolling task includes defuse progress
  only while the round is still undecided.
- Corrected player team lookup to use GRID's `state.teamId`, captured the series
  ID from the event envelope, and moved the time origin past buy time.
- Added source validation, explicit exclusion counts, and tests for cutoff
  boundaries, corrections, side changes, and split overlap.
- Replaced random round splits with independent series partitions. A separate
  smoke-test mode holds out a map and makes no test-set claim.
- Added training-prior, snapshot logistic, and history-summary logistic baselines
  alongside the temporal CNN. Rolling samples are weighted by round; CNN checkpoint
  selection uses validation log loss averaged by round. The test partition is
  evaluated afterward.
- Separated split and model seeds, and bound prediction exports to run manifests,
  source hashes, model artifact hashes, feature schemas, cutoffs, and partitions.
- Simplified training to local PyTorch and local safetensors checkpoints, removing
  Hub integration and the Transformers/Datasets training wrappers.

The original full-round checkpoints are incompatible with this feature schema.
They and the generated class reports are not included here.

## Layout

```text
data/                   GRID event data and provenance notes
data_processing.py      JSONL parsing, window construction, and data audit
evaluation.py           Group partitions and binary classification metrics
baseline_features.py    Temporal summaries of the same one-second observations
baseline_models.py      Baseline fitting and JSON coefficient storage
experiment.py           Run manifests and prediction provenance checks
rolling_features.py     Multimodal rolling observations and coverage audit
rolling_model.py        Causal temporal CNN (optional ML runtime)
rolling_experiment.py   Rolling baselines, metrics, and provenance
train_rolling.py        Grouped, round-weighted rolling experiment
predict_rolling.py      Time-indexed exports from an approved rolling run
model.py                CNN architecture and local checkpoint I/O
train.py                Training, baselines, run manifest, and held-out evaluation
predict.py              Offline prediction export from a completed run
replay_data.py          Compact player-state and ability replay projection
serve.py                Read-only localhost server (standard library)
viewer/                 Local HTML, CSS, and JavaScript replay interface
tests/                  Standard-library data and evaluation regression tests
tests_ml/               Optional ML fitting, probability, and checkpoint tests
docs/methodology.md      Experimental protocol and limitations
```

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

## Current limits and next work

The available sample contains 44 completed rounds across Breeze and Ascent from
one series, plus a canceled initial game. One completed round ends by the
30-second cutoff. The saved [data audit](docs/data-audit.json) records the counts
and source hash. No new CNN has been trained or evaluated during the cleanup.
The old model's metrics should not be reused for this task.

Next steps are owner review of gameplay assumptions, inspection of reconstructed
rolling features against source events, and ML validation in an approved environment.
Independent series would be needed for a generalization study. Compare the temporal
CNN with its baselines across model seeds on fixed partitions. Repeated teams,
maps, and patches can still create dependence across series.

## Data and local artifacts

The data was supplied through the Riot API during a hackathon; access ended after
the event. The owner has confirmed permission to redistribute the included file.
See [data provenance](data/README.md). The data is not a license grant for other
datasets. No pretrained weights, pickle checkpoints, API credentials, downloads,
telemetry, or upload integrations are used by this implementation. Dependencies
still require the usual review before installation or execution.

Personal report drafts are maintained outside the repository. This repository
contains the implementation and its technical documentation.
