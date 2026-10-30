# VALORANT Round-Outcome Modeling

Estimate attacker round-win probability from recent utility, combat, positioning,
economy, and objective state. A temporal CNN receives 20 seconds of observations
and updates every five seconds. This began as an undergraduate ML final project
and is now an independent study of temporal modeling with sparse event data.

**Current status:** revised `rolling-gameplay-v2` features pass 58 standard-library
tests and the full event audit. Its CNN and neural control are implemented but
have **not** been trained or runtime-tested. The earlier v1 three-epoch run passed
ML checks; its CNN did not outperform the logistic baselines.
[Historical results](docs/ml-validation.md#completed-run) are separate from
[the pending v2 experiment](docs/controlled-experiment.md).

The dataset contains 44 rounds from Breeze and Ascent in one FURIA–NRG series:
VCT Americas Kickoff, February 16, 2024. Its 555 overlapping windows are not
independent samples. There is no independent test set in this sample.

## Research question

Does a temporal CNN learn useful event sequences beyond current state and
history summaries? Gameplay priorities are utility combinations before contact,
movement following information-related casts, and trades.

The v2 model has 222 input channels and a designed 2,602 parameters. It uses
shared ability functions instead of ability-specific channels and excludes map,
player, team, and agent identities. Cast locations, kill origin/victim locations,
alive counts, economy, reported health, and objective state remain available.
Coordinates still reflect map geometry; removing the map indicator does not make
the representation map-invariant.

Comparisons include a training prior, snapshot logistic regression, history-summary
logistic regression, and a similarly sized neural control that pools independently
encoded bins. Feature ablations isolate utility, spatial observations, and trade
proxies. Every model uses the same training-only support mask: channels constant
in training stay zero during validation and inference.
See [feature definitions and limits](docs/rolling-design.md).

The feed records casts, not successful blinds or reveals. Sparse position reports
support limited displacement measurements, not verified rotations or map control.
This is retrospective analysis of spectator state. Associations in predictions
cannot establish that an action caused a change in win probability.

## Visualizer

No dependencies or model loading are needed to inspect the events:

```bash
python3 -E -S -B serve.py --data-dir data --port 8767
```

Open [the visualizer](http://127.0.0.1:8767/). Select a map and round, then play or
scrub the timeline. It shows calibrated minimaps, player states, raw ability
names, kills, observation windows, and position-report coverage. The v2 display
also counts matched position reports and those following recon casts. Hover an
ability-use chip to inspect its provisional function tags.

Saved probabilities require prediction JSONL and its matching run manifest.
The server reads JSON and never loads models. Provenance identifies the schema,
partition, hashes, and (for v2 runs) feature set and masked-channel count.
The retained v1 output still uses its original 365-channel contract:

```bash
python3 -E -S -B serve.py --data-dir data --port 8766 \
  --rolling-predictions runs/ml-validation-20261029/rolling-smoke-5s/predictions.jsonl \
  --rolling-run-manifest runs/ml-validation-20261029/rolling-smoke-5s/run.json
```

That ignored local run directory is not included in a fresh clone. New inference
code rejects v1 checkpoints; old probabilities are not presented as v2 results.

## Validation

Standard-library checks run on Python 3.9+:

```bash
python3 -E -S -B -m unittest discover -s tests -v
python3 -E -S -B audit_rolling.py data
python3 -E -S -B audit_inputs.py data
```

All 16,650 v2 event-channel comparisons pass across 555 windows.
[The event audit](docs/feature-audit-v2.json) and
[training-support audit](docs/feature-support-v2.json) bind results to the source
hash. [Review examples](docs/feature-review.md) identify concrete sequences for
checking gameplay assumptions.

ML execution uses NumPy, scikit-learn, PyTorch, and safetensors. No new dependency
is needed for v2. The reviewed environment and v1 artifacts remain in
`runs/ml-validation-20261029/`. The [pending execution plan](docs/controlled-experiment.md)
lists exact commands, risks, and separate outputs. None of those ML commands has
been executed. Owner approval is required before ML execution or installation.

## Changes from the class project

- Replaced whole-round inputs with causal windows excluding future events and
  resolved outcomes; time starts after buy time.
- Corrected team lookup and series IDs, validated source data, and tested
  corrections, side changes, timestamp boundaries, and partition overlap.
- Grouped windows by series for evaluation. The two-map development mode keeps
  each map together and makes no independent-test claim.
- Added baselines, a neural control, feature ablations, and equal total weight
  per round. Validation log loss averaged by round selects checkpoints.
- Bound predictions to data, schemas, input policies, artifacts, and partitions.
- Replaced Hub/Transformers wrappers with local model code and safetensors.
  Original whole-round checkpoints and generated reports are excluded.

The [earlier fixed-cutoff pipeline](docs/fixed-cutoff.md) remains available.
New exports can enter the [ingestion pipeline](docs/pipeline.md). Independent
matches are needed before claiming generalization; more windows from these
rounds do not resolve that limitation.

## Source layout

| Files | Purpose |
| --- | --- |
| `data_processing.py`, `data/` | Event ingestion, validation, provenance |
| `rolling_features.py`, `ability_taxonomy.py` | Observations and shared cast functions |
| `rolling_inputs.py` | Training-only masks and feature-family controls |
| `rolling_model.py`, `rolling_experiment.py` | Neural models, baselines, metrics |
| `train_rolling.py`, `predict_rolling.py` | Approved-runtime training and export |
| `audit_rolling.py`, `audit_inputs.py` | Standard-library audits |
| `serve.py`, `replay_data.py`, `viewer/` | Local replay and prediction inspection |
| `tests/`, `tests_ml/` | Separate standard-library and optional ML tests |
| `schemas/` | Historical contracts for JSON replay |

The owner confirmed redistribution permission for the supplied hackathon data.
See [data provenance](data/README.md) and map artwork credits in the visualizer.
Personal report drafts remain outside the repository.
