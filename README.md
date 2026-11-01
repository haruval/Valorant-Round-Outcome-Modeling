# VALORANT Round-Outcome Modeling

Estimate attacker round-win probability from recent utility, combat, positioning,
economy, and objective state. A temporal CNN receives 20 seconds of observations
and updates every five seconds. This began as an undergraduate ML final project
and is now an independent study of temporal modeling with sparse event data.

**Current status:** 74 standard-library tests pass. The unchanged ML execution
code previously passed four runtime tests and the full event audit. A
[three-seed follow-up](docs/controlled-experiment.md#completed-reduced-input-seed-check)
found that removing current spatial features improved log loss in 11 of 12
matched neural comparisons. The pooled neural control had better average log
loss than the temporal CNN on both maps. A reliable benefit from temporal
convolution remains unestablished.

The [full-input study](docs/controlled-experiment.md#completed-12-run-study),
[feature ablations](docs/controlled-experiment.md#completed-feature-ablations),
and follow-up retain every outcome, including weaker runs. All 4,440 new
follow-up prediction rows reproduced exactly after reload. The visualizer
compares separately verified validation runs and includes an
[untrained utility/movement prototype](docs/feature-review.md#utilitymovement-prototype)
for domain review. Earlier results remain in the [v1 record](docs/ml-validation.md#completed-run).

The dataset contains 44 rounds from Breeze and Ascent in one FURIA–NRG series:
VCT Americas Kickoff, February 16, 2024. Its 555 overlapping windows are not
independent samples. There is no independent test set in this sample.

## Research question

Does a temporal CNN learn useful event sequences beyond current state and
history summaries? Gameplay priorities are utility combinations before contact,
movement following information-related casts, and trades.

The v2 model has 222 input channels and 2,602 parameters. It uses
shared ability functions instead of ability-specific channels and excludes map,
player, team, and agent identities. Cast locations, kill origin/victim locations,
alive counts, economy, reported health, and objective state remain available.
Coordinates still reflect map geometry; removing the map indicator does not make
the representation map-invariant.

Comparisons include a training prior, snapshot logistic regression, history-summary
logistic regression, and a similarly sized neural control that pools independently
encoded bins. Feature ablations remove groups of utility, spatial observations,
and trade proxies; other inputs may retain correlated information. Every model
uses the same training-only support rule: channels constant
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
To inspect the study's seed-42 CNN and baseline predictions (Breeze training,
Ascent validation; the demonstration seed was fixed independently of scores):

```bash
python3 -E -S -B serve.py --data-dir data --port 8769 \
  --rolling-predictions runs/gameplay-v2-study/ascent-all-temporal_cnn-seed42/predictions.jsonl \
  --rolling-run-manifest runs/gameplay-v2-study/ascent-all-temporal_cnn-seed42/run.json
```

That ignored local run directory is not included in a fresh clone. The original
v1 prediction JSON remains viewable under its 365-channel contract. New inference
code rejects v1 checkpoints; old probabilities are not presented as v2 results.

With the retained full-input and ablation artifacts, compare both neural models
and their logistic baselines on the same rounds:

```bash
python3 -E -S -B serve.py --data-dir data --port 8771 \
  --comparison-config docs/visualizer-comparison.json
```

Open [the comparison visualizer](http://127.0.0.1:8771/). Each map uses its
validation runs; seed 42 is fixed for inspection independently of scores.
Changing **Experiment** changes the complete run and its provenance together.
**Review cases** includes Ascent round 13 at 25s, focused on mwzera, plus examples
with and without nearby allied casts. It hides probabilities on selection;
uncheck **Hide probabilities** after forming an initial assessment. Expand a
movement pair to inspect casts or highlight its observed displacement on the map.
These prototype features do not affect any displayed prediction.
The pinned JSON configuration requires local ignored artifacts; a fresh clone
can use the feature-only command above without ML dependencies.

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
checking gameplay assumptions. The standard-library
[disagreement review](review_windows.py) reconstructs six validation cases with
outcomes and probabilities withheld by default. Its saved
[evidence](docs/disagreement-review.json) includes source events, position ages,
input-mask exclusions and run provenance. Initial owner guidance covers entry
movement supported by utility and information-driven rotations; case assessments
remain incomplete.

ML execution uses NumPy, scikit-learn, PyTorch, and safetensors. No new dependency
is needed for v2. The reviewed environment and v1 artifacts remain in
`runs/ml-validation-20261029/`. New validation artifacts live separately in
`runs/gameplay-v2-validation/`; the 12-run comparison lives in
`runs/gameplay-v2-study/`; the ablation study lives in
`runs/gameplay-v2-ablations/`. The [execution record](docs/controlled-experiment.md)
lists results, exact commands, risks, and removal instructions. The
[16-run ablation plan](docs/ablation-plan.json) is complete, with
[all results](docs/ablation-results.json) retained. The
[eight-run follow-up plan](docs/reduced-seed-plan.json) is also complete, with
[three-seed results](docs/reduced-seed-results.json) and separate artifacts in
`runs/gameplay-v2-seed-check/`. Player-level utility/movement relationships are
implemented as a standard-library review prototype, with no model training or
feature-contract changes. Additional ML execution or installation needs owner approval.

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
| `entry_support.py`, `comparison_data.py` | Untrained relationship evidence and separately verified comparison runs |
| `serve.py`, `replay_data.py`, `viewer/` | Local replay and prediction inspection |
| `tests/`, `tests_ml/` | Separate standard-library and optional ML tests |
| `schemas/` | Historical contracts for JSON replay |

The owner confirmed redistribution permission for the supplied hackathon data.
See [data provenance](data/README.md) and map artwork credits in the visualizer.
Personal report drafts remain outside the repository.
