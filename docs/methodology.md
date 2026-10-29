# Fixed-cutoff experimental protocol

This documents the earlier 30-second task. The current rolling temporal study is
specified in [rolling-design.md](rolling-design.md).

## Inputs and labels

One example is one completed round with a known winner, a live-start event, and
an outcome strictly after the observation cutoff. Label `1` means the attacker
won; `0` means the defender won. Sides are read anew at each round start.

The default window is 30 seconds of feed wall time after buy time ends. There
are no padded post-round frames. All eligible examples have the same window.
For bin `t`, updates with relative timestamps `< t + 1` are applied. An event at
exactly 30 seconds is excluded from a 30-second input. Round-win events supply
labels only. Explicit defuse/explosion events at or before the cutoff also
exclude the round, even if its win notification arrives later.

| Feature | Definition |
| --- | --- |
| `elapsed_fraction` | Bin endpoint / observation seconds |
| `atk_alive`, `def_alive` | Remaining roster members / 5 |
| `atk_kills`, `def_kills` | Opponent kills since live start / 5 |
| `spike_planted` | Plant observed before the bin endpoint |
| `atk_loadout`, `def_loadout` | Team loadout at live start / 25,000 |
| `atk_ability_uses`, `def_ability_uses` | Uses since live start / 20, capped at 2 |
| `first_kill_known` | Whether an opponent kill has occurred |
| `first_kill_atk` | Whether the attacker made that first kill |

Normalization constants are fixed engineering choices, not statistics fitted on
the evaluation data. The first-kill pair distinguishes “no kill yet” from
“defender first kill.” Suicide and teamkill events reduce living player counts
without counting as opponent kills. Roster sets prevent repeated death updates
from decrementing a count twice.

## Source integrity

The parser requires ordered envelopes, timezone-aware timestamps, unique event
IDs, known round sides and winners, and resolvable player teams. Duplicate round
identities across input files fail instead of appearing in multiple partitions.

Retractions (`!` events) exclude their affected game; the parser does not attempt
to reverse unknown event semantics. An unassignable correction fails parsing.
Series invalidation after completed rounds also fails and needs source review.
Incomplete rounds, rounds without a live start, rounds ending by the cutoff,
and explicitly paused rounds are counted and excluded. Unexpected revive or
respawn events within the observation window fail pending schema validation.

The supplied sample includes an abandoned initial game. The audit records its
corrections and unfinished round separately from the two completed maps.

## Splits and selection

Normal runs group by series ID, shuffle sorted group IDs with the specified
`--split-seed`, and allocate at least one group each to validation and test (approximately
20% of groups each for larger datasets). Remaining groups are used for training.
All rounds from a series stay in one partition. Splits are not stratified;
training requires both classes. ROC AUC and balanced accuracy are `null` for a
partition with only one class. The manifest records every round assignment.

Smoke tests group by `(series_id, game_id)`, reserve validation maps, and omit a
test partition. They are labeled `development_smoke_test` in saved results.

The optional CNN uses Conv1d channels 8 and 16, kernel width 3, ReLU and dropout
0.1 per block, global mean pooling, and a linear head: 730 trainable parameters.
There is no batch normalization or max pooling. Its five-bin receptive field
captures short local patterns before pooling. This is fixed-window prediction;
convolutions can use both directions within the already-observed window.
Adam optimizes cross entropy. Initialization, batch shuffling, and NumPy use
`--model-seed`, independently of the split seed. The checkpoint with lowest
validation log loss is retained. Test labels
are used for final metrics only, after selection.

Baselines fit only the training partition:

- Majority class: a constant hard prediction (ties choose defender).
- Training prior: a constant empirical attacker-win probability.
- Logistic snapshot: the 12 features from the last observed bin, standardized
  inside a training-only pipeline, then logistic regression with fixed `C=1`.
- Logistic temporal: the same snapshot plus the eight summaries below, with
  its own training-only scaler and logistic regression with fixed `C=1`.

Training fits the three probability baselines by default. `--include-cnn` adds
the CNN; it never replaces the baselines. Majority is an additional metric
reference, not a probability model offered in the viewer. No baseline
hyperparameters are selected using validation or test labels.

Temporal summaries use the exact same binned sequence available to the CNN:

| Additional feature | Definition |
| --- | --- |
| First-kill time fraction | Endpoint of first bin containing an opponent kill / window |
| Elimination observed | Any decrease in living players, including suicide/teamkill |
| Time since last elimination | (Window − endpoint of last bin containing a death) / window |
| Time since plant | (Window − endpoint of first planted bin) / window |
| ATK/DEF kills in last 5s | Change in normalized cumulative kills over `[cutoff − 5, cutoff)` |
| ATK/DEF ability uses in last 5s | Change in normalized cumulative uses over the same interval |

Missing event times use zero and the corresponding presence flag (first-kill
known, elimination observed, or spike planted). Timing is quantized to bin ends,
not subsecond event timestamps. Recent activity retains the sequence's fixed
normalization and ability-count cap; it cannot recover uses above that cap.
Both logistic models are saved as JSON scaler/coefficient vectors, not pickled
scikit-learn objects. Optional ML tests compare exported probabilities against
scikit-learn; those tests have not yet been executed.

The run manifest records feature schemas, input hashes, each round's source and
partition, model artifact hashes, split/model seeds, and implementation hashes.
Seeds alone do not freeze a split when new groups are added. Existing assignments
must be preserved separately for a long-lived benchmark.

Metrics are accuracy, balanced accuracy, attacker F1, ROC AUC, log loss, Brier
score, and a confusion matrix with rows=true and columns=predicted, ordered
defender then attacker. Probability 0.5 predicts defender consistently.
Log loss clips probabilities to `[1e-7, 1-1e-7]`.

## Interpretation limits

- Only one real series is currently included. No independent test score can be
  obtained from it, and no performance claim is made.
- The task covers rounds still active at the chosen cutoff. Counts and exclusion
  rates must accompany results when comparing different cutoffs.
- Feed timestamps are not guaranteed to measure game-clock time. Pauses without
  explicit state flags, delays, and delayed win notifications remain risks. An
  elimination before the win event can reveal an already-decided outcome;
  manual alignment with the feed is still required before claiming prediction.
- Team loadout uses the end-of-buy-time snapshot when present, otherwise the
  round-start snapshot. Missing or stale source snapshots require inspection.
- Correction handling is deliberately conservative and can discard an entire
  otherwise usable map. It is not a general GRID correction engine.
- Ability semantics and revive variants have not been validated across agents,
  patches, or all GRID schemas. Missing events can invalidate derived counts.
- Series holdouts can still share teams, players, maps, patches, and tournament
  conditions. More data and chronological/team-level evaluation are needed.
- The current audit and metric calculations prioritize small, inspectable
  experiments. Event data and tensors are held in memory; pairwise ROC AUC is
  quadratic in partition size. Large-scale ingestion is future work.
- Choosing cutoffs, architecture, or features after inspecting a test score
  contaminates that test set. Keep those choices on training/validation data.

## Implementation references

Grouping follows the same independence principle described in
[scikit-learn's grouped cross-validation documentation](https://scikit-learn.org/stable/modules/cross_validation.html).
The code uses a small standard-library group partitioner so split invariants can
be tested without importing the ML runtime.

Checkpoints use the local `save_file` / `load_file` tensor APIs documented by
[Safetensors](https://huggingface.co/docs/safetensors/main/en/api/torch).
There is no fallback to pickle or remote model resolution. This is not a reason
to execute unknown dependencies or load untrusted model artifacts without review.
