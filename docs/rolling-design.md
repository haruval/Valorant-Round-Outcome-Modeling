# Rolling gameplay features and temporal model

The task estimates attacker round-win probability from twenty one-second bins,
starting 20 seconds after buy time and updating every five seconds. Windows are
`[0,20)`, `[5,25)`, `[10,30)`, and so on. Events exactly at the endpoint are
excluded. There is no output during warm-up, at/after a recorded outcome, or after
a decisive elimination. Eliminating attackers decides an unplanted round;
eliminating defenders decides either phase. Revive variants need adapter review.

This is retrospective analysis of recorded spectator state. Enemy coordinates
are not restricted to what a team could see. The model estimates associations,
not the causal effect of a strategy.

## Input contract: rolling-gameplay-v2

`rolling_features.schema()` binds feature order, the ability taxonomy, timing,
map calibration, freshness, trade thresholds, and movement rules. There are
**222 scalar channels per bin**, down from 365 in v1.

Map identity, team/player identities, agent identity, and match score are excluded.
Map names remain necessary metadata for coordinate calibration, partitioning, and
the replay. Coordinates, map geometry, and compositions can still serve as indirect
map cues. Identically numbered grid cells on different maps do not necessarily
represent the same tactical area.

| Family | Inputs | Interpretation |
| --- | --- | --- |
| Shared utility | Per-side counts, mean caster x/y, located-cast counts for ten function tags; total/cumulative unique casts | Cast origin and function, not impact geometry, duration, or successful effects |
| Combat | Kills/deaths, cumulative kills, separate killer/victim 3×3 cells | No continuous damage, aim, or line-of-sight stream |
| Trade candidates | Return-kill counts and delays | An opponent kills a teammate and is killed in return within `0 < delay <= 5s`; no proximity requirement |
| Position | Fresh-player cells, centroid, spread, unknown fraction, report age | Geometric presence, not verified map control |
| Movement | Same-player report-pair coverage and mean normalized displacement; corresponding values after recon casts | Sparse paired reports, not route, velocity, or confirmed rotation |
| Recon timing | Known flag and clipped age of the team's latest recon-tagged cast | Does not establish that information was acquired |
| State | Alive counts, initial/alive loadout, reported health/armor with known fractions and age | Last reports can be stale; loadout value does not identify the held weapon |
| Objective/time | Plant state/age, defusing/checkpoint, reported game clock/age, elapsed feed time | No completed-outcome input; reported clock is not silently extrapolated |

Events with unavailable locations retain their nonspatial counts and explicit
unknown-location channels. Same-bin casts of a category retain counts but average
their origins. The input loses within-bin order, exact player interactions, and
spatial detail within cells.

### Provisional ability functions

The exact mapping lives in `ability_taxonomy.py` and is embedded in every schema.
Original names remain in replay evidence. The broad vision-impairment category
currently groups flash-drive and paranoia; this is a starting design for owner
review, not an assertion that their effects are interchangeable.

| Tag | Included abilities |
| --- | --- |
| vision_block | cloudburst, cyber-cage, dark-cover, poison-cloud, toxic-screen, viper's-pit |
| vision_impair | flash-drive, paranoia |
| recon | neural-theft, owl-drone, recon-bolt, spycam, zero-point |
| detection_deployable | alarmbot, trapwire, turret |
| damage | frag-ment, hunter's-fury, nanoswarm, shock-bolt, snake-bite, turret |
| mobility | from-the-shadows, shrouded-step, tailwind, updraft |
| suppression | null-cmd, zero-point |
| detain | lockdown |
| weapon_ultimate | blade-storm |
| other | Unknown ability names |

Tags describe selected cast functions, not every secondary effect. For example,
vulnerability, decay, sound cues, duration, and activation state are not modeled.
Deploying a detection ability does not imply it triggered. Multi-tagged casts
increment each relevant category but only increment total casts once. Category
counts must not be summed to recover unique casts. Any taxonomy change creates
a different schema contract and requires new training.

### Movement and freshness

Current-position inputs accept coordinates at most two seconds old. A movement
pair requires the same living player, a new coordinate-report timestamp, fresh
endpoints, and at most two seconds between reports. At most one pair per player
per one-second bin is retained. Deaths and changes in the set of observed players
cannot masquerade as movement.

After-recon pairs additionally require both reports at/after the team's latest
recon-tagged cast and the second report within five seconds of that cast.
Movement is straight-line distance in normalized minimap coordinates. It is not
path length, is not normalized by elapsed time, and cannot distinguish teleporting.
A zero displacement without report-pair coverage means no measurement.

The audit still finds 555 windows from 44 rounds and one series. About 46% of
alive-player observation points meet the two-second freshness rule. Median
position age is 2.31s, p90 11.02s, maximum 33.52s. Sparse observations limit what
can be learned about rotations. The replay may connect last-known points across
larger gaps; the movement features deliberately do not.

## Training-only support and ablations

Before fitting, derive per-channel minimum and maximum using training windows
only. Mask every training-constant channel to zero in every partition. Save this
policy in the manifest, baseline JSON, and neural configuration. Inference checks
the same schema and policy; evaluation values never expand training support.
This prevents validation-only values from activating weights whose inputs never
varied during fitting. It cannot solve general distribution shift.

The standard-library [support audit](feature-support-v2.json) reports:

| Training → validation | Retained / 222 | Masked | Training-constant channels changing in validation |
| --- | ---: | ---: | ---: |
| Breeze → Ascent | 174 | 48 | 20 |
| Ascent → Breeze | 181 | 41 | 13 |

All models share these controls:

- `all`: all input families, subject to the training support mask.
- `state_only`: fourteen clock/objective, alive-count, and loadout channels.
- `no_utility`: remove utility counts/origins, recon timing, and movement-after-recon.
- `no_spatial`: remove locations, grids, freshness, and displacement; keep cast counts.
- `no_trades`: remove candidate-trade counts/delay; keep underlying kill observations.

Ablations overlap. Removing a proxy does not remove all information from which
the phenomenon might be inferred. For example, kills can still indicate trades.

## Models and controlled comparisons

The temporal CNN projects 222 inputs to eight hidden channels, followed by four
causal residual convolutions (kernel 3, dilations 1/2/4/8, ReLU, dropout 0.1) and
a two-class last-state head. Its theoretical receptive field is 31 bins and
covers the full 20-bin input. Runtime checks confirm **2,602 trainable parameters**.

Comparisons use the same input policy and equal total sample weight per round:

| Model | Inputs / architecture |
| --- | --- |
| Training prior | Weighted training prevalence |
| Snapshot logistic | Last observed bin |
| History logistic | Last bin plus each channel's window mean |
| Pooled neural control | Shared per-bin projection; four residual 8→12→8 pointwise blocks; mean history plus last bin |
| Temporal CNN | Causal convolution across ordered bins |

The pooled control has 2,666 parameters, about 2.5% more than the CNN.
With its last bin fixed, permuting the preceding nineteen bins leaves eval-mode
output unchanged. Time-valued channels remain present in both models. This tests
the temporal architecture more fairly than a linear baseline alone, but does
not perfectly isolate every difference in capacity or optimization.

All windows from a round remain together. Standard runs split by series before
fitting. With this single series, `--smoke-test --validation-map ascent` or
`breeze` explicitly selects the held-out map and creates no test partition.
Neural checkpoint selection uses round-mean validation log loss. Histories save
training and validation metrics per epoch; reports include round-mean Brier score
and elapsed-time metrics with sample counts.

The two map directions are development comparisons, not independent test folds.
Later windows include only rounds still active at that time, so their metrics
cannot be compared as if they covered the same examples.
See [the validation results and subsequent study protocol](controlled-experiment.md).

## Provenance, compatibility, and verification

Each prediction binds its source, feature schema, manifest, artifact hashes,
partition, round, and observation time. The viewer reads JSON only, rejects
mismatched data/times, and never invents probabilities. It shows the model's
feature set and masked-channel count for v2 runs. Raw ability names and report-pair
counts support inspecting the feature evidence.

The [frozen v1 schema](../schemas/rolling-multimodal-v1.json) supports existing
prediction JSON in the viewer. Its 365-channel inputs and historical probabilities
remain v1; new inference rejects v1 weights. The completed v1 run is documented
in [historical ML validation](ml-validation.md).

The standard-library suite now passes 74 tests, including review-window,
utility/movement-prototype and comparison-provenance regressions. The independent event-count audit
passes all 16,650 comparisons. It shares the declarative ability mapping but
counts source events separately; it does not validate tactical labels or model
performance. All four optional ML tests passed on October 31, 2026, and both
three-epoch neural runs completed. Each run's 555 saved predictions matched
exactly after reload. The CNN did not beat the baselines in this bounded check.
The subsequent [12-run, 20-epoch comparison](controlled-experiment.md#completed-12-run-study)
improved both neural architectures over the logistic baselines at their selected
validation checkpoints, but did not establish a consistent CNN advantage over
the pooled control. All 6,660 study prediction rows matched exactly after reload.
The subsequent [16-run ablation study](controlled-experiment.md#completed-feature-ablations)
also completed, with all 8,880 new prediction rows matching after reload.
Spatial removal improved both neural architectures in both directions, while
snapshot logistic without spatial features had the best Breeze log loss.
Generalization and a stable temporal-CNN advantage remain unestablished.

The [eight-run repeat](controlled-experiment.md#completed-reduced-input-seed-check)
adds seeds 43 and 44 to the reduced-input comparison. Spatial removal improved
11 of 12 matched log losses over three seeds; the pooled control had lower mean
loss than the CNN on both maps. All 4,440 new rows reproduced after reload.
The separate `entry-support-prototype-v0` has 34 review-only channels and leaves
the trained v2 contract unchanged. Its rules are in the [`entry_support.py`](../entry_support.py) contract.
