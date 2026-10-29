# Rolling temporal model: feature design

The task is to estimate attacker round-win probability using a 20-second history,
starting 20 seconds after buy time and updating every five seconds. The first
windows are `[0,20)`, `[5,25)`, `[10,30)`. Each contains twenty one-second bins.
There is no prediction during warm-up, no use of future observations to fill gaps,
and no prediction at or after a recorded terminal event or a decisive elimination.
Eliminating all attackers only decides an unplanted round; eliminating all defenders
decides either phase. Revive/respawn variants require adapter review.

This is analysis from the full recorded spectator state. Enemy positions are not
restricted to what a player or team could see. The model cannot establish the
causal benefit of an action or infer confirmed information from a utility cast.

## Gameplay priorities

The initial priorities, selected by the project owner, are:

1. Utility combinations before contact.
2. Rotations after information.
3. Trades.

Features preserve raw observable ingredients instead of assigning tactical success
labels. Recon or flash use does not become “enemy revealed” or “enemy blinded.”
There is no damage-event stream in this export, so a kill is an observed combat
event, not the start of contact. Position changes following information-related
casts can be studied, but confirmed information acquisition is not recorded.

## Feature contract

`rolling_features.schema()` is the authoritative versioned contract. It includes
feature order, a fixed ability vocabulary, map calibration, timing, spatial-grid
size, and staleness/trade thresholds. The initial version has 365 scalar channels
per second. Missing observations have explicit indicators; zero is not silently
treated as an observed coordinate. Individual/player/team identities and match
scores are excluded. Map identity is included to distinguish different geometry.

| Family | Recorded inputs | Important interpretation |
| --- | --- | --- |
| Ability use | Per-side counts for 28 named abilities plus an unknown bucket; per-ability mean caster x/y and located-cast count per bin | Cast origin, not trajectory, impact area, duration, reveal, suppression, or blind success |
| Combat | Kill/death counts per bin, cumulative kills, separate killer/victim grid cells | No continuous damage, crosshair, aim, or line-of-sight observations |
| Trades | Return-kill candidates and delay | A player kills an opponent who killed their teammate within the previous five seconds; not a provider trade annotation |
| Positioning | Fresh-player counts per 3×3 grid cell, centroid, dispersion, unknown counts, and position age | Grid cells are geometric bins, not verified sites or claims of map control |
| Economy | Initial team loadout and last reported loadout of living players | Weapon value does not identify the currently held weapon |
| Health/armor | Last reported totals, known fractions, and report age | Missing damage events mean these are not guaranteed current health totals |
| Objective/time | Planted flag, plant age, defusing, persistent defuse checkpoint, reported game clock and age, elapsed feed time | Feed time and game-clock reports remain distinct; the clock is not silently extrapolated |

Ability and kill location channels are separate: an opponent can be killed in a
different cell from the shooter. Events with unknown/stale locations retain their
nonspatial channels and explicit unknown-location counts. Multiple casts of the
same ability within one bin retain their count but average their origin coordinates.
The representation loses within-bin ordering and exact player-to-player geometry.
These are deliberate prototype limits, not evidence that such detail is irrelevant.

## Freshness and source audit

Only coordinates reported within the last two seconds contribute to current
positioning channels. Stale positions remain visible as last-known positions in
the replay, while the model masks them. The distinction matters because this
feed reports positions at events, with longer gaps during quieter periods.

The initial audit found 555 overlapping windows across 44 rounds and one series.
At one-second observation points, position age has a median of about 2.31 seconds,
90th percentile 11.02 seconds, and maximum 33.52 seconds. Roughly 46% of living
player observations meet the provisional two-second freshness rule. The saved
`rolling-audit.json` records the exact source hash, counts, and coverage.

The ability vocabulary and calibration are fixed source-schema choices based on
the available export. New abilities enter the `other` bucket; unknown maps retain
nonspatial inputs and mark positions unknown until calibrated. Neither fallback
establishes validity on new agents, maps, patches, or tournaments.

## Models and evaluation

The temporal CNN projects the 365 channels to eight hidden channels with a 1×1
convolution. Four residual causal convolutions use kernel width three and
dilations 1, 2, 4, and 8, with ReLU and dropout 0.1. The last hidden state feeds
the outcome head. The theoretical receptive field spans 31 bins and covers the
entire 20-second input; padding occurs only to its left. There are 3,746 trainable
parameters by architecture definition. This runtime has not been executed yet.

Comparisons are a training prior, logistic regression on the current snapshot,
and logistic regression on the snapshot plus each channel's window mean. The
last comparison has the same ingredients without their within-window ordering.
It is a baseline, not a complete set of controls for nonlinear model capacity.
Future ablations should separately test temporal kernels and feature families.

Every round has total sample weight one, divided over its windows. The logistic
scalers and classifiers are fitted on training windows with those weights. The
CNN's loss uses the same weighting. Series grouping occurs before model fitting;
all windows from a round remain together. Smoke tests hold out a map within the
single series and have no test partition.

CNN selection uses mean validation log loss computed per round, then averaged
over rounds. Results also report Brier score averaged the same way and metrics at
each elapsed prediction time, with sample counts. Later times include only rounds
still active then. They are different subsets, so improving late-round accuracy
alone does not demonstrate a better model. 555 windows are still only 44 dependent
rounds from one series, not 555 independent examples.

## Provenance and viewer

Rolling runs use a separate schema and entry point from the earlier fixed-cutoff
experiment. A prediction is identified by `(series, game, round, prediction time)`.
Each row binds its window start, feature-contract hash, source hash, model artifact
hashes, partition, run ID, and run-manifest hash. The viewer rejects duplicate
timestamps, mismatched files, and predictions at ineligible times. It reads JSON
only and never opens fitted artifacts.

The viewer advances its observed window every five seconds, exposes ability-use
counts and trade candidates, and optionally overlays victim locations from that
window. Probability curves show only saved predictions reached by the scrubber;
future values stay hidden. An older prediction can remain visible with its own
timestamp if a later prediction is missing. No probability is invented while the
ML runtime awaits approval.

## Gameplay decisions for owner review

- Is a five-second return-kill rule a useful first trade proxy, or should the
  experiment compare several delays? It currently allows a different teammate
  to make the return kill and does not require proximity or line of sight.
- Which ability sequences should be inspected manually before contact? Cast
  timestamps and origins can support examples; the feed cannot verify effects.
- Which observed relocation patterns would you call a rotation rather than local
  repositioning? Named areas/paths should be annotated with your input before
  labeling either in the dataset. The current model consumes geometric changes.

No tactical labels or winning-strategy rules should be added from model output
alone. These questions guide the representation and manual audits, not a search
for favorable validation scores.
