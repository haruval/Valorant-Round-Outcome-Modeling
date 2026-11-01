# Gameplay feature review

The five-second return-kill rule was accepted by the project owner on October 29,
2026. The two-second position freshness rule remains an experimental choice.

The standard-library [event audit](../audit_rolling.py) compares 30 event channels
per eligible window with direct counts and kill pairs from the source events.
It uses roster membership rather than the feature extractor's team lookup and
does not use the extractor's projected event evidence as its reference.
All 16,650 comparisons pass across 555 windows and 44 rounds under `rolling-gameplay-v2`. The completed
round streams contain 1,102 ability uses, 289 enemy kills, and 59 candidate trades;
these source totals include events outside eligible observation windows.

Reproduce the check without installing packages or loading models:

```bash
python3 -E -S -B audit_rolling.py data
```

[The saved v2 result](feature-audit-v2.json) includes source hashes, event IDs, counts,
and selected review timelines. This checks the transformation of recorded events,
not whether a tactical interpretation is correct. Round extraction is shared
with the training pipeline and has its own regression tests. This audit does not
independently validate round boundaries, map calibration, or position freshness.
The shared ability-tag mapping is used by both implementations; the audit checks
counting, not whether the provisional taxonomy is tactically useful. The earlier
[v1 audit](feature-audit.json) remains as a historical record.

## Disagreement review after the 12-run study

[Six reconstructed cases](disagreement-review.json) contain only evidence before
each prediction endpoint. Outcomes and probabilities are withheld by default;
the cases were selected by disagreement, so they are not an unbiased sample.
Initial owner guidance is recorded below; case assessments remain incomplete.
Names and agents identify replay evidence and
are not added to the model inputs.

| Map / round | Endpoint after buy time | State at endpoint | Review focus |
| --- | ---: | --- | --- |
| Ascent / 13 | 25s | 5v5, unplanted; 11 ATK / 7 DEF casts in the window | Entry out of B main supported by allied utility |
| Ascent / 1 | 50s | 4v3, unplanted; 0.467s return kill | Trade context and subsequent space |
| Ascent / 7 | 55s | 5v5, unplanted; starting loadout 8,550 / 25,200 | Economy versus utility and position |
| Breeze / 23 | 115s | 2v3; planted 29.761s earlier | Post-plant time, positioning and recent deaths |
| Breeze / 13 | 20s | 3v4, unplanted; 0.241s return kill | A fast trade with a remaining player disadvantage |
| Breeze / 5 | 25s | 4v4, unplanted; 11 ATK / 3 DEF casts | Utility burst and kills on different opponents |

Alive counts and loadouts are ATK / DEF. Events belong to `[endpoint−20, endpoint)`.
Fresh-position coverage across these windows ranges from 25% to about 80%; a
fresh endpoint snapshot does not establish continuous tracking throughout the
window. Cast positions are the caster's reported position, not the location of
the ability's effect. Recon casts do not establish what information a team gained.

The evidence includes source event IDs, earlier kills/objectives, living players'
last position ages, candidate trades, training-mask exclusions, window hashes,
and both models' run and manifest hashes. Every candidate is checked against
the matching **validation** prediction from each architecture. This is an input
audit, not an attribution method for explaining neural probabilities.

Reproduce it with the standard library and retained study JSON; no model loading
or inference occurs:

```bash
python3 -E -S -B review_windows.py > /tmp/valorant-disagreement-review.json
```

Add `--reveal` to include saved probabilities and eventual outcomes **after**
recording an initial assessment. The script refuses changed study source code,
data, manifests, or candidate probabilities. Two regression tests check the
half-open boundary/future-outcome isolation and nonzero constant-channel masking.

The [Ascent validation viewer](http://127.0.0.1:8769/) uses the fixed seed-42 CNN.
Its Breeze predictions are training predictions. The corresponding
[Breeze validation viewer](http://127.0.0.1:8770/) uses the opposite training map:

```bash
python3 -E -S -B serve.py --data-dir data --port 8770 \
  --rolling-predictions runs/gameplay-v2-study/breeze-all-temporal_cnn-seed42/predictions.jsonl \
  --rolling-run-manifest runs/gameplay-v2-study/breeze-all-temporal_cnn-seed42/run.json
```

Those viewers show probabilities. Use the saved review JSON for an initial
assessment without seeing those values, and avoid scrubbing beyond the endpoint.
Neither viewer loads weights. The pooled-control values remain in the completed
study results and can be included in the review JSON with `--reveal`.

## Completed feature-family comparison

[The execution plan](ablation-plan.json) records all 16 command argument lists,
four existing full-input controls, frozen source/data hashes and expected channel
counts. All **16 runs / 320 epochs completed** after owner approval on October 31,
2026. [Results and interpretation](controlled-experiment.md#completed-feature-ablations)
include every ablation, logistic baseline and matched full-input control.
Both architectures use seed 42, both map directions and 20 epochs per run.
The original feature definitions remain fixed while reviewing the representation.

| Input set | Retained when validating Ascent | Retained when validating Breeze |
| --- | ---: | ---: |
| All (completed controls) | 174 | 181 |
| State only | 13 | 13 |
| Without utility | 94 | 91 |
| Without spatial features | 56 | 59 |
| Without trade features | 170 | 177 |

All networks keep the same 222-channel input width; excluded channels are zeroed.
State-only still receives a sequence of clock, objective, alive and loadout states.
Each result is compared with its matching seed-42 full-input control, with both
selected-epoch and epoch-20 metrics reported. Remaining channels may proxy removed
information, so these comparisons do not isolate causal gameplay effects.
One seed and one series cannot establish stable generalization. Any representation
changes suggested by owner review belong in a separate experiment.

Spatial removal improved both neural architectures in both directions. The
strongest Breeze result was the snapshot-logistic baseline without spatial
features; the strongest Ascent result was the CNN without spatial features.
The [three-seed follow-up](controlled-experiment.md#completed-reduced-input-seed-check)
confirmed lower reduced-input log loss in 11 of 12 matched runs. The pooled
control had the better mean on both maps. The representation prototype below
remains untrained and needs comparison with these reduced-input benchmarks.

## Owner guidance and representation follow-up

[Owner annotations](gameplay-annotations.json) identify the relationship in Ascent
round 13 as **mwzera entering out of B main off allied utility**. The review now
includes his timestamped position reports. Havoc's flash-drive is recorded at
13.961s. Mwzera's reported position changes between 15.116s and 16.577s, with
liazzi's paranoia at 16.325s and kon4n's recon-bolt at 16.577s. These are source
observations supporting review of the timing; the B-main interpretation comes
from the owner. No map polygon, successful blind/reveal, exact crossing time or
causal benefit has been established by this reconstruction.

Current inputs aggregate each team's movement and utility. They do not explicitly
connect an individual entry player's movement to a teammate's supporting cast.
A separate representation experiment should retain same-player movement pairs
and their timing relative to **allied** utility, separating self-casts from support.
Player IDs can join observations within a round without becoming identity inputs.
Named choke crossings need reviewed geometry before they can be labeled reliably.

For rotations, the owner prioritizes confirmed enemy sightings, then a teammate's
death and its location. The owner notes that a dying player may see or hear more
opponents joining the fight. This guidance does not establish an exact observed
enemy count for any death in the data.

[The export inventory](visibility-audit.json) checks all 2,268 envelopes / 2,390
events, including corrected or invalidated segments. It found no matching
visibility, spotting, reveal, audio, or facing field, state path, observed event
type, or declared event type. The full field/type inventory is saved for review;
this is not a guarantee about undocumented provider semantics. Reproduce it with:

```bash
python3 -E -S -B audit_visibility.py data
```

The current export therefore does not supply a verified team-visibility signal.
A future ingestion adapter should accept timestamped team sightings with an
explicit availability mask; absent sightings must stay unknown. For this sample,
the measurable follow-up is movement after a teammate's death, with the death's
location, fresh position-pair coverage and time since the event. It should be
labeled as a reaction proxy. Spectator enemy positions can describe nearby combat
context but cannot establish which opponents a team knew about. These proposed
features have not been added to the frozen v2 models or trained.

## Utility/movement prototype

`entry_support.py` implements `entry-support-prototype-v0` separately from the
222-channel trained contract. Its 34 numeric channels describe reported
same-player displacement, cast timing, distinct allied casters, self-casts,
three shared utility categories, and observation/spacing availability. It does
not assign successful support, entry, contact, rotation, or choke-crossing labels.

- Both position reports must be inside the 20-second window and at most two
  seconds apart. Missing coordinates, stale gaps and deaths cannot form a pair.
- Casts must be inside the window, from the same team, strictly before the
  second report and at most five seconds earlier. Self-casts remain separate.
  Casts at/before the first report are distinguished from casts between reports.
  Equal-time position reports are merged; casts at the second timestamp are
  excluded from that pair because within-timestamp ordering is not established.
- Displacement and support spacing use Euclidean feed-coordinate units, not
  meters, route length or absolute map cells. Spacing uses a living caster's last
  position at the first movement report, with a two-second freshness limit.
  Unknown spacing has a separate known-pair count. Feed-unit comparability and
  any training-only scaling need review before these channels are fitted.
- IDs join observations within a round and names identify review evidence.
  Neither enters the numeric channels. Relative geometry can still correlate
  with map/composition; dropping coordinates does not eliminate every map cue.
- One cast can precede several report pairs. Pair counts depend on report
  cadence and are not unique cast counts or counts of tactical entries.

[Seven cases](entry-support-review.json), selected without outcome/probability
filters, cover both maps: the owner's Ascent R13/25s example, movement near allied
casts, movement without recent allied casts, and windows with no usable moving
pairs. All 555 eligible windows were reconstructed. The JSON contains numeric
rows, timestamps, cast/position source IDs, missingness, implementation hashes,
and owner annotations from `gameplay-annotations.json`. Whole-window assessment
fields remain blank when only a specific cast has been assessed. Reproduce
without dependencies or model loading:

```bash
python3 -E -S -B entry_support.py --data-dir data > /tmp/valorant-entry-review.json
```

The prototype connects mwzera's movement to the flash/paranoia/recon sequence
discussed by the owner, while also surfacing temporally nearby casts that may be
unrelated. Review these as candidate relationships. Missing effect locations
prevent automatic confirmation that a cast helped the entry lane. The next
design decision is which relationships survive owner review, followed by a
small, separately approved feature experiment against reduced-input controls.
Death-triggered movement remains a later feature proposal; confirmed visibility
is still unavailable.

The owner subsequently identified Khalil's nanoswarm in Ascent R13/25s as a
lineup landing CT to help mwzera enter. The assessment is attached to cast event
`46a150db-1a13-49d7-9538-92a1b0e1b019` (12.270s) and supported player `57`,
separately from the source observations. CT is an owner-reported effect area;
the feed does not establish its landing location, active interval or damage.
This case supports retaining ranged lineup utility even when its caster is
far from the entry player. Caster spacing is context, not an eligibility filter
for useful support. The current prototype already retains such casts; its
numeric features, model inputs and saved probabilities are unchanged. The
assessment does not label every nanoswarm, every linked movement pair, or every
cast in this window as useful, and does not establish a causal win-probability gain.

The comparison viewer at port 8771 includes these cases and per-player filtering,
displacement highlights, a probability-hiding control, and full/reduced neural
run selection with the matching baselines. It loads JSON only. Standard-library
tests cover temporal boundaries, stale/missing positions, same-player joins,
caster death, equal-time reports, translation/rotation invariance, and comparison
provenance rejection. HTTP checks covered all 44 rounds and 2,220 validation
prediction rows. Browser layout and interaction have not been visually verified.

The earlier disagreement reviewer continues to check computational source and
map-calibration hashes; it now excludes server/UI files that do not participate
in evidence reconstruction. Historical study hashes are retained unchanged.

## Earlier windows to inspect

All examples are FURIA vs NRG, VCT Americas Kickoff 2024, February 16, 2024.
Times are feed seconds after buy time. Open the local visualizer, select the map
and round, and move the time bar to the endpoint shown below.

| Map / round | Window | Recorded contents | Review purpose |
| --- | --- | --- | --- |
| Breeze / 1 | 0–20s | 15 ability uses, no kills | Utility timing before the first recorded kill |
| Breeze / 1 | 20–40s | 5 ability uses, 2 kills, 1 candidate trade | A direct return-kill pair |
| Ascent / 1 | 0–20s | 11 ability uses, no kills | Utility ordering on different geometry |
| Ascent / 1 | 20–40s | 1 ability use, 2 kills, 1 candidate trade | Trade handling across the two team sides |
| Breeze / 23 | 85–105s | 13 ability uses, 2 kills, no candidate trades | Movement after utility, with intervening combat |

### Utility sequence

Breeze round 1 records mwzera's tailwind at 9.011s, liazzi's toxic screen at
9.216s, mwzera's cloudburst at 9.935s, havoc's zero point at 10.285s, and mwzera's
updraft at 10.585s. The replay preserves ability identities; v2 model inputs use
shared function tags (mobility, vision block, recon, suppression) in one-second bins:
the first three fall in `[9,10)`, and the next two in `[10,11)`.

The CNN can distinguish those bins, but cannot recover ordering within a bin.
“No kills” is not equivalent to “no contact”; this export has no continuous
damage-event stream. These casts alone do not establish a successful combination.

### Trade sequences

In Breeze round 1, crashies kills Khalil at 35.212s; kon4n kills crashies at
35.735s. The 0.523s return kill is an attacker candidate trade in the window
ending at 40s. Their event IDs are `d159d5a6-24cf-484f-95b7-855b568c8842` and
`716d2dbe-f0ab-420e-9024-b642552f230e`.

In Ascent round 1, havoc kills Demon1 at 31.682s; Marved kills havoc at 32.149s.
The delay is 0.467s. The implementation counts a return kill once even when the
opponent killed multiple teammates. Tests also cover exactly five seconds,
just over five seconds, and simultaneous kills. Proximity and line of sight are
not required by this initial rule.

### Movement candidate, not a verified rotation

In Breeze round 23, NRG attacks and FURIA defends. Ethan uses zero point at
90.329s. The accompanying snapshot reports Victor at world coordinates
`(7107.85449, -273.802765)`. A snapshot at 102.135s reports him at
`(747.243896, 1390.15918)`. The normalized minimap points are approximately
`(0.446, 0.336)` and `(0.562, 0.781)`, in grid cells 4 and 7 respectively.
Both timestamps fit in the 85–105s window.

The source event IDs are `a9170c58-0c33-4e1f-a094-b8acc7094c01` and
`fb4d510d-98b1-49a1-bd87-f9064ecf237f`. The latter is mwzera's updraft event;
its snapshot includes Victor's position. Victor also kills liazzi at 89.904s,
and Khalil kills Ethan at 94.769s. Those intervening events are available to the
model too. The cast does not establish what information NRG acquired or why
Victor moved. The two positions do not establish the route he took.

The new paired-report movement features deliberately do not bridge this
11.806-second gap: they require reports at most two seconds apart. A visual trail
between these positions does not imply that a continuous movement was observed.

This is a useful candidate for owner interpretation. The current model can use
team-level spatial changes and event timing, but it cannot track an individual
player's identity through its aggregated presence channels. If that prevents
distinguishing the desired patterns, player-level trajectory features should be
a separate, explicitly evaluated representation change.
