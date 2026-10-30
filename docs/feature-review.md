# Feature review before ML execution

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

## Windows to inspect

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
