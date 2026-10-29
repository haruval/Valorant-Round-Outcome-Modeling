# Data provenance

- File: `events_2629390_grid.jsonl`
- Source: Riot API data supplied during the hackathon, as described by the
  project owner. The provided JSONL uses a GRID-style event envelope and was
  copied unchanged from the original
  [Valorant-ML-Project repository](https://github.com/haruval/Valorant-ML-Project).
- Series ID: `2629390`; event timestamps begin on February 16, 2024 (UTC).
- Observed content: 44 completed rounds, two completed maps (Breeze and Ascent),
  and an initial canceled game with correction events.
- Redistribution: the project owner confirmed permission on October 28, 2026.
  The underlying data remains subject to its applicable provider terms; this
  statement does not grant rights to other datasets.

Run `python3 -S -B data_processing.py data` for SHA-256, label counts, and
cutoff-specific eligibility/exclusion counts. Every training run saves the same
audit in its manifest. No downloader, API key, or account connection is needed.
The checked-in [audit](../docs/data-audit.json) uses a 30-second cutoff and
retains 43 rounds: 17 attacker wins and 26 defender wins.

This single series is a development sample. It does not provide independent
training, validation, and test series. Synthetic cases in `tests/` exercise edge
conditions without depending on this file.

API access ended after the hackathon. No ongoing API access is required: the
audit, training pipeline, and viewer discover additional JSONL exports recursively
under this directory. Keep complete event streams in their original format and
do not put prediction outputs here. See [the pipeline](../docs/pipeline.md).
