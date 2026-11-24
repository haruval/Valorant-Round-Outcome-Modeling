# Saved predictions for the visualizer

These files let a fresh clone show saved win probabilities without installing
anything or loading a model:

```bash
python3 -E -S -B serve.py --data-dir data --port 8771 --comparison-config demo/comparison.json
```

Each folder holds two JSON files copied unchanged from one completed validation
run: `run.json` (the run manifest) and `predictions.jsonl` (one row per
prediction window). There are eight runs: full and reduced inputs, temporal CNN
and pooled control, for each validation map, all with seed 42. Seed 42 is an
inspection convention, not a best-seed selection. No model weights are included.

`comparison.json` pins each file by SHA-256, and the server refuses to start if
one differs. `.gitattributes` keeps these files byte-identical on checkout.
Results for all seeds are in [the experiment record](../docs/experiment-results.md).
