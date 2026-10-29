"""Optional tests: execute ML libraries, fit baselines, and load local artifacts."""
import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from rolling_features import FEATURE_NAMES
from rolling_model import RollingTemporalCNN
from rolling_experiment import fit_baselines, baseline_probabilities, load_baselines, summary


class RollingModelTests(unittest.TestCase):
    def test_temporal_blocks_are_causal_and_checkpoint_roundtrips(self):
        torch.manual_seed(1)
        model = RollingTemporalCNN().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 3746)
        a = torch.randn(2, 20, len(FEATURE_NAMES))
        b = a.clone()
        b[:, 10:, :] += 5
        with torch.inference_mode():
            before = model.temporal(torch.relu(model.project(a.transpose(1, 2))))
            after = model.temporal(torch.relu(model.project(b.transpose(1, 2))))
            torch.testing.assert_close(before[:, :, :10], after[:, :, :10])
            expected = model(a)
        with tempfile.TemporaryDirectory() as directory:
            model.save(directory)
            restored = RollingTemporalCNN.load(directory)
            with torch.inference_mode():
                torch.testing.assert_close(restored(a), expected)
        model.train()
        loss = torch.nn.functional.cross_entropy(model(a[:1]), torch.tensor([1]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_round_weighted_baselines_export_exact_probabilities(self):
        rng = np.random.default_rng(7)
        windows = rng.normal(size=(8, 20, len(FEATURE_NAMES))).tolist()
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        weights = [.25] * 4 + [.5] * 4
        bundle = fit_baselines(windows, labels, weights, 7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baselines.json"
            path.write_text(json.dumps(bundle))
            probabilities = baseline_probabilities(load_baselines(path), windows)
        self.assertAlmostEqual(probabilities["train_prior"][0], 2 / 3)
        for name, transform in (("snapshot_logistic", lambda w: w[-1]), ("history_logistic", summary)):
            values = np.asarray([transform(window) for window in windows])
            scaler = StandardScaler().fit(values, sample_weight=weights)
            estimator = LogisticRegression(C=1, max_iter=3000, random_state=7)
            estimator.fit(scaler.transform(values), labels, sample_weight=weights)
            np.testing.assert_allclose(probabilities[name], estimator.predict_proba(scaler.transform(values))[:, 1], atol=1e-12)
