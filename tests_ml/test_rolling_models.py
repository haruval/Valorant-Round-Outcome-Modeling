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
from rolling_model import RollingTemporalCNN, PooledHistoryMLP
from rolling_inputs import derive_input_policy, mask_window
from rolling_experiment import fit_baselines, baseline_probabilities, load_baselines, summary


class RollingModelTests(unittest.TestCase):
    def inputs(self):
        values = np.random.default_rng(7).normal(size=(8, 20, len(FEATURE_NAMES)))
        values[:, :, 0] = 0
        return values.tolist()

    def test_temporal_blocks_are_causal_and_checkpoint_roundtrips(self):
        torch.manual_seed(1)
        policy = derive_input_policy(self.inputs())
        model = RollingTemporalCNN(policy).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 8 * len(FEATURE_NAMES) + 826)
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
            self.assertEqual(restored.input_policy, policy)
            with torch.inference_mode():
                torch.testing.assert_close(restored(a), expected)
        model.train()
        loss = torch.nn.functional.cross_entropy(model(a[:1]), torch.tensor([1]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_neural_models_mask_unseen_inputs_and_preserve_policy(self):
        policy = derive_input_policy(self.inputs())
        a = torch.randn(2, 20, len(FEATURE_NAMES))
        b = a.clone()
        b[:, :, 0] = 1_000_000
        for cls in (RollingTemporalCNN, PooledHistoryMLP):
            with self.subTest(model=cls.__name__):
                model = cls(policy).eval()
                with torch.inference_mode():
                    expected = model(a)
                    torch.testing.assert_close(expected, model(b))
                with tempfile.TemporaryDirectory() as directory:
                    model.save(directory)
                    restored = cls.load(directory)
                    self.assertEqual(restored.input_policy, policy)
                    with torch.inference_mode():
                        torch.testing.assert_close(restored(b), expected)

    def test_pooled_control_is_invariant_to_history_order(self):
        model = PooledHistoryMLP(derive_input_policy(self.inputs())).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 8 * len(FEATURE_NAMES) + 890)
        a = torch.randn(2, 20, len(FEATURE_NAMES))
        b = torch.cat((a[:, :19].flip(1), a[:, 19:]), dim=1)
        with torch.inference_mode():
            torch.testing.assert_close(model(a), model(b))
        model.train()
        loss = torch.nn.functional.cross_entropy(model(a), torch.tensor([0, 1]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_round_weighted_baselines_export_exact_probabilities(self):
        windows = self.inputs()
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        weights = [.25] * 4 + [.5] * 4
        bundle = fit_baselines(windows, labels, weights, 7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baselines.json"
            path.write_text(json.dumps(bundle))
            probabilities = baseline_probabilities(load_baselines(path), windows)
        self.assertAlmostEqual(probabilities["train_prior"][0], 2 / 3)
        masked = [mask_window(window, bundle["input_policy"]) for window in windows]
        for name, transform in (("snapshot_logistic", lambda w: w[-1]), ("history_logistic", summary)):
            values = np.asarray([transform(window) for window in masked])
            scaler = StandardScaler().fit(values, sample_weight=weights)
            estimator = LogisticRegression(C=1, max_iter=3000, random_state=7)
            estimator.fit(scaler.transform(values), labels, sample_weight=weights)
            np.testing.assert_allclose(probabilities[name], estimator.predict_proba(scaler.transform(values))[:, 1], atol=1e-12)
        changed = json.loads(json.dumps(windows))
        for window in changed:
            for row in window:
                row[0] = 1_000_000
        self.assertEqual(baseline_probabilities(bundle, changed), probabilities)
