"""Optional tests: execute ML libraries and fit baselines. Run only in the approved environment."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from rolling_experiment import baseline_probabilities, fit_baselines
from rolling_features import FEATURE_NAMES
from rolling_inputs import derive_input_policy
from rolling_model import MODEL_CLASSES as FROZEN
from structure_inputs import (EXTENSION_NAMES, baseline_rows, derive_extension_policy,
                              extended_baseline_probabilities, fit_extended_baselines, load_extended_baselines)
from structure_model import MODEL_CLASSES as EXTENDED

LABELS = [0, 0, 0, 0, 1, 1, 1, 1]
WEIGHTS = [.25] * 4 + [.5] * 4


class StructureModelTests(unittest.TestCase):
    def inputs(self):
        rng = np.random.default_rng(7)
        values = rng.normal(size=(8, 20, len(FEATURE_NAMES)))
        values[:, :, 0] = 0
        extra = rng.uniform(size=(8, 20, len(EXTENSION_NAMES)))
        return values.tolist(), extra.tolist()

    def tensors(self):
        values, extra = self.inputs()
        return (values, extra, torch.tensor(values, dtype=torch.float32), torch.tensor(extra, dtype=torch.float32))

    def test_base_arm_equals_the_frozen_model_through_training(self):
        values, extra, x, e = self.tensors()
        policy, base = derive_input_policy(values), derive_extension_policy(extra, "base")
        y = torch.tensor(LABELS)
        for name in FROZEN:
            with self.subTest(model=name):
                torch.manual_seed(3)
                frozen = FROZEN[name](policy)
                stream_after_frozen = torch.rand(4)
                torch.manual_seed(3)
                extended = EXTENDED[name](policy, base, init_seed=3)
                stream_after_extended = torch.rand(4)
                # Building the extension must not draw from the global stream.
                self.assertTrue(torch.equal(stream_after_frozen, stream_after_extended))
                self.assertEqual(sum(p.numel() for p in extended.parameters()),
                                 sum(p.numel() for p in frozen.parameters()) + 8 * len(EXTENSION_NAMES))
                initial = extended.extension.detach().clone()
                steps = ((frozen, torch.optim.Adam(frozen.parameters(), lr=.001), lambda m: m(x)),
                         (extended, torch.optim.Adam(extended.parameters(), lr=.001), lambda m: m(x, e)))
                for step in range(5):
                    losses = []
                    for model, optimizer, forward in steps:
                        torch.manual_seed(100 + step)  # Same dropout draws for both models.
                        model.train()
                        optimizer.zero_grad(set_to_none=True)
                        loss = torch.nn.functional.cross_entropy(forward(model), y)
                        loss.backward()
                        optimizer.step()
                        losses.append(loss.item())
                    self.assertEqual(losses[0], losses[1])
                self.assertTrue(torch.equal(extended.extension, initial))
                for key, value in frozen.state_dict().items():
                    self.assertTrue(torch.equal(value, extended.state_dict()[key]), key)
                frozen.eval(), extended.eval()
                with torch.inference_mode():
                    self.assertTrue(torch.equal(frozen(x), extended(x, e)))

    def test_only_kept_extension_channels_reach_the_models(self):
        values, extra, x, e = self.tensors()
        policy, structure = derive_input_policy(values), derive_extension_policy(extra, "structure")
        masked, kept = EXTENSION_NAMES.index("atk_support_casts_live"), EXTENSION_NAMES.index("atk_combo_casters")
        for name, cls in EXTENDED.items():
            with self.subTest(model=name):
                model = cls(policy, structure, init_seed=1).eval()
                unseen, seen = e.clone(), e.clone()
                unseen[:, :, masked] += 1_000_000
                seen[:, :, kept] += 1
                with torch.inference_mode():
                    expected = model(x, e)
                    torch.testing.assert_close(model(x, unseen), expected)
                    self.assertFalse(torch.allclose(model(x, seen), expected))
                model.train()
                loss = torch.nn.functional.cross_entropy(model(x, e), torch.tensor(LABELS))
                loss.backward()
                gradient = model.extension.grad[:, :, 0]
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertTrue((gradient[:, masked] == 0).all())
                self.assertTrue((gradient[:, kept] != 0).any())
                with self.assertRaises(ValueError):
                    model(x, e[:, :, :-1])

    def test_checkpoints_roundtrip_and_the_frozen_loader_rejects_them(self):
        values, extra, x, e = self.tensors()
        policy, structure = derive_input_policy(values), derive_extension_policy(extra, "structure")
        for name, cls in EXTENDED.items():
            with self.subTest(model=name), tempfile.TemporaryDirectory() as directory:
                model = cls(policy, structure, init_seed=5).eval()
                with torch.inference_mode():
                    expected = model(x, e)
                model.save(directory)
                restored = cls.load(directory)
                self.assertEqual((restored.input_policy, restored.extension_policy), (policy, structure))
                with torch.inference_mode():
                    torch.testing.assert_close(restored(x, e), expected)
                with self.assertRaises(ValueError):
                    FROZEN[name].load(directory)
                other = next(c for n, c in EXTENDED.items() if n != name)
                with self.assertRaises(ValueError):
                    other.load(directory)

    def test_pooled_control_stays_invariant_to_history_order(self):
        values, extra, x, e = self.tensors()
        model = EXTENDED["pooled_mlp"](derive_input_policy(values), derive_extension_policy(extra, "structure"),
                                       init_seed=2).eval()
        flip = lambda t: torch.cat((t[:, :19].flip(1), t[:, 19:]), dim=1)
        with torch.inference_mode():
            torch.testing.assert_close(model(x, e), model(flip(x), flip(e)))

    def test_temporal_blocks_stay_causal_in_the_extension_input(self):
        values, extra, x, e = self.tensors()
        model = EXTENDED["temporal_cnn"](derive_input_policy(values), derive_extension_policy(extra, "structure"),
                                         init_seed=2).eval()
        later = e.clone()
        later[:, 10:, :] += 5
        with torch.inference_mode():
            before = model.temporal(model.projected(x, e))
            after = model.temporal(model.projected(x, later))
        torch.testing.assert_close(before[:, :, :10], after[:, :, :10])

    def test_base_arm_baselines_equal_the_frozen_baselines(self):
        values, extra = self.inputs()
        policy, base = derive_input_policy(values), derive_extension_policy(extra, "base")
        frozen = baseline_probabilities(fit_baselines(values, LABELS, WEIGHTS, 7, policy), values)
        bundle = fit_extended_baselines(values, extra, LABELS, WEIGHTS, 7, policy, base)
        self.assertEqual(extended_baseline_probabilities(bundle, values, extra), frozen)

    def test_structure_baselines_export_exact_probabilities(self):
        values, extra = self.inputs()
        policy, structure = derive_input_policy(values), derive_extension_policy(extra, "structure")
        bundle = fit_extended_baselines(values, extra, LABELS, WEIGHTS, 7, policy, structure)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baselines.json"
            path.write_text(json.dumps(bundle))
            probabilities = extended_baseline_probabilities(load_extended_baselines(path), values, extra)
        self.assertAlmostEqual(probabilities["train_prior"][0], 2 / 3)
        rows = [baseline_rows(w, x, bundle) for w, x in zip(values, extra)]
        for name in ("snapshot_logistic", "history_logistic"):
            matrix = np.asarray([row[name] for row in rows])
            self.assertEqual(matrix.shape[1], (len(FEATURE_NAMES) + 6) * (2 if name == "history_logistic" else 1))
            scaler = StandardScaler().fit(matrix, sample_weight=WEIGHTS)
            estimator = LogisticRegression(C=1, max_iter=3000, random_state=7)
            estimator.fit(scaler.transform(matrix), LABELS, sample_weight=WEIGHTS)
            np.testing.assert_allclose(probabilities[name], estimator.predict_proba(scaler.transform(matrix))[:, 1],
                                       atol=1e-12)


if __name__ == "__main__":
    unittest.main()
