"""Optional tests: these import and execute installed third-party ML libraries."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from data_processing import FEATURE_NAMES
from model import ValorantRoundCNN


class ModelTests(unittest.TestCase):
    def test_small_architecture_supports_a_single_training_example(self):
        model = ValorantRoundCNN(30).train()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 730)
        self.assertFalse(any(isinstance(module, torch.nn.BatchNorm1d) for module in model.modules()))
        output = model(torch.zeros(1, 30, len(FEATURE_NAMES)))
        self.assertEqual(tuple(output.shape), (1, 2))
        output.sum().backward()

    def test_training_step_and_local_checkpoint_roundtrip(self):
        torch.manual_seed(7)
        model = ValorantRoundCNN(30)
        values = torch.randn(4, 30, len(FEATURE_NAMES))
        targets = torch.tensor([0, 1, 0, 1])
        optimizer = torch.optim.Adam(model.parameters())
        loss = torch.nn.functional.cross_entropy(model(values), targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            expected = model(values)
        self.assertEqual(tuple(expected.shape), (4, 2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            model.save(path)
            restored = ValorantRoundCNN.load(path)
            with torch.no_grad():
                torch.testing.assert_close(restored(values), expected)
            config = json.loads((path / "config.json").read_text())
            config["feature_names"] = list(reversed(FEATURE_NAMES))
            (path / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                ValorantRoundCNN.load(path)
            config["feature_names"] = FEATURE_NAMES
            config["architecture"] = "valorant-round-cnn-v1"
            (path / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                ValorantRoundCNN.load(path)

    def test_invalid_window_and_input_shape(self):
        with self.assertRaises(ValueError):
            ValorantRoundCNN(7)
        with self.assertRaises(ValueError):
            ValorantRoundCNN(30)(torch.zeros(1, 29, len(FEATURE_NAMES)))


if __name__ == "__main__":
    unittest.main()
