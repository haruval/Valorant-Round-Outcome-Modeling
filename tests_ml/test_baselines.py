"""Optional ML tests: fit local synthetic data and load saved JSON coefficients."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseline_features import temporal_summary
from baseline_models import baseline_probabilities, fit_baselines, load_baselines, save_baselines


class BaselineTests(unittest.TestCase):
    def test_exported_probabilities_match_sklearn_and_retain_training_scaler(self):
        windows = []
        for kill_bin in (2, 5, 10, 14, 20, 27):
            windows.append([[ (i + 1) / 30, 1., 1. - .2 * (i >= kill_bin),
                              .2 * (i >= kill_bin), 0., 0., .4, .4, 0., 0.,
                              float(i >= kill_bin), float(i >= kill_bin)] for i in range(30)])
        labels = [1, 1, 0, 1, 0, 0]
        bundle = fit_baselines(windows, labels, seed=7)
        held_out = copy.deepcopy(windows[:2])
        for row in held_out[0]:
            row[6] = 3.0
        before = copy.deepcopy(bundle)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baselines.json"
            save_baselines(bundle, path)
            restored = load_baselines(path)
            observed = baseline_probabilities(restored, held_out)
        self.assertEqual(bundle, before)
        self.assertEqual(observed["train_prior"], [.5, .5])
        for name, transform in (("logistic_snapshot", lambda window: window[-1]),
                                 ("logistic_temporal", temporal_summary)):
            reference = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000, random_state=7))
            reference.fit([transform(window) for window in windows], labels)
            expected = reference.predict_proba([transform(window) for window in held_out])[:, 1]
            np.testing.assert_allclose(observed[name], expected, rtol=1e-12, atol=1e-12)
