import unittest
import numpy as np
from asd_pipeline.model import evaluate_classifier, evaluate_models


class TestModel(unittest.TestCase):
    def test_evaluate_classifier(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(200, 50))
        y = rng.integers(0, 2, size=200)
        metrics = evaluate_classifier(X, y)
        self.assertIn("test_roc_auc", metrics)

    def test_evaluate_models_group(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(200, 30))
        y = rng.integers(0, 2, size=200)
        groups = rng.integers(0, 5, size=200)
        res = evaluate_models(X, y, cv_strategy="group", groups=groups, cv_splits=3)
        self.assertIn("logistic_l2", res)
        self.assertIn("roc_auc", res["logistic_l2"])


if __name__ == "__main__":
    unittest.main()
