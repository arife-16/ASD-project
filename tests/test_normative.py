import unittest
import numpy as np
from asd_pipeline.normative import personalized_deviation_maps


class TestNormative(unittest.TestCase):
    def test_zscores_mahalanobis(self):
        rng = np.random.default_rng(0)
        X_td = rng.normal(size=(50, 100))
        X_all = rng.normal(size=(80, 100))
        dev = personalized_deviation_maps(X_td, X_all)
        self.assertIn("feature_z", dev)
        self.assertIn("mahalanobis", dev)
        self.assertEqual(dev["feature_z"].shape, X_all.shape)
        self.assertEqual(dev["mahalanobis"].shape[0], X_all.shape[0])


if __name__ == "__main__":
    unittest.main()
