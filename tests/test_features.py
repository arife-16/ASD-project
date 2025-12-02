import unittest
import numpy as np
from asd_pipeline.features import sliding_windows, correlation_matrix, dynamic_state_features, build_feature_vector_with_states


class TestFeatures(unittest.TestCase):
    def test_sliding_windows(self):
        ts = np.random.randn(10, 100)
        wins = sliding_windows(ts, 20, 10)
        self.assertTrue(len(wins) > 0)

    def test_dynamic_state_features(self):
        ts = np.random.randn(20, 200)
        feats = dynamic_state_features(ts, 40, 20, n_states=4)
        self.assertEqual(feats["state_occ"].shape[0], 4)
        self.assertEqual(feats["transitions"].shape[0], 16)
        self.assertEqual(feats["dwell_mean"].shape[0], 4)
        self.assertEqual(feats["dwell_std"].shape[0], 4)
        self.assertEqual(feats["entropy"].shape[0], 1)
        self.assertEqual(feats["asymmetry"].shape[0], 1)

    def test_build_feature_vector_with_states(self):
        ts = np.random.randn(12, 120)
        feat, idxs = build_feature_vector_with_states(ts, tr=2.0, window_size=30, step=10, n_states=3)
        self.assertTrue(len(feat) > 0)
        self.assertIn("entropy", idxs)


if __name__ == "__main__":
    unittest.main()
