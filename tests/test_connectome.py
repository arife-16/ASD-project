import unittest
import numpy as np
from asd_pipeline.connectome import fc_matrix, precision_partial_corr, threshold_adjacency, node_strength, local_clustering, build_connectome_feature_vector


class TestConnectome(unittest.TestCase):
    def test_fc_partial(self):
        ts = np.random.randn(20, 200)
        fc = fc_matrix(ts)
        pc = precision_partial_corr(ts)
        self.assertEqual(fc.shape, pc.shape)

    def test_graph_metrics(self):
        ts = np.random.randn(12, 100)
        fc = fc_matrix(ts)
        adj = threshold_adjacency(fc, thr=0.2)
        strength = node_strength(fc)
        cluster = local_clustering(adj)
        self.assertEqual(len(strength), ts.shape[0])
        self.assertEqual(len(cluster), ts.shape[0])

    def test_connectome_feature_vector(self):
        ts = np.random.randn(10, 120)
        vec, idxs = build_connectome_feature_vector(ts, window_size=30, step=10, thr=0.25)
        self.assertTrue(len(vec) > 0)
        self.assertIn("fc_mean", idxs)


if __name__ == "__main__":
    unittest.main()
