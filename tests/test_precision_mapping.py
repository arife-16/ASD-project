import unittest
import numpy as np
import os
from asd_pipeline.precision_mapping import (
    compute_dense_connectivity, 
    match_communities_to_template, 
    calculate_network_surface_areas,
    precision_mapping_workflow
)

class TestPrecisionMapping(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create small synthetic time series
        # 100 timepoints, 50 vertices
        self.n_time = 100
        self.n_vertices = 50
        self.ts = np.random.randn(self.n_time, self.n_vertices)
        
        # Create fake template labels (3 networks)
        self.template = np.random.randint(1, 4, size=self.n_vertices)
        
    def test_dense_connectivity(self):
        # Keep top 10%
        edges, n_v = compute_dense_connectivity(self.ts, top_k_percent=10.0)
        self.assertEqual(n_v, self.n_vertices)
        
        # Check number of edges
        # N^2 = 2500. Diagonal = 50. Off-diag = 2450.
        # Top 10% ~ 245.
        # Edges should be around 245.
        self.assertTrue(len(edges) > 0)
        
    def test_community_detection_and_matching(self):
        # Fake modules from community detection
        # Assume 5 communities found
        modules = np.random.randint(1, 6, size=self.n_vertices)
        
        # Match to template (3 networks)
        remapped = match_communities_to_template(modules, self.template, n_template_networks=3)
        
        # Output should be in range 0..3
        self.assertTrue(np.all(remapped >= 0))
        self.assertTrue(np.all(remapped <= 3))
        
    def test_surface_areas(self):
        modules = np.array([1, 1, 2, 3, 0])
        areas = calculate_network_surface_areas(modules, n_networks=3)
        # Net 1: 2 nodes
        # Net 2: 1 node
        # Net 3: 1 node
        expected = np.array([2.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(areas, expected)
        
    def test_full_workflow(self):
        # Mock Infomap to avoid dependency in test environment if needed?
        # But we installed it.
        try:
            import infomap
        except ImportError:
            self.skipTest("Infomap not installed")
            
        areas = precision_mapping_workflow(self.ts, self.template, top_k_percent=5.0)
        self.assertEqual(len(areas), 3) # 3 template networks

if __name__ == '__main__':
    unittest.main()
