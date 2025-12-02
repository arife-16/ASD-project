import unittest
import numpy as np
from asd_pipeline.model import site_stratified_kfold


class TestCV(unittest.TestCase):
    def test_site_stratified(self):
        rng = np.random.default_rng(0)
        y = np.array([0,1]*50)
        groups = np.array([0]*50 + [1]*50)
        splits = site_stratified_kfold(groups, y, n_splits=5)
        for train, test in splits:
            gtest = groups[test]
            self.assertTrue(len(np.unique(gtest)) >= 2)
            ytest = y[test]
            self.assertTrue(0 in ytest and 1 in ytest)


if __name__ == "__main__":
    unittest.main()
