import unittest
import numpy as np
from asd_pipeline.normative import estimate_normative_model, personalized_deviation_maps

class TestNormativeNonLinear(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n_samples = 100
        n_features = 2
        
        # Create synthetic data: Quadratic relationship with Age
        self.age = np.linspace(5, 20, n_samples).reshape(-1, 1)
        # Feature = 10 + 2.0*Age - 0.1*Age^2 + noise
        # Peak at 10. Max val ~ 20. Min val (at 20) ~ 10 + 40 - 40 = 10.
        # Range ~10. Noise 0.5. SNR should be high.
        self.features = 10 + 2.0 * self.age - 0.1 * self.age**2 + np.random.normal(0, 0.5, (n_samples, n_features))
        
        self.td_covars = self.age
        self.all_covars = self.age  # Test on same data for simplicity
        
    def test_gpr_model(self):
        """Test Gaussian Process Regression normative model"""
        pred_mean, pred_std = estimate_normative_model(
            self.features, self.td_covars, self.all_covars, model_type="gpr"
        )
        
        self.assertEqual(pred_mean.shape, self.features.shape)
        self.assertEqual(pred_std.shape, self.features.shape)
        
        # Check if GPR fits reasonably well (R2 > 0)
        resid = self.features - pred_mean
        mse = np.mean(resid**2)
        var = np.var(self.features)
        r2 = 1 - mse/var
        self.assertGreater(r2, 0.5, "GPR should fit quadratic data well")

    def test_lowess_model(self):
        """Test Lowess normative model"""
        try:
            import statsmodels.api as sm
        except ImportError:
            self.skipTest("statsmodels not installed")
            
        pred_mean, pred_std = estimate_normative_model(
            self.features, self.td_covars, self.all_covars, model_type="lowess"
        )
        
        self.assertEqual(pred_mean.shape, self.features.shape)
        self.assertEqual(pred_std.shape, self.features.shape)
        
        # Check fit
        resid = self.features - pred_mean
        mse = np.mean(resid**2)
        var = np.var(self.features)
        r2 = 1 - mse/var
        self.assertGreater(r2, 0.5, "Lowess should fit quadratic data well")

    def test_personalized_deviation_maps_nonlinear(self):
        """Test full deviation map function with GPR"""
        res = personalized_deviation_maps(
            self.features, self.features, covars=self.td_covars, model_type="gpr"
        )
        self.assertIn("feature_z", res)
        self.assertIn("pred_mean", res)
        self.assertEqual(res["feature_z"].shape, self.features.shape)

if __name__ == '__main__':
    unittest.main()
