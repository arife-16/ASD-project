from typing import Dict, Tuple, Optional, Union
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler


def fit_mvn(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = x.mean(axis=0)
    lw = LedoitWolf().fit(x)
    c = lw.covariance_
    return m, c


def mahalanobis(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    inv = np.linalg.pinv(cov)
    d = x - mean
    return np.sqrt(np.einsum("...i,ij,...j->...", d, inv, d))


def zscores(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    s = np.where(std < 1e-8, 1e-8, std)
    return (x - mean) / s


def estimate_normative_model(
    td_features: np.ndarray, 
    td_covars: np.ndarray, 
    all_covars: np.ndarray, 
    model_type: str = "linear"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimates normative distribution (mean and std) for each subject in all_covars
    based on the relationship learned from td_features and td_covars.
    """
    n_samples_all = all_covars.shape[0]
    n_features = td_features.shape[1]
    
    pred_mean = np.zeros((n_samples_all, n_features))
    pred_std = np.zeros((n_samples_all, n_features))
    
    # Standardize covariates for stability
    scaler = StandardScaler()
    td_covars_s = scaler.fit_transform(td_covars)
    all_covars_s = scaler.transform(all_covars)

    if model_type == "linear":
        reg = LinearRegression()
        reg.fit(td_covars_s, td_features)
        pred_mean = reg.predict(all_covars_s)
        
        # Homoscedastic noise assumption: std is constant (residual std of TD)
        residuals = td_features - reg.predict(td_covars_s)
        std_td = residuals.std(axis=0)
        pred_std[:] = std_td
        
    elif model_type == "gpr":
        # Gaussian Process Regression
        # We fit one GPR per feature to allow feature-specific hyperparameters
        # Optimization: For very high dim features, this is slow. 
        # But normative modeling is often feature-wise.
        # To speed up, we can use a shared kernel if features are similar, 
        # but standard normative modeling usually fits independent models per voxel/ROI.
        
        # Check dimensionality. If > 500 features, warn or use a batched approach?
        # For now, we'll implement a loop.
        
        kernel = ConstantKernel() * RBF() + WhiteKernel()
        
        # If too many features, maybe we should fallback or use a shared kernel approximation?
        # Let's try fitting independent GPRs. 
        # Note: This can be VERY slow for 400x400 FC.
        # Optimization: Fit on PCA components of features? No, we need per-feature norms.
        # Fallback: Use sklearn's multi-output GPR (shares kernel params).
        
        gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, copy_X_train=False)
        gpr.fit(td_covars_s, td_features)
        # Predict
        pm, ps = gpr.predict(all_covars_s, return_std=True)
        pred_mean = pm
        # ps shape depends on GPR implementation for multi-output
        # Sklearn GPR with multi-target returns std as (n_samples,) if kernel is shared.
        if ps.ndim == 1:
            pred_std = np.tile(ps[:, np.newaxis], (1, n_features))
        else:
            pred_std = ps

    elif model_type == "lowess":
        # Use statsmodels lowess
        import statsmodels.api as sm
        # Lowess is 1D covariate typically (Age). If multiple covariates, use GPR.
        # If we have >1 covariate, we'll just use the first one (assume Age) or warn.
        # Here we assume the first column is Age.
        
        x_td = td_covars[:, 0]
        x_all = all_covars[:, 0]
        
        # Sort for lowess interpolation
        sort_idx = np.argsort(x_td)
        x_td_sorted = x_td[sort_idx]
        
        for i in range(n_features):
            y_td = td_features[:, i]
            y_td_sorted = y_td[sort_idx]
            
            # Fit lowess
            # frac=0.6 is typical for developmental curves
            est = sm.nonparametric.lowess(y_td_sorted, x_td_sorted, frac=0.6)
            # est is (n, 2) array of (x, y)
            
            # Interpolate to all_covars
            pred_mean[:, i] = np.interp(x_all, est[:, 0], est[:, 1])
            
            # Estimate std (rolling std of residuals or constant)
            # Simple approach: constant residual std
            resid = y_td - np.interp(x_td, est[:, 0], est[:, 1])
            pred_std[:, i] = resid.std()

    else:
        raise ValueError(f"Unknown normative model type: {model_type}")
        
    return pred_mean, pred_std


def personalized_deviation_maps(
    td_features: np.ndarray, 
    all_features: np.ndarray, 
    covars: Optional[np.ndarray] = None,
    model_type: str = "linear"
) -> Dict[str, np.ndarray]:
    
    if covars is not None:
        # Split covariates
        td_covars = covars[: td_features.shape[0]]
        all_covars = covars
        
        # Estimate normative model
        pred_mean, pred_std = estimate_normative_model(td_features, td_covars, all_covars, model_type=model_type)
        
        # Calculate Z-scores (deviations)
        # Avoid division by zero
        pred_std = np.where(pred_std < 1e-8, 1e-8, pred_std)
        z = (all_features - pred_mean) / pred_std
        
        # For Mahalanobis, we can use the residuals
        # Residuals of the normative model
        residuals = all_features - pred_mean
        
        # We need a covariance matrix of the residuals to compute Mahalanobis
        # We can fit it on the TD residuals
        td_pred_mean = pred_mean[: td_features.shape[0]]
        td_residuals = td_features - td_pred_mean
        
        _, mvn_cov = fit_mvn(td_residuals)
        
        # Mahalanobis distance of the residuals
        m = mahalanobis(residuals, np.zeros_like(residuals[0]), mvn_cov)
        
        return {"feature_z": z, "mahalanobis": m, "pred_mean": pred_mean, "pred_std": pred_std}
        
    else:
        # No covariates: simple z-score against TD group
        mean = td_features.mean(axis=0)
        std = td_features.std(axis=0, ddof=1)
        mvn_mean, mvn_cov = fit_mvn(td_features)
        z = zscores(all_features, mean, std)
        m = mahalanobis(all_features, mvn_mean, mvn_cov)
        return {"feature_z": z, "mahalanobis": m}


import pickle

def save_normative_model(
    model_data: Dict[str, Union[np.ndarray, object]], 
    output_path: str
):
    """
    Saves the normative model parameters to a file.
    
    Args:
        model_data: Dictionary containing model parameters/objects.
                    For Linear: 'coef', 'intercept', 'resid_std', 'scaler_mean', 'scaler_scale'
                    For GPR: 'gpr_object', 'scaler_mean', 'scaler_scale'
                    For Lowess: 'x_train', 'y_train', 'resid_std'
        output_path: Path to save (.pkl or .npz).
    """
    if output_path.endswith(".pkl"):
        with open(output_path, "wb") as f:
            pickle.dump(model_data, f)
    elif output_path.endswith(".npz"):
        # Filter out non-array objects
        arrays = {k: v for k, v in model_data.items() if isinstance(v, np.ndarray)}
        np.savez(output_path, **arrays)
    else:
        # Default pickle
        with open(output_path + ".pkl", "wb") as f:
            pickle.dump(model_data, f)

def fit_and_save_normative_model(
    td_features: np.ndarray, 
    td_covars: np.ndarray, 
    output_path: str,
    model_type: str = "linear"
):
    """
    Fits the model on TD data and saves it.
    """
    # Standardize covariates
    scaler = StandardScaler()
    td_covars_s = scaler.fit_transform(td_covars)
    
    model_data = {
        "model_type": model_type,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_
    }
    
    if model_type == "linear":
        reg = LinearRegression()
        reg.fit(td_covars_s, td_features)
        
        residuals = td_features - reg.predict(td_covars_s)
        std_td = residuals.std(axis=0)
        
        model_data["coef"] = reg.coef_
        model_data["intercept"] = reg.intercept_
        model_data["resid_std"] = std_td
        
    elif model_type == "gpr":
        kernel = ConstantKernel() * RBF() + WhiteKernel()
        gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, copy_X_train=False)
        gpr.fit(td_covars_s, td_features)
        model_data["gpr_object"] = gpr
        
    elif model_type == "lowess":
        # Save training data for lazy evaluation
        model_data["x_train"] = td_covars  # Save raw covariates
        model_data["y_train"] = td_features
        # Calculate residuals for std estimation (approximate via linear or smooth?)
        # For simplicity in saving, we might just save data.
        # Or pre-calculate std?
        # Let's just save data.
        pass
        
    save_normative_model(model_data, output_path)
    return model_data
