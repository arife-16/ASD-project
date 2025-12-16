from typing import Dict, Tuple, Optional, Union, Any
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
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
        # Optimization: For high-dimensional features (e.g., >10k), optimizing kernel parameters 
        # on all features simultaneously is prohibitively slow.
        # We use a heuristic: Optimize kernel on a random subset of features, then fix it for all.
        
        n_features_for_opt = min(2000, n_features)
        
        if n_features > 5000:
            print(f"  Optimizing GPR kernel on subset of {n_features_for_opt} features (Total: {n_features})...", flush=True)
            
            # 1. Sample features
            rng = np.random.RandomState(42)
            idx_subset = rng.choice(n_features, n_features_for_opt, replace=False)
            td_features_subset = td_features[:, idx_subset]
            
            # 2. Fit to optimize hyperparameters
            kernel = ConstantKernel() * RBF() + WhiteKernel()
            gpr_opt = GaussianProcessRegressor(kernel=kernel, normalize_y=True, copy_X_train=False)
            gpr_opt.fit(td_covars_s, td_features_subset)
            
            print(f"  Optimal kernel found: {gpr_opt.kernel_}", flush=True)
            
            # 3. Create new GPR with fixed kernel
            # We set optimizer=None to prevent further optimization
            # Instead of using sklearn's fit (which can be slow/memory intensive for huge Y),
            # we manually compute alpha = K^-1 Y using Cholesky.
            
            print("  Computing Kernel Matrix and Cholesky Decomposition...", flush=True)
            kernel = gpr_opt.kernel_
            K = kernel(td_covars_s) # (N, N)
            
            # Ensure positive definiteness (WhiteKernel handles diagonal, but safety check)
            K[np.diag_indices_from(K)] += 1e-10 
            
            try:
                L = cholesky(K, lower=True)
            except np.linalg.LinAlgError:
                print("  Warning: Kernel matrix not positive definite, adding jitter...", flush=True)
                K[np.diag_indices_from(K)] += 1e-6
                L = cholesky(K, lower=True)
            
            print("  Solving for weights (alpha) in blocks to save RAM...", flush=True)
            # alpha = L^-T L^-1 Y
            # If Y is (476, 308546), it takes ~1.2GB RAM (float64).
            # But intermediate operations might copy it.
            # Let's solve in chunks of features (columns of Y)
            
            n_samples, n_features_all = td_features.shape
            alpha = np.zeros_like(td_features)
            
            chunk_size = 50000 # Process 50k features at a time
            
            for start_col in range(0, n_features_all, chunk_size):
                end_col = min(start_col + chunk_size, n_features_all)
                # print(f"    Processing features {start_col}-{end_col}", flush=True)
                y_chunk = td_features[:, start_col:end_col]
                alpha[:, start_col:end_col] = cho_solve((L, True), y_chunk)
            
            # Save components
            # Note: We cannot save the full alpha if it's too big for pickle? 
            # 476 * 300k * 8 bytes = 1.2GB. Pickle should handle it (up to 4GB).
            # But let's check.
            
            model_data.update({
                "alpha": alpha,
                "L": L,
                "X_train": td_covars_s,
                "kernel": kernel
            })
            
            # Skip gpr.fit()
            return model_data
            
        else:
            # Standard fit for smaller dimensions
            kernel = ConstantKernel() * RBF() + WhiteKernel()
            gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, copy_X_train=False)
            gpr.fit(td_covars_s, td_features)
            model_data["gpr_object"] = gpr
            return model_data
            
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
        # Use protocol 4 or 5 for large objects (>4GB support)
        with open(output_path, "wb") as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif output_path.endswith(".npz"):
        # Filter out non-array objects
        arrays = {k: v for k, v in model_data.items() if isinstance(v, np.ndarray)}
        np.savez_compressed(output_path, **arrays) # Compressed to save space
    else:
        # Default pickle
        with open(output_path + ".pkl", "wb") as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)

def fit_and_save_normative_model(
    td_features: np.ndarray, 
    td_covars: np.ndarray, 
    output_path: str,
    model_type: str = "linear",
    use_gpu: bool = False
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
        model_data.update({
            "coef": reg.coef_,
            "intercept": reg.intercept_,
            "resid_std": residuals.std(axis=0)
        })
        return model_data

    elif model_type == "gpr":
        # Gaussian Process Regression
        # Optimization: For high-dimensional features (e.g., >10k), optimizing kernel parameters 
        # on all features simultaneously is prohibitively slow.
        # We use a heuristic: Optimize kernel on a random subset of features, then fix it for all.
        
        n_features = td_features.shape[1]
        n_features_for_opt = min(2000, n_features)
        
        # 1. Optimize Hyperparameters (Always on CPU via Sklearn for robustness)
        print(f"  Optimizing GPR kernel on subset of {n_features_for_opt} features (Total: {n_features})...", flush=True)
        
        rng = np.random.RandomState(42)
        idx_subset = rng.choice(n_features, n_features_for_opt, replace=False)
        td_features_subset = td_features[:, idx_subset]
        
        kernel = ConstantKernel() * RBF() + WhiteKernel()
        gpr_opt = GaussianProcessRegressor(kernel=kernel, normalize_y=True, copy_X_train=False)
        gpr_opt.fit(td_covars_s, td_features_subset)
        
        print(f"  Optimal kernel found: {gpr_opt.kernel_}", flush=True)
        
        # 2. Fit on Full Data (CPU Block-wise or GPU)
        if use_gpu:
            try:
                import torch
                if not torch.cuda.is_available():
                    print("  Warning: GPU requested but not available. Falling back to CPU.", flush=True)
                    use_gpu = False
            except ImportError:
                print("  Warning: Torch not installed. Falling back to CPU.", flush=True)
                use_gpu = False

        if use_gpu:
            print("  Using GPU for GPR fitting...", flush=True)
            import torch
            
            # Extract kernel params from sklearn
            # k1 = Constant * RBF, k2 = White
            # k1.k1 = Constant, k1.k2 = RBF
            k_combined = gpr_opt.kernel_
            k_expr = k_combined.k1
            k_white = k_combined.k2
            
            constant_val = k_expr.k1.constant_value
            length_scale = k_expr.k2.length_scale
            noise_level = k_white.noise_level
            
            # Prepare Data on GPU
            X_torch = torch.tensor(td_covars_s, dtype=torch.float32).cuda()
            Y_torch = torch.tensor(td_features, dtype=torch.float32) # Keep Y on CPU initially if huge
            
            # Compute Kernel Matrix K on GPU
            # RBF: exp(-0.5 * dist^2 / l^2)
            # dist^2 = x^2 + y^2 - 2xy
            # Pytorch cdist computes euclidean distance (sqrt(dist^2))
            
            # Manual RBF
            dists = torch.cdist(X_torch, X_torch, p=2) # (N, N)
            K = constant_val * torch.exp(-0.5 * (dists ** 2) / (length_scale ** 2))
            
            # Add noise + jitter
            K += torch.eye(K.shape[0]).cuda() * (noise_level + 1e-6)
            
            # Cholesky
            print("  Computing Cholesky on GPU...", flush=True)
            L = torch.linalg.cholesky(K)
            
            # Solve alpha = K^-1 Y = L^-T L^-1 Y
            # We solve in blocks if Y is huge, even on GPU
            print("  Solving weights on GPU...", flush=True)
            
            # Move Y to GPU in chunks if it doesn't fit
            # 1.2GB fits in VRAM easily.
            try:
                Y_gpu = Y_torch.cuda()
                alpha_gpu = torch.cholesky_solve(Y_gpu, L)
                alpha = alpha_gpu.cpu().numpy()
                del Y_gpu, alpha_gpu
            except RuntimeError as e: # OOM
                print(f"  GPU OOM ({e}), switching to block-wise GPU solve...", flush=True)
                alpha = np.zeros_like(td_features)
                chunk_size = 20000
                for start in range(0, n_features, chunk_size):
                    end = min(start + chunk_size, n_features)
                    y_chunk = Y_torch[:, start:end].cuda()
                    res = torch.cholesky_solve(y_chunk, L)
                    alpha[:, start:end] = res.cpu().numpy()
                    del res, y_chunk
                    torch.cuda.empty_cache()

            # Save
            model_data.update({
                "alpha": alpha,
                "L": L.cpu().numpy(),
                "X_train": td_covars_s,
                "kernel": gpr_opt.kernel_ # Save the sklearn kernel object for reference
            })
            
            del X_torch, K, L
            torch.cuda.empty_cache()
            return model_data
            
        else:
            # CPU Implementation (Block-wise)
            print("  Computing Kernel Matrix and Cholesky Decomposition (CPU)...", flush=True)
            kernel = gpr_opt.kernel_
            K = kernel(td_covars_s)
            K[np.diag_indices_from(K)] += 1e-10 
            
            try:
                L = cholesky(K, lower=True)
            except np.linalg.LinAlgError:
                print("  Warning: Kernel matrix not positive definite, adding jitter...", flush=True)
                K[np.diag_indices_from(K)] += 1e-6
                L = cholesky(K, lower=True)
            
            print("  Solving for weights (alpha) in blocks to save RAM...", flush=True)
            n_samples, n_features_all = td_features.shape
            alpha = np.zeros_like(td_features)
            chunk_size = 50000
            
            for start_col in range(0, n_features_all, chunk_size):
                end_col = min(start_col + chunk_size, n_features_all)
                y_chunk = td_features[:, start_col:end_col]
                alpha[:, start_col:end_col] = cho_solve((L, True), y_chunk)
            
            model_data.update({
                "alpha": alpha,
                "L": L,
                "X_train": td_covars_s,
                "kernel": kernel
            })
            return model_data

    elif model_type == "lowess":
        # Save training data for lazy evaluation
        model_data["x_train"] = td_covars  # Save raw covariates
        model_data["y_train"] = td_features
        return model_data
        
    return model_data


def load_normative_model(model_path: str) -> Dict[str, Any]:
    """Loads a saved normative model."""
    if model_path.endswith(".npz"):
        return dict(np.load(model_path, allow_pickle=True))
    else:
        with open(model_path, "rb") as f:
            return pickle.load(f)


def predict_normative_model(
    model_data: Dict[str, Any],
    covars: np.ndarray,
    features: Optional[np.ndarray] = None,
    use_gpu: bool = False
) -> Dict[str, np.ndarray]:
    """
    Predicts mean (and std) for new subjects.
    If features are provided, also calculates Z-scores.
    """
    model_type = model_data.get("model_type", "linear")
    
    # Standardize covariates
    scaler_mean = model_data["scaler_mean"]
    scaler_scale = model_data["scaler_scale"]
    covars_s = (covars - scaler_mean) / scaler_scale
    
    n_samples = covars.shape[0]
    # We don't know n_features yet unless features is passed or we look at alpha/coef
    if "alpha" in model_data:
        n_features = model_data["alpha"].shape[1]
    elif "coef" in model_data:
        n_features = model_data["coef"].shape[0] # Transposed
    elif features is not None:
        n_features = features.shape[1]
    else:
        n_features = 0

    pred_mean = None
    pred_std = None
    
    if model_type == "linear":
        coef = model_data["coef"] # (n_features, n_covars)
        intercept = model_data["intercept"] # (n_features,)
        resid_std = model_data["resid_std"] # (n_features,)
        
        # Pred = X @ coef.T + intercept
        pred_mean = covars_s @ coef.T + intercept
        pred_std = np.tile(resid_std, (n_samples, 1))
        
    elif model_type == "gpr":
        alpha = model_data["alpha"] # (n_train, n_features)
        X_train = model_data["X_train"] # (n_train, n_covars)
        kernel = model_data["kernel"] # sklearn kernel object
        
        # Check GPU
        if use_gpu:
            try:
                import torch
                if not torch.cuda.is_available():
                    use_gpu = False
            except ImportError:
                use_gpu = False
                
        if use_gpu:
            print("  Predicting GPR on GPU...", flush=True)
            import torch
            
            # Kernel params
            k_combined = kernel
            k_expr = k_combined.k1
            k_white = k_combined.k2
            constant_val = k_expr.k1.constant_value
            length_scale = k_expr.k2.length_scale
            noise_level = k_white.noise_level
            
            X_test_torch = torch.tensor(covars_s, dtype=torch.float32).cuda()
            X_train_torch = torch.tensor(X_train, dtype=torch.float32).cuda()
            
            # 1. Compute Kernel(X_test, X_train)
            dists = torch.cdist(X_test_torch, X_train_torch, p=2)
            K_trans = constant_val * torch.exp(-0.5 * (dists ** 2) / (length_scale ** 2))
            
            # 2. Predict Mean = K_trans @ alpha
            alpha_torch = torch.tensor(alpha, dtype=torch.float32) # Keep on CPU first
            
            pred_mean = np.zeros((n_samples, n_features), dtype=np.float32)
            
            chunk_size = 20000
            for start in range(0, n_features, chunk_size):
                end = min(start + chunk_size, n_features)
                alpha_chunk = alpha_torch[:, start:end].cuda()
                res = torch.matmul(K_trans, alpha_chunk)
                pred_mean[:, start:end] = res.cpu().numpy()
                del res, alpha_chunk
            
            # 3. Predict Variance
            L = model_data["L"]
            L_torch = torch.tensor(L, dtype=torch.float32).cuda()
            
            # v_term = solve_triangular(L, K_trans.T)
            v_term = torch.linalg.solve_triangular(L_torch, K_trans.T, upper=False)
            
            v_term_sq = torch.sum(v_term ** 2, dim=0) # (n_test,)
            
            pred_var = (constant_val + noise_level) - v_term_sq
            pred_std_val = torch.sqrt(torch.clamp(pred_var, min=1e-8)).cpu().numpy()
            
            pred_std = np.tile(pred_std_val[:, np.newaxis], (1, n_features))
            
            del X_test_torch, X_train_torch, K_trans, L_torch, v_term
            torch.cuda.empty_cache()
            
        else:
            # CPU implementation
            print("  Predicting GPR on CPU...", flush=True)
            kernel = model_data["kernel"]
            K_trans = kernel(covars_s, X_train) # (n_test, n_train)
            
            # Mean
            pred_mean = np.zeros((n_samples, n_features))
            chunk_size = 50000
            for start in range(0, n_features, chunk_size):
                end = min(start + chunk_size, n_features)
                pred_mean[:, start:end] = K_trans @ alpha[:, start:end]
                
            # Variance
            L = model_data["L"]
            v_term = solve_triangular(L, K_trans.T, lower=True)
            k_diag = kernel.diag(covars_s) # (n_test,)
            
            # Manual noise addition if needed, but kernel.diag usually handles signal
            # We need to add noise_level manually to match K_train diagonal
            noise_level = kernel.k2.noise_level
            
            pred_var = k_diag + noise_level - np.sum(v_term**2, axis=0)
            pred_std_val = np.sqrt(np.maximum(pred_var, 1e-8))
            
            pred_std = np.tile(pred_std_val[:, np.newaxis], (1, n_features))

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    result = {
        "pred_mean": pred_mean,
        "pred_std": pred_std
    }
    
    if features is not None:
        # Z-score
        pred_std_safe = np.where(pred_std < 1e-8, 1e-8, pred_std)
        z = (features - pred_mean) / pred_std_safe
        result["feature_z"] = z
        
    return result
