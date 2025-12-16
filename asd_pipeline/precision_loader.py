import numpy as np
from nilearn import datasets, input_data, masking
from scipy import sparse
import nibabel as nib
from typing import Tuple, Union, Optional

def extract_dense_gm_timeseries(
    func_file: Union[str, nib.nifti1.Nifti1Image], 
    confounds: Optional[Union[str, np.ndarray]] = None,
    mask_img: Optional[Union[str, nib.nifti1.Nifti1Image]] = None
) -> Tuple[np.ndarray, input_data.NiftiMasker]: 
    """ 
    Extracts time-series only from Gray Matter voxels, reducing dimensionality. 
    """ 
    # 1. Fetch Standard Gray Matter Mask (MNI152) if not provided
    if mask_img is None:
        # threshold=0.2 means voxels with <20% probability of being GM are discarded 
        gm_mask = datasets.fetch_icbm152_brain_gm_mask(threshold=0.2)
    else:
        gm_mask = mask_img

    # 2. Initialize Masker 
    # This automatically resamples the GM mask to match your functional data's resolution (e.g., 3mm) 
    masker = input_data.NiftiMasker( 
        mask_img=gm_mask, 
        standardize=True,      # Important for correlation 
        detrend=True, 
        memory='nilearn_cache', 
        verbose=1 
    ) 

    # 3. Fit and Transform 
    # Output shape: (n_timepoints, n_gm_voxels) -> e.g., (200, ~60000) 
    time_series = masker.fit_transform(func_file, confounds=confounds) 
    
    return time_series, masker 

def compute_sparse_connectivity(time_series: np.ndarray, top_k_percent: float = 0.1) -> sparse.csr_matrix: 
    """ 
    Computes dense correlation matrix but returns a SPARSE matrix to save RAM. 
    Only keeps the strongest 'top_k_percent' connections (crucial for Infomap). 
    """ 
    # 1. Compute full correlation (fast with numpy dot product) 
    # Correlation = (X.T @ X) / n_samples (since data is standardized) 
    n_samples = time_series.shape[0] 
    
    # Note: For very large N (e.g. > 60k), this dot product (60k x 60k) might still be heavy (28GB float64).
    # If memory is an issue, we can compute blocks.
    # Assuming 60k floats is ~3.6GB for the matrix, it fits in Colab RAM (12GB+).
    corr_matrix = np.dot(time_series.T, time_series) / n_samples 
    
    # 2. Thresholding 
    # We only want the top x% strongest correlations for the graph 
    threshold_val = np.percentile(corr_matrix, 100 - top_k_percent) 
    
    # 3. Create Sparse Graph 
    # Zero out weak connections 
    corr_matrix[corr_matrix < threshold_val] = 0 
    
    # Convert to sparse format (efficient for Infomap) 
    sparse_graph = sparse.csr_matrix(corr_matrix) 
    
    return sparse_graph
