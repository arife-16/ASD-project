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
    Computes dense correlation matrix using a blocked approach to save RAM.
    Only keeps the strongest 'top_k_percent' connections.
    """ 
    n_samples, n_voxels = time_series.shape
    
    # 1. Estimate Threshold from a random subset to avoid computing full matrix
    # Sample 2000 voxels (or fewer if total < 2000)
    sample_size = min(2000, n_voxels)
    indices = np.random.choice(n_voxels, sample_size, replace=False)
    sample_ts = time_series[:, indices]
    
    # Compute correlation of sample against all voxels
    # Shape: (sample_size, n_voxels)
    sample_corr = np.dot(sample_ts.T, time_series) / n_samples
    
    # Compute threshold from this sample
    # We want top k percent
    threshold_val = np.percentile(sample_corr, 100 - top_k_percent)
    del sample_corr # Free memory
    
    # 2. Compute Correlation in Blocks and sparsify immediately
    # Drastically reduce block size to prevent memory spikes
    block_size = 500  # Smaller block size
    sparse_rows = []
    
    # Cast to float32 to save 50% RAM
    time_series = time_series.astype(np.float32)
    
    for start_idx in range(0, n_voxels, block_size):
        end_idx = min(start_idx + block_size, n_voxels)
        
        # Compute block correlation: (block_size, n_voxels)
        # block_ts: (n_samples, block_width)
        block_ts = time_series[:, start_idx:end_idx]
        
        # Use float32 dot product
        # Result shape: (500, 60000) * 4 bytes ≈ 120MB per block (Very safe)
        block_corr = np.dot(block_ts.T, time_series) / n_samples
        
        # Apply threshold
        block_corr[block_corr < threshold_val] = 0
        
        # Convert to sparse CSR immediately
        sparse_block = sparse.csr_matrix(block_corr)
        sparse_rows.append(sparse_block)
        
        # Aggressively free memory
        del block_corr, block_ts
        import gc
        gc.collect()
        
    # 3. Stack all sparse blocks
    sparse_graph = sparse.vstack(sparse_rows)
    
    return sparse_graph
