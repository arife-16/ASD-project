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
    # This automatically resamples the GM mask to match functional data's resolution (e.g., 3mm) 
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

def compute_sparse_connectivity(
    time_series: np.ndarray, 
    top_k_percent: float = 0.1,
    use_gpu: bool = False
) -> sparse.csr_matrix: 
    """ 
    Computes dense correlation matrix using a blocked approach to save RAM.
    Only keeps the strongest 'top_k_percent' connections.
    Supports GPU acceleration via PyTorch.
    """ 
    n_samples, n_voxels = time_series.shape
    
    # Check GPU availability
    if use_gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                print("Warning: GPU requested but not available. Falling back to CPU.", flush=True)
                use_gpu = False
        except ImportError:
            print("Warning: Torch not installed. Falling back to CPU.", flush=True)
            use_gpu = False

    if use_gpu:
        print(f"Computing sparse connectivity on GPU ({n_voxels} voxels)...", flush=True)
        import torch
        
        # Move time_series to GPU
        # Check VRAM limit? 60k * 200 * 4 bytes ~ 48MB. Tiny.
        ts_gpu = torch.tensor(time_series, dtype=torch.float32).cuda()
        
        # 1. Estimate Threshold
        sample_size = min(2000, n_voxels)
        # Random indices
        indices = torch.randperm(n_voxels)[:sample_size]
        sample_ts = ts_gpu[:, indices]
        
        # Correlation: X.T @ X / N
        # (sample, time) @ (time, all) -> (sample, all)
        # sample_ts is (time, sample) -> sample_ts.T is (sample, time)
        sample_corr = torch.matmul(sample_ts.T, ts_gpu) / n_samples
        
        # Calculate percentile
        # Flatten and sort
        flat_corr = sample_corr.flatten()
        k_val = int((1.0 - top_k_percent / 100.0) * flat_corr.numel())
        # Top k values -> we want the value at index k_val in sorted array
        # sort is expensive. kthvalue is faster.
        # we want top_k_percent, so we keep top X. 
        # threshold is the value at (100 - top_k) percentile.
        # e.g. top 10% -> 90th percentile.
        # torch.kthvalue finds the k-th smallest element.
        # index = floor( (100-top_k)/100 * N )
        # Ensure index is valid
        k_idx = max(1, min(flat_corr.numel(), int((1.0 - top_k_percent / 100.0) * flat_corr.numel())))
        threshold_val = torch.kthvalue(flat_corr, k_idx).values.item()
        
        del sample_corr, flat_corr, sample_ts
        torch.cuda.empty_cache()
        
        # 2. Compute in Blocks on GPU
        block_size = 2000 # Can be larger on GPU
        sparse_rows = []
        
        for start_idx in range(0, n_voxels, block_size):
            end_idx = min(start_idx + block_size, n_voxels)
            
            block_ts = ts_gpu[:, start_idx:end_idx]
            block_corr = torch.matmul(block_ts.T, ts_gpu) / n_samples
            
            # Thresholding on GPU
            # Create mask
            mask = block_corr < threshold_val
            block_corr[mask] = 0
            
            # Convert to sparse on CPU
            # To sparse CSR: we need to move to CPU first?
            # block_corr is dense (block, all). 
            # If it's very sparse, we can use to_sparse() but scipy needs cpu numpy.
            
            # Optimization: If highly sparse, we can get indices on GPU?
            # For simplicity, move dense block to CPU and sparsify with scipy (robust)
            # OR use pytorch sparse tensors. But we need to return scipy sparse.
            
            block_numpy = block_corr.cpu().numpy()
            sparse_block = sparse.csr_matrix(block_numpy)
            sparse_rows.append(sparse_block)
            
            del block_corr, block_ts, block_numpy
            torch.cuda.empty_cache()
            
        del ts_gpu
        torch.cuda.empty_cache()
        
        sparse_graph = sparse.vstack(sparse_rows)
        return sparse_graph

    # CPU Implementation
    # 1. Estimate Threshold from a random subset to avoid computing full matrix
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
