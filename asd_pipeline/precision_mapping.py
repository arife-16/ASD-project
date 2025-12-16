import numpy as np
import os
from typing import Tuple, List, Optional, Dict, Union
from nilearn.image import resample_to_img
import nibabel as nib
from scipy import sparse
from .precision_loader import extract_dense_gm_timeseries, compute_sparse_connectivity

try:
    from infomap import Infomap
except ImportError:
    Infomap = None

def run_infomap_community_detection(sparse_graph: sparse.csr_matrix) -> np.ndarray:
    """
    Runs Infomap community detection on the sparse graph.
    
    Args:
        sparse_graph: Sparse adjacency matrix (N_nodes x N_nodes)
        
    Returns:
        modules: Array of shape (n_nodes,) with module ID for each node.
    """
    if Infomap is None:
        raise ImportError("Infomap is not installed. Please install with `pip install infomap`.")
        
    im = Infomap("--two-level --silent")
    
    # Add links from sparse matrix
    # sparse_graph is (N_nodes x N_nodes)
    rows, cols = sparse_graph.nonzero()
    weights = sparse_graph.data
    
    for i in range(len(rows)):
        # Infomap uses 1-based indexing, python uses 0-based
        source = int(rows[i]) + 1
        target = int(cols[i]) + 1
        weight = float(weights[i])
        im.add_link(source, target, weight)
        
    # Run
    im.run()
    
    # Extract results
    # Initialize with 0 (unassigned)
    n_nodes = sparse_graph.shape[0]
    modules = np.zeros(n_nodes, dtype=int)
    
    for node in im.tree:
        if node.is_leaf:
            # node.node_id is 1-based index from add_link
            modules[node.node_id - 1] = node.module_index
            
    return modules


def match_communities_to_template(
    subject_modules: np.ndarray, 
    template_labels: np.ndarray,
    n_template_networks: int = 7
) -> np.ndarray:
    """
    Matches subject-specific communities to template networks based on spatial overlap (Dice).
    """
    # Get unique subject modules
    subj_ids = np.unique(subject_modules)
    subj_ids = subj_ids[subj_ids > 0] # Assume 0 is background
    
    remapped_modules = np.zeros_like(subject_modules)
    
    # For each subject module, find best matching template network
    for sid in subj_ids:
        s_mask = (subject_modules == sid)
        
        best_overlap = 0
        best_tid = 0
        
        for tid in range(1, n_template_networks + 1):
            t_mask = (template_labels == tid)
            
            # Dice coefficient
            intersection = np.logical_and(s_mask, t_mask).sum()
            total = s_mask.sum() + t_mask.sum()
            
            if total > 0:
                dice = 2 * intersection / total
                if dice > best_overlap:
                    best_overlap = dice
                    best_tid = tid
        
        # Assign subject module to best matching template network
        if best_tid > 0:
            remapped_modules[s_mask] = best_tid
            
    return remapped_modules


def calculate_network_surface_areas(
    modules: np.ndarray, 
    n_networks: int, 
    vertex_areas: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculates surface area (or voxel count) for each network.
    """
    areas = np.zeros(n_networks)
    
    for i in range(1, n_networks + 1):
        mask = (modules == i)
        if vertex_areas is not None:
            areas[i-1] = np.sum(vertex_areas[mask])
        else:
            areas[i-1] = np.sum(mask)
            
    return areas


def precision_mapping_workflow(
    timeseries_or_path: Union[str, np.ndarray], 
    template_labels_path: Union[str, np.ndarray],
    top_k_percent: float = 0.1,
    vertex_areas: Optional[np.ndarray] = None,
    mask_img: Optional[str] = None,
    confounds: Optional[np.ndarray] = None,
    use_gpu: bool = False
) -> np.ndarray:
    """
    Full workflow: Dense GM Extraction -> Infomap -> Template Matching -> Surface Area Features.
    
    Args:
        timeseries_or_path: Path to NIfTI or numpy array.
        template_labels_path: Path to template NIfTI or numpy array.
        top_k_percent: Percentile of edges to keep.
        vertex_areas: Optional area per voxel.
        mask_img: Path to GM mask (if None, uses MNI152).
        confounds: Confounds for regression.
        use_gpu: Enable GPU acceleration for connectivity calculation.
    """
    
    # 1. Extract Dense GM Timeseries (and Mask)
    if isinstance(timeseries_or_path, str):
        # Use loader to extract from NIfTI
        print(f"[Precision] Extracting GM timeseries from {os.path.basename(timeseries_or_path)}...", flush=True)
        ts, masker = extract_dense_gm_timeseries(timeseries_or_path, confounds=confounds, mask_img=mask_img)
        
        # We also need to mask the template labels to match the GM voxels!
        if isinstance(template_labels_path, str):
            # Resample template to match masker
            # Note: template is categorical, use nearest neighbor
            # NiftiMasker can handle this but we need to be careful with interpolation
            print("[Precision] Resampling template labels to match functional data...", flush=True)
            from nilearn import image
            
            # Load template image
            tpl_img = nib.load(template_labels_path)
            
            # Resample to match functional data (or mask)
            # masker.mask_img_ is the mask used
            tpl_resampled = image.resample_to_img(
                tpl_img, 
                masker.mask_img_, 
                interpolation='nearest',
                copy_header=True,
                force_resample=True
            )
            
            # Apply mask to get 1D array
            template_labels = masker.transform(tpl_resampled).ravel()
            # Convert back to int if needed
            template_labels = np.round(template_labels).astype(int)
        else:
            template_labels = template_labels_path
            
    else:
        # Pre-extracted array provided
        ts = timeseries_or_path
        template_labels = template_labels_path # Assume matching
        
    # 2. Sparse Connectivity
    print("[Precision] Computing sparse connectivity graph...", flush=True)
    sparse_graph = compute_sparse_connectivity(ts, top_k_percent, use_gpu=use_gpu)
    
    # 3. Community Detection
    print("[Precision] Running Infomap community detection...", flush=True)
    modules = run_infomap_community_detection(sparse_graph)
    
    # 4. Template Matching
    print("[Precision] Matching communities to template...", flush=True)
    n_template_networks = int(np.max(template_labels))
    remapped_modules = match_communities_to_template(modules, template_labels, n_template_networks)
    
    # 5. Feature Extraction
    areas = calculate_network_surface_areas(remapped_modules, n_template_networks, vertex_areas)
    print("[Precision] Feature extraction complete.", flush=True)
    
    return areas


def individual_parcellation(
    nifti_path: str, 
    n_components: int = 20, 
    method: str = "dict_learning",
    smoothing_fwhm: float = 6.0,
    standardize: bool = True,
    memory: str = "nilearn_cache",
    n_jobs: int = 1
) -> Tuple[np.ndarray, nib.nifti1.Nifti1Image]:
    """
    Performs individual parcellation on a subject's NIfTI file to extract
    subject-specific functional networks.
    
    This implements the 'Precision Functional Mapping' concept where networks
    are derived from the individual's data rather than a fixed atlas.
    
    Args:
        nifti_path: Path to the 4D NIfTI file.
        n_components: Number of networks/components to extract.
        method: 'dict_learning' (sparse) or 'canica' (independent).
        
    Returns:
        timeseries: (n_timepoints, n_components)
        maps_img: Nifti1Image of the spatial maps (4D).
    """
    # Load image (lazy loading handled by nilearn)
    # Note: For individual parcellation, we fit on a single subject.
    # Standard CanICA/DictLearning is often used on a group, but here we apply it individually
    # to capture subject-specific topography.
    
    if method == "dict_learning":
        decomposer = DictLearning(
            n_components=n_components,
            smoothing_fwhm=smoothing_fwhm,
            standardize=standardize,
            random_state=42,
            memory=memory,
            n_jobs=n_jobs
        )
    elif method == "canica":
        decomposer = CanICA(
            n_components=n_components,
            smoothing_fwhm=smoothing_fwhm,
            standardize=standardize,
            random_state=42,
            memory=memory,
            n_jobs=n_jobs
        )
    else:
        raise ValueError(f"Unknown method: {method}")
        
    # Fit on the single subject image
    # Note: CanICA expects a list of 4D images (even if len=1)
    decomposer.fit([nifti_path])
    
    maps_img = decomposer.components_img_
    timeseries = decomposer.transform([nifti_path])[0]
    
    return timeseries, maps_img


def calculate_spatial_overlap(
    subject_maps_img: nib.nifti1.Nifti1Image, 
    template_maps_img: nib.nifti1.Nifti1Image, 
    threshold: float = 2.0
) -> Dict[str, float]:
    """
    Calculates overlap metrics (Dice, Jaccard) between subject maps and template maps.
    Useful for quantifying "Language Network Expansion" or shifts.
    
    Args:
        subject_maps_img: 4D image of subject components.
        template_maps_img: 4D image of template components (must match n_components).
        threshold: Z-score threshold for binarizing maps.
        
    Returns:
        Dictionary of overlap metrics averaged across components.
    """
    # Resample subject maps to template space if needed
    if subject_maps_img.shape != template_maps_img.shape:
        subject_maps_img = resample_to_img(subject_maps_img, template_maps_img)
        
    subj_data = subject_maps_img.get_fdata()
    temp_data = template_maps_img.get_fdata()
    
    n_comps = subj_data.shape[-1]
    if temp_data.shape[-1] != n_comps:
        # If component counts differ, we can't do direct 1-to-1 matching easily
        # without Hungarian algorithm or similar matching.
        # For now, return empty or raise warning.
        return {"dice_mean": 0.0, "jaccard_mean": 0.0}
        
    dice_scores = []
    jaccard_scores = []
    
    for i in range(n_comps):
        # Binarize
        s_map = np.abs(subj_data[..., i]) > threshold
        t_map = np.abs(temp_data[..., i]) > threshold
        
        intersection = np.logical_and(s_map, t_map).sum()
        union = np.logical_or(s_map, t_map).sum()
        s_sum = s_map.sum()
        t_sum = t_map.sum()
        
        if s_sum + t_sum > 0:
            dice = 2.0 * intersection / (s_sum + t_sum)
        else:
            dice = 0.0
            
        if union > 0:
            jaccard = intersection / union
        else:
            jaccard = 0.0
            
        dice_scores.append(dice)
        jaccard_scores.append(jaccard)
        
    return {
        "dice_mean": np.mean(dice_scores),
        "dice_std": np.std(dice_scores),
        "jaccard_mean": np.mean(jaccard_scores),
        "jaccard_std": np.std(jaccard_scores)
    }
