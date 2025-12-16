import numpy as np
import os
from typing import Tuple, List, Optional, Dict
from nilearn.decomposition import DictLearning, CanICA
from nilearn.image import load_img, resample_to_img, math_img
from nilearn.masking import apply_mask
import nibabel as nib
try:
    from infomap import Infomap
except ImportError:
    Infomap = None

def compute_dense_connectivity(timeseries: np.ndarray, top_k_percent: float = 0.1) -> Tuple[List[Tuple[int, int, float]], int]:
    """
    Computes dense vertex-to-vertex correlation and thresholds to keep top K percent connections.
    
    Args:
        timeseries: (n_timepoints, n_vertices)
        top_k_percent: Percentile to keep (e.g., 0.1 means top 0.1%).
        
    Returns:
        edges: List of (source, target, weight) tuples.
        n_vertices: Number of vertices.
    """
    n_vertices = timeseries.shape[1]
    
    # Pearson correlation
    # Note: For large N, this is huge. We compute it row-by-row or in blocks if needed.
    # For N ~ 30k (surface), N^2 is manageable (~900M floats = 3.6GB).
    # If N > 50k, we should be careful.
    
    # Standardize time series to make dot product equivalent to correlation
    ts_std = (timeseries - timeseries.mean(axis=0)) / (timeseries.std(axis=0) + 1e-8)
    
    # Compute correlation matrix
    corr_matrix = np.dot(ts_std.T, ts_std) / timeseries.shape[0]
    
    # Zero out diagonal
    np.fill_diagonal(corr_matrix, 0)
    
    # Thresholding
    # Find threshold value
    # We only care about positive correlations usually for community detection
    # or absolute values? Infomap usually works with flow/positive weights.
    # Typically we use absolute correlation or just positive. Let's assume positive.
    
    # Get values to find percentile
    # Sampling approach if matrix is too big?
    # For exact percentile:
    threshold = np.percentile(corr_matrix, 100 - top_k_percent)
    
    # Sparse representation
    # Get indices where corr > threshold
    sources, targets = np.where(corr_matrix > threshold)
    weights = corr_matrix[sources, targets]
    
    edges = list(zip(sources, targets, weights))
    return edges, n_vertices


def run_infomap_community_detection(edges: List[Tuple[int, int, float]], n_nodes: int) -> np.ndarray:
    """
    Runs Infomap community detection on the sparse graph.
    
    Args:
        edges: List of (source, target, weight)
        n_nodes: Number of nodes
        
    Returns:
        modules: Array of shape (n_nodes,) with module ID for each node.
    """
    if Infomap is None:
        raise ImportError("Infomap is not installed. Please install with `pip install infomap`.")
        
    im = Infomap("--two-level --silent")
    
    # Add links
    for source, target, weight in edges:
        im.add_link(int(source), int(target), float(weight))
        
    # Run
    im.run()
    
    # Extract results
    modules = np.zeros(n_nodes, dtype=int)
    for node in im.tree:
        if node.is_leaf:
            modules[node.node_id] = node.module_id
            
    return modules


def match_communities_to_template(
    subject_modules: np.ndarray, 
    template_labels: np.ndarray,
    n_template_networks: int = 7
) -> np.ndarray:
    """
    Matches subject-specific communities to template networks based on spatial overlap (Dice).
    
    Args:
        subject_modules: (n_vertices,) module IDs
        template_labels: (n_vertices,) template network IDs (1..N)
        n_template_networks: Number of networks in template
        
    Returns:
        remapped_modules: (n_vertices,) with IDs matching template (1..N)
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
        # Note: Multiple subject modules can map to the same template network (fragmentation)
        # or we can enforce 1-to-1 if needed, but usually we aggregate them.
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
    
    Args:
        modules: (n_vertices,) network IDs (1..N)
        n_networks: Number of networks expected
        vertex_areas: (n_vertices,) Area of each vertex in mm^2. 
                      If None, assumes unit area (voxel count).
        
    Returns:
        areas: (n_networks,) area for each network (1..N)
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
    timeseries: np.ndarray, 
    template_labels: np.ndarray,
    top_k_percent: float = 0.1,
    vertex_areas: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Full workflow: Dense Connectivity -> Infomap -> Template Matching -> Surface Area Features.
    """
    # 1. Dense Connectivity & Thresholding
    edges, n_vertices = compute_dense_connectivity(timeseries, top_k_percent)
    
    # 2. Community Detection
    modules = run_infomap_community_detection(edges, n_vertices)
    
    # 3. Template Matching
    n_template_networks = int(np.max(template_labels))
    remapped_modules = match_communities_to_template(modules, template_labels, n_template_networks)
    
    # 4. Feature Extraction
    areas = calculate_network_surface_areas(remapped_modules, n_template_networks, vertex_areas)
    
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
