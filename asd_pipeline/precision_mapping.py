import numpy as np
import os
from typing import Tuple, List, Optional, Dict
from nilearn.decomposition import DictLearning, CanICA
from nilearn.image import load_img, resample_to_img, math_img
from nilearn.masking import apply_mask
import nibabel as nib

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
