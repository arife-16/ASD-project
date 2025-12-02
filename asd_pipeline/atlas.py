from typing import Optional, Tuple
import numpy as np
from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker
from nilearn.image import load_img


def extract_roi_timeseries(nifti_path: str, atlas_path: str, atlas_type: str = "labels", tr: Optional[float] = None, high_pass: Optional[float] = None, low_pass: Optional[float] = None, detrend: bool = True, standardize: str = "zscore") -> np.ndarray:
    img = load_img(nifti_path)
    atlas = load_img(atlas_path)
    if atlas_type == "labels":
        masker = NiftiLabelsMasker(
            labels_img=atlas,
            standardize=standardize,
            detrend=detrend,
            t_r=tr,
            high_pass=high_pass,
            low_pass=low_pass,
        )
    else:
        masker = NiftiMapsMasker(
            maps_img=atlas,
            standardize=standardize,
            detrend=detrend,
            t_r=tr,
            high_pass=high_pass,
            low_pass=low_pass,
        )
    ts = masker.fit_transform(img)
    return ts.T
