from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from nilearn.image import load_img
from nilearn.masking import compute_brain_mask
from nilearn.maskers import NiftiMasker


def load_tsv_confounds(tsv_path: str, preferred_columns: Optional[List[str]] = None) -> Optional[np.ndarray]:
    if not tsv_path or not os.path.exists(tsv_path):
        return None
    df = pd.read_csv(tsv_path, sep="\t")
    cols = preferred_columns or [
        "trans_x",
        "trans_y",
        "trans_z",
        "rot_x",
        "rot_y",
        "rot_z",
        "framewise_displacement",
    ]
    available = [c for c in cols if c in df.columns]
    if len(available) == 0:
        return None
    M = df[available].values
    M = np.nan_to_num(M)
    return M


def extract_tissue_confounds(nifti_path: str, wm_mask_path: Optional[str] = None, csf_mask_path: Optional[str] = None, tr: Optional[float] = None) -> Optional[np.ndarray]:
    img = load_img(nifti_path)
    cols = []
    if wm_mask_path:
        wm_masker = NiftiMasker(mask_img=wm_mask_path, standardize=False, detrend=False, t_r=tr)
        wm_ts = wm_masker.fit_transform(img)
        cols.append(wm_ts.mean(axis=1))
    if csf_mask_path:
        csf_masker = NiftiMasker(mask_img=csf_mask_path, standardize=False, detrend=False, t_r=tr)
        csf_ts = csf_masker.fit_transform(img)
        cols.append(csf_ts.mean(axis=1))
    if len(cols) == 0:
        return None
    M = np.vstack(cols).T
    return M


def build_confounds(nifti_path: str, tsv_path: Optional[str], wm_mask_path: Optional[str], csf_mask_path: Optional[str], tr: Optional[float]) -> Optional[np.ndarray]:
    tsv = load_tsv_confounds(tsv_path)
    tissues = extract_tissue_confounds(nifti_path, wm_mask_path, csf_mask_path, tr=tr)
    if tsv is None and tissues is None:
        return None
    if tsv is None:
        return tissues
    if tissues is None:
        return tsv
    n = min(tsv.shape[0], tissues.shape[0])
    return np.hstack([tsv[:n], tissues[:n]])
