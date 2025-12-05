from typing import Optional, Tuple
import numpy as np
import pandas as pd
from neuroHarmonize import harmonizationLearn, harmonizationApply


def combat_harmonize(features: np.ndarray, covars: pd.DataFrame, site_col: str, continuous_covars: Optional[list] = None, categorical_covars: Optional[list] = None) -> Tuple[np.ndarray, dict]:
    model = harmonizationLearn(features, covars, site_col, continuous_covars or [], categorical_covars or [])
    harmonized, _ = harmonizationApply(features, covars, model)
    return harmonized, model
