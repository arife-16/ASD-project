 # ASD Project Technical Report

## Overview

This report documents the rs-fMRI pipeline for ASD classification implemented in this repository. It covers data handling, preprocessing, feature engineering (static and dynamic connectivity, graph metrics, ALFF, temporal state features), site harmonization, covariate-aware normative modeling, and site-aware model evaluation. Key orchestration is via `scripts/run_pipeline.py:43`.

## Data & Preprocessing

- ROI time series extraction (`asd_pipeline/atlas.py:7`): uses `nilearn` maskers to extract ROI-level signals from NIfTI based on an atlas image, optionally applying bandpass filtering (`high_pass=0.01`, `low_pass=0.1`) and standardization.
- Confounds (`asd_pipeline/confounds.py:47`): combines motion (TSV) and tissue signals (WM/CSF) per subject into a confound matrix.
- Regression and cleaning (`asd_pipeline/preprocess.py:7`): ordinary least squares residualization per ROI followed by `nilearn.signal.clean`.
  - OLS per ROI: for confounds `C \in \mathbb{R}^{T\times p}` and time series `x_i \in \mathbb{R}^{T}`, solve `\beta_i = \arg\min_\beta \|x_i - C\beta\|_2^2`; residuals `\hat{x}_i = x_i - C\beta_i`.
  - Bandpass focuses on 0.01–0.1 Hz, typical for rs-fMRI.
                                
## Functional Connectivity Features

- Static FC (`asd_pipeline/features.py:6`): per-ROI z-scoring across time then Pearson correlation matrix `C \in \mathbb{R}^{R\times R}`.
  - \( C_{ij} = \frac{\sum_t (x_{i,t}-\bar{x}_i)(x_{j,t}-\bar{x}_j)}{(T-1)\,\sigma_i\sigma_j} \).
- Upper-triangle vectorization (`asd_pipeline/features.py:12`): returns `v \in \mathbb{R}^{R(R-1)/2}`.
- Dynamic FC (`asd_pipeline/features.py:27`): sliding windows over time (`asd_pipeline/features.py:17`) produce window-wise FC vectors `v^{(w)}`; outputs mean and std across windows:
  - \( \mu = \frac{1}{W}\sum_{w=1}^{W} v^{(w)} \), \( \sigma = \sqrt{\frac{1}{W-1}\sum_{w}(v^{(w)}-\mu)^2} \).
- ALFF (`asd_pipeline/features.py:41`): average FFT power in 0.01–0.1 Hz per ROI.
  - \( \text{ALFF}_i = \frac{1}{|F|}\sum_{f\in[0.01,0.1]} |X_i(f)|^2 \).
- Combined feature vector (`asd_pipeline/features.py:50`): concatenates static FC, dynamic mean/std, and ALFF. Returns `(feat, idxs)` where `idxs` maps named segments (e.g., `fc`, `dyn_mean`, `dyn_std`, `alff`).

## Temporal State Metrics

- K-means over windowed FC (`asd_pipeline/features.py:66`): cluster `v^{(w)}` into `K` states.
- Metrics (`asd_pipeline/features.py:66–123`):
  - Occupancy: \( p_s = \frac{1}{W}\sum_t \mathbf{1}[s_t=s] \).
  - Transition probabilities: \( P_{ab} = \frac{N_{a\to b}}{\sum_b N_{a\to b}} \).
  - Dwell mean/std: run-length statistics per state.
  - Entropy: \( H = -\sum_s p_s \log p_s \).
  - Asymmetry: \( A = \sum_{i,j} |P_{ij} - P_{ji}| \).
- Augmented feature vector (`asd_pipeline/features.py:125`): base connectivity features plus state metrics.

## Connectome Metrics

- Partial correlation via precision (`asd_pipeline/connectome.py:12`): fit Ledoit–Wolf covariance `\hat{\Sigma}`; precision `P = \hat{\Sigma}^{-1}`; partial correlation:
  - \( \rho^{\text{partial}}_{ij} = -\frac{P_{ij}}{\sqrt{P_{ii}P_{jj}}} \) with diagonal set to 1.
- Graph features:
  - Thresholded adjacency (`asd_pipeline/connectome.py:22`): binary adjacency from absolute FC with threshold `thr`.
  - Node strength (`asd_pipeline/connectome.py:26`): \( s_i = \sum_j |C_{ij}| \).
  - Local clustering coefficient (`asd_pipeline/connectome.py:30`): triangles among neighbors normalized by \( \binom{k_i}{2} \).
- Network-level summaries (`asd_pipeline/connectome.py:45`): intra-network mean FC and inter-network mean FC.
- Windowed statistics and variance from sums (`asd_pipeline/connectome.py:144`): compute std via \( \operatorname{Var} = \frac{\sum x^2 - (\sum x)^2/n}{n-1} \), then \( \sigma=\sqrt{\operatorname{Var}} \).
- Final connectome feature vector (`asd_pipeline/connectome.py:165–178`): concatenates FC/PC mean/std (upper-triangle), strength/clustering mean/std, intra/inter-network mean/std.

## Site Harmonization (ComBat)

- ComBat (`asd_pipeline/harmonize.py:7`): empirical Bayes to remove batch (`SITE`) effects while preserving covariate (`AGE`, `SEX`) effects.
  - Per feature i: \( y_{ij} = \alpha_i + \beta_i^T X_j + \gamma_{i,b(j)} + \delta_{i,b(j)}\,\epsilon_{ij} \). Harmonized features are obtained by subtracting `\gamma` and scaling by `\delta`.

## Normative Modeling

- Covariate residualization (`asd_pipeline/normative.py:25`): fit TD-only OLS from covariates to features; apply to all subjects and subtract expected value.
- Z-scores and Mahalanobis (`asd_pipeline/normative.py:37`):
  - Feature z-scores relative to TD: \( z = \frac{x - \mu}{\sigma} \).
  - Mahalanobis distance per subject: \( D_M(x) = \sqrt{(x-\mu)^\top \Sigma^{-1} (x-\mu)} \).
- Returns per-feature deviations (`feature_z`) and global deviation (`mahalanobis`).

## Models

- Pipelines (`asd_pipeline/model.py:29`):
  - `logistic_l2`: `StandardScaler` → `NormDiffSelector(k=500)` → Logistic L2 (`liblinear`) with class weights.
  - `logistic_elasticnet`: elastic-net logistic (`saga`, `l1_ratio=0.5`).
  - `linear_svc_calibrated`: `LinearSVC` calibrated to probabilities via `CalibratedClassifierCV(method="sigmoid")`.
- Simple filter selector (`asd_pipeline/model.py:77`): top-k features by absolute class mean difference.
  - \( \mathrm{score}_j = | \bar{x}_{1,j} - \bar{x}_{0,j} | \).
- Losses:
  - Logistic: \( \sum_i \log(1+\exp(-y_i w^\top x_i)) + \lambda\,R(w) \), where `R` is L2 or elastic-net.
  - Linear SVM: \( \tfrac{1}{2}\|w\|_2^2 + C \sum_i \max(0,\,1 - y_i w^\top x_i) \); probabilities via Platt scaling.

## Cross-Validation

- Strategies (`asd_pipeline/model.py:58`): `stratified`, `group` (`GroupKFold` on `SITE`), `loso` (`LeaveOneGroupOut` by site), and `site_stratified` (`asd_pipeline/model.py:38`).
- Scoring: ROC AUC, F1, accuracy (`asd_pipeline/model.py:69`). Outputs mean scores across folds.

## Pipeline Orchestration

- CLI (`scripts/run_pipeline.py:43`):
  1. Read phenotype CSV (requires `SUB_ID`, `SITE`, `DX_GROUP`, `AGE_AT_SCAN`, `SEX`).
  2. Extract ROI time series (`asd_pipeline/atlas.py:7`), build confounds (`asd_pipeline/confounds.py:47`), and regress confounds (`asd_pipeline/preprocess.py:7`).
  3. Build features with or without state metrics (`asd_pipeline/features.py:50`, `asd_pipeline/features.py:125`).
  4. Harmonize features via ComBat (`asd_pipeline/harmonize.py:7`).
  5. Compute normative deviations (`asd_pipeline/normative.py:37`).
  6. Evaluate tuned models across CV strategies or baseline classifier (`asd_pipeline/model.py:58`).
  7. Save JSON results.

## Tests

- Feature windows and state metric shapes: `tests/test_features.py:7–27`.
- CV split coverage and label balance: `tests/test_cv.py:11–16`.
- Normative outputs shapes: `tests/test_normative.py:11–15`.
- Model evaluation API: `tests/test_model.py:11–22`.

## Notebook Experiments (Complementary)

- Classical ML over FC: `notebooks/functional_connectivity_analysis.ipynb` generates 19,900-dim FC features, trains logistic/SVM with ROC AUC evaluation.
- CNN over FC matrices: `notebooks/Non_linear_Relation_Capturing_and_CNN.ipynb` builds a 2D CNN with Keras and compares against SVM, often favoring SVM stability in high-dim settings.
- Domain-adversarial training: `notebooks/Deep_Learning_with_ABIDE.ipynb` implements GRL and domain discriminator in PyTorch (`notebooks/Deep_Learning_with_ABIDE.ipynb:272–313`) to reduce site bias. Objective:
  - \( \min_{F,C} \max_{D} \; L_y(C(F(X)), y) - \lambda L_d(D(F(X)), s) \), where `s` denotes site labels.

## Mathematical Notes & Assumptions

- Pearson and partial correlations assume approximately stationary signals post-cleaning; Ledoit–Wolf improves covariance estimation in high dimensions.
- ComBat assumes additive/multiplicative site effects with shared priors across batches; covariates preserved.
- Normative z-scores treat TD distribution per feature as Gaussian with empirical `\mu, \sigma`; Mahalanobis uses MVN fitted to TD for multivariate deviation.
- Classifiers rely on regularization and simple filter selection to mitigate `p >> n`.

## Reproducibility

- Primary pipeline entry: `scripts/run_pipeline.py --phenotype phenotype.csv --nifti_dir ... --atlas ... --cv_strategy site_stratified --tune_models --output pipeline_results.json` (see `README.md:44`).
- Unit tests can be executed via `python3 -m unittest discover -s tests -p "test_*.py" -v`.

## References (Code Locations)

- Atlas extraction: `asd_pipeline/atlas.py:7`.
- Confounds: `asd_pipeline/confounds.py:47`.
- Cleaning: `asd_pipeline/preprocess.py:7`.
- FC and dynamic features: `asd_pipeline/features.py:6`, `asd_pipeline/features.py:27`.
- State metrics: `asd_pipeline/features.py:66`.
- Connectome metrics: `asd_pipeline/connectome.py:12`, `asd_pipeline/connectome.py:26`, `asd_pipeline/connectome.py:30`, `asd_pipeline/connectome.py:45`.
- Harmonization: `asd_pipeline/harmonize.py:7`.
- Normative modeling: `asd_pipeline/normative.py:37`.
- Models and CV: `asd_pipeline/model.py:29`, `asd_pipeline/model.py:58`.
- Orchestration: `scripts/run_pipeline.py:43`.

## BNA Atlas Report

- Purpose
  - Uses the Brainnetome (BNA) atlas to parcellate rs-fMRI into ROI time series and derive network-aware connectivity features for ASD analysis.
- Files and Location
  - Atlas image: `third_party/autism_connectome/fullbrain_atlas_thr0-2mm.nii.gz`.
  - Labels: `third_party/autism_connectome/BNA_subregions.xlsx`.
  - Resolved at runtime in `scripts/run_bna_pipeline.py:43–46` and verified in strict mode at `scripts/run_bna_pipeline.py:71–72`.
- Strict Mode and Fallback
  - When `--strict_bna` is set, the pipeline requires both atlas and labels; else raises an error: `scripts/run_bna_pipeline.py:71–72`.
  - If BNA is not available and strict mode is off, it falls back to Harvard–Oxford (`cort-maxprob-thr25-2mm`) at `scripts/run_bna_pipeline.py:79–82`.
- Label and Network Mapping
  - Labels and optional network assignments are loaded from the Excel file via `asd_pipeline/atlas_labels.py:6–23`.
  - The loader auto-detects name and network columns and returns `[names], [networks]` (networks may be `None`).
- ROI Extraction
  - Per subject, the pipeline extracts ROI time series with `NiftiLabelsMasker` using the BNA atlas image and bandpass/standardization parameters: `asd_pipeline/atlas.py:7–29`.
  - Parameters used by the BNA pipeline: `tr=2.0`, `high_pass=0.01`, `low_pass=0.1`, `standardize="zscore"`, `detrend=True`: `scripts/run_bna_pipeline.py:120–122`.
  - Time series are saved under `out_dir/_roi_ts/<SUB_ID>.npy`: `scripts/run_bna_pipeline.py:34–35, 121–122`.
- NIfTI Discovery and Naming
  - The pipeline supports custom file naming via `--nifti_pattern` and now accepts `{SUB_ID}`, `{FILE_ID}`, and `{SITE}` placeholders: `scripts/run_bna_pipeline.py:108–116`.
  - If no pattern is provided, it tries common names: `<SUB_ID>.nii.gz`, `<SUB_ID>_func_preproc.nii.gz`, `<SUB_ID>_bold.nii.gz`, `<SUB_ID>_bold_preproc.nii.gz`: `scripts/run_bna_pipeline.py:112–116`.
  - Subject IDs originate from phenotype `FILE_ID` when available, otherwise `SUB_ID`: `scripts/run_bna_pipeline.py:40–42`.
- Phenotype Requirements
  - Required columns: `SUB_ID` or `FILE_ID`, `DX_GROUP`, `AGE_AT_SCAN`, `SEX`, `SITE`.
  - The pipeline reads `--phenotype_path` or `out_dir/phenotype.csv`, with a clear error if missing: `scripts/run_bna_pipeline.py:36–39`.
- Feature Engineering
  - After ROI extraction, it builds a connectome feature vector per subject (static FC, partial correlation, graph summaries, network means): `asd_pipeline/connectome.py:12–48, 165–178`.
  - Network-aware aggregations use the BNA networks returned by `load_bna_labels`: passed as `networks` in `scripts/run_bna_pipeline.py:129–131`.
- Normative Deviations and Evaluation
  - TD-only reference distribution is used to compute per-feature z-scores and subject-level deviations: `asd_pipeline/normative.py:25–60`.
  - Site-aware CV (`site_stratified`) evaluates classification performance on deviation features: `scripts/run_bna_pipeline.py:139–143`.
- Drive Integration and Reproducibility
  - Paths can point to Google Drive for phenotype, NIfTI, atlas, and output directories. Use `--nifti_pattern` with `{SITE}` to match ABIDE naming (e.g., `Caltech_0051456_func_preproc.nii.gz`).
  - Recommended command:
    - `python scripts/run_bna_pipeline.py --phenotype_path "/content/drive/MyDrive/ABIDE_Project/phenotype.csv" --nifti_dir "/content/drive/MyDrive/ABIDE_Project/NIFTI" --nifti_pattern "{SITE}_{FILE_ID}_func_preproc.nii.gz" --atlas_path "/content/drive/MyDrive/ABIDE_Project/BNA/fullbrain_atlas_thr0-2mm.nii.gz" --labels_path "/content/drive/MyDrive/ABIDE_Project/BNA/BNA_subregions.xlsx" --strict_bna --out_dir "/content/drive/MyDrive/ABIDE_Project/ASD_Results"`.
- Outputs
  - ROI time series: `out_dir/_roi_ts/*.npy`.
  - Feature matrix summary and model metrics: `out_dir/bna_results.json` from `scripts/run_bna_pipeline.py:141–145`.
  - Intermediate arrays kept in memory during feature construction; per-subject vectors are stacked at `scripts/run_bna_pipeline.py:131–134`.

### Neuroscientific Approach

- Connectivity-based parcellation
  - BNA defines subregions by grouping voxels with similar long-range connectivity profiles derived from diffusion MRI tractography (structural) and resting-state fMRI correlations (functional).
  - Each subregion has a “connectivity fingerprint” that captures its preferential connections across the brain; boundaries are placed to maximize within-region homogeneity and between-region separability.
- Multimodal validation and hierarchy
  - Parcellation is validated across modalities and cohorts for reproducibility; boundaries respect macroanatomy while subdividing large gyri into finer units to reflect functional specialization.
  - Hemispheric symmetry and hierarchical organization ensure that subregions map consistently across left/right hemispheres and across scales.
- Network assignment
  - Subregions are assigned to large-scale canonical systems (e.g., default-mode, attention, somatomotor, visual, limbic, frontoparietal control, salience) based on their connectivity fingerprints and literature alignment.
  - The Excel labels file provides the network membership used by the pipeline: `asd_pipeline/atlas_labels.py:6–23`.
- Extraction and signal model
  - ROI signals represent region-level averages after detrending, bandpass filtering (0.01–0.1 Hz), and standardization; see `asd_pipeline/atlas.py:7–29`.
  - This frequency band targets spontaneous neural fluctuations typical in rs-fMRI while reducing physiological noise, enabling stable functional connectivity estimates.
- Implications for ASD analysis
  - Region-level signals and network membership enable system-level summaries (intra- and inter-network connectivity), which are sensitive to ASD-related dysconnectivity hypotheses.
  - Personalized deviations measured against TD-only distributions quantify atypical connectivity at the feature level and support subject-specific interpretation.
- Limitations and considerations
  - BNA is derived primarily from healthy populations; pediatric or clinical cohorts may exhibit anatomical/functional shifts that reduce atlas alignment fidelity.
  - Small subregions can be susceptible to partial volume and registration errors; careful QC and motion/confound regression mitigate these risks (`asd_pipeline/preprocess.py:7`).
  - Site effects in ABIDE require harmonization and site-aware evaluation to avoid confounding interpretations (`asd_pipeline/harmonize.py:7`, `scripts/run_bna_pipeline.py:139–143`).
