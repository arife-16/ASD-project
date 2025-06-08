# ASD-project

ROIs and Atlases: ROIs are specific brain regions that are defined based on anatomical or functional criteria.  Brain atlases (like AAL, Schaefer, Harvard-Oxford, etc.) provide standardized definitions and locations of these ROIs in a standard space.  When calculating FCs, you use an atlas to determine which voxels (3D pixels) in the fMRI data belong to which ROI.  Then, you can calculate the average fMRI signal within each ROI and compute the correlation between the average signals of different ROIs. This correlation value becomes the entry (i, j) in the FC matrix.

Functional Connectivity Matrices (FCs): A very common approach is to compute Functional Connectivity Matrices (FCs) from the raw fMRI data.  These matrices represent the statistical relationships (usually correlations) between the activity of different brain regions.  Each element (i, j) of the FC matrix represents the connectivity strength between brain region i and brain region j.
