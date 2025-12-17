import argparse
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

def load_atlas_labels(path: str) -> pd.DataFrame:
    """
    Loads atlas labels from a file (CSV/Excel/Txt).
    Expected columns: 'index' (optional), 'label'/'name', 'network'.
    Returns DataFrame with 'label' and 'network' columns.
    """
    if path.endswith('.xlsx'):
        df = pd.read_excel(path)
    elif path.endswith('.csv'):
        try:
            df = pd.read_csv(path)
        except:
            df = pd.read_csv(path, sep='\t')
    elif path.endswith('.tsv') or path.endswith('.txt'):
        df = pd.read_csv(path, sep='\t')
    else:
        raise ValueError(f"Unsupported file format: {path}")
    
    # Normalize columns
    df.columns = [c.lower() for c in df.columns]
    
    # Find label column
    label_col = next((c for c in df.columns if 'label' in c or 'name' in c or 'roi' in c), None)
    if not label_col:
        # Fallback: assume first column
        label_col = df.columns[0]
        
    # Find network column
    network_col = next((c for c in df.columns if 'net' in c or 'module' in c or 'system' in c), None)
    
    out_df = pd.DataFrame()
    out_df['label'] = df[label_col]
    
    if network_col:
        out_df['network'] = df[network_col]
    else:
        out_df['network'] = 'Unknown'
        
    return out_df

def get_triu_mapping(n_features: int) -> int:
    """
    Determines N (number of regions) from M (number of edges).
    M = N * (N - 1) / 2
    2M = N^2 - N => N^2 - N - 2M = 0
    N = (1 + sqrt(1 + 8M)) / 2
    """
    delta = 1 + 8 * n_features
    n = (1 + np.sqrt(delta)) / 2
    if not n.is_integer():
        raise ValueError(f"Feature count {n_features} does not correspond to a valid upper triangle size.")
    return int(n)

def main():
    parser = argparse.ArgumentParser(description="Extract advanced features from Normative Model Z-scores")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing z_scores.npy")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save advanced features")
    parser.add_argument("--atlas_labels", type=str, help="Path to atlas labels file (CSV/Excel/TSV) for Network aggregation")
    parser.add_argument("--atlas_img", type=str, help="Path to atlas NIfTI image (for voxel-wise aggregation)")
    parser.add_argument("--top_k_percent", type=float, default=10.0, help="Percentage of top deviations to average (default: 10)")
    parser.add_argument("--use_harmonized", action="store_true", help="Use harmonized Z-scores if available")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Data
    z_filename = "z_scores_harmonized.npy" if args.use_harmonized else "z_scores.npy"
    z_path = os.path.join(args.results_dir, z_filename)
    
    if not os.path.exists(z_path) and args.use_harmonized:
        print(f"Harmonized Z-scores not found at {z_path}. Falling back to standard.", flush=True)
        z_path = os.path.join(args.results_dir, "z_scores.npy")
        
    if not os.path.exists(z_path):
        raise FileNotFoundError(f"Z-scores not found at {z_path}")
        
    print(f"Loading Z-scores from {z_path}...", flush=True)
    z = np.load(z_path)
    n_subjects, n_features = z.shape
    print(f"Loaded: {n_subjects} subjects, {n_features} features", flush=True)
    
    # Load pred_std for NLL if available
    std_path = os.path.join(args.results_dir, "pred_std.npy")
    pred_std = None
    if os.path.exists(std_path):
        print(f"Loading predicted std from {std_path}...", flush=True)
        pred_std = np.load(std_path)
    else:
        print("pred_std.npy not found. Skipping NLL calculation.", flush=True)
        
    # Initialize Output DataFrame
    features_df = pd.DataFrame()
    
    # --- Feature 1: Extreme Value Statistics (Top-K) ---
    print("Calculating Extreme Value Statistics...", flush=True)
    abs_z = np.abs(z)
    
    # Top K%
    k_percent = args.top_k_percent
    k = max(1, int(n_features * (k_percent / 100.0)))
    print(f"  Top {k_percent}% = {k} features", flush=True)
    
    # Sort and take top k (per subject)
    # np.sort is slow for large arrays, np.partition is faster
    top_k_vals = -np.partition(-abs_z, k, axis=1)[:, :k] # Get top k (unsorted)
    
    features_df[f"mean_top_{int(k_percent)}_percent"] = np.mean(top_k_vals, axis=1)
    features_df[f"median_top_{int(k_percent)}_percent"] = np.median(top_k_vals, axis=1)
    
    # Top 1% (Extreme)
    k_1 = max(1, int(n_features * 0.01))
    top_1_vals = -np.partition(-abs_z, k_1, axis=1)[:, :k_1]
    features_df["mean_top_1_percent"] = np.mean(top_1_vals, axis=1)
    
    # --- Feature 2: Probability-Based (NLL) ---
    if pred_std is not None:
        print("Calculating Negative Log-Likelihood (NLL)...", flush=True)
        # NLL = 0.5 * log(2*pi*sigma^2) + (x-mu)^2 / (2*sigma^2)
        # Since Z = (x-mu)/sigma, term 2 is 0.5 * Z^2
        # Term 1: log(sigma) + const
        # We compute mean NLL per subject
        
        # Avoid log(0)
        safe_std = np.where(pred_std < 1e-9, 1e-9, pred_std)
        
        term1 = np.log(safe_std) # Ignoring constants
        term2 = 0.5 * (z ** 2)
        
        nll_per_feature = term1 + term2
        features_df["mean_nll"] = np.mean(nll_per_feature, axis=1)
        
        # Also Top-K NLL?
        top_k_nll = -np.partition(-nll_per_feature, k, axis=1)[:, :k]
        features_df[f"mean_top_{int(k_percent)}_percent_nll"] = np.mean(top_k_nll, axis=1)
        
    # --- Feature 3: Network-Level Aggregation ---
    if args.atlas_img and os.path.exists(args.atlas_img):
        print(f"Loading atlas image from {args.atlas_img}...", flush=True)
        try:
            from nilearn import image
            atlas_img = image.load_img(args.atlas_img)
            atlas_data = atlas_img.get_fdata()
            
            # Assuming mask is 0=Background, 1..K=Networks
            # Flatten atlas data to match feature dimension?
            # Warning: This assumes z_scores are voxels extracted using the SAME mask order.
            # Usually NiftiLabelsMasker or NiftiMasker.
            # If NiftiMasker was used with this atlas as mask, then features = non-zero voxels.
            
            flat_atlas = atlas_data.ravel()
            non_zero_mask = flat_atlas != 0
            n_voxels = np.sum(non_zero_mask)
            
            if n_voxels == n_features:
                print(f"  Feature count ({n_features}) matches non-zero voxels in atlas image.", flush=True)
                print("  Aggregating voxel-wise Z-scores by Network...", flush=True)
                
                # Get network labels for each voxel
                voxel_networks = flat_atlas[non_zero_mask]
                unique_networks = np.unique(voxel_networks)
                
                # Map network IDs to names if TSV provided
                net_names = {}
                if args.atlas_labels and os.path.exists(args.atlas_labels):
                    try:
                        lbl_df = load_atlas_labels(args.atlas_labels)
                        # Assuming TSV has 'index' or 'label' matching the integer values in NIfTI
                        # If 'index' column exists, use it. Else assume row order + 1?
                        # Yeo TSV usually has 'Network' name.
                        # Let's try to find a mapping.
                        # Common Yeo TSV: "7Networks_1", "7Networks_2"...
                        # The NIfTI usually has 1..7 or 1..17.
                        
                        # Heuristic: Create a map {int_id: name}
                        # If 'index' col exists:
                        index_col = next((c for c in lbl_df.columns if 'index' in c or 'id' in c), None)
                        if index_col:
                            for _, row in lbl_df.iterrows():
                                net_names[int(row[index_col])] = str(row['network'])
                        else:
                            # Assume 1-based indexing
                            for i, row in enumerate(lbl_df.itertuples()):
                                net_names[i+1] = str(row.network)
                                
                    except Exception as e:
                        print(f"  Warning: Could not load network names from TSV: {e}", flush=True)
                
                for net_id in unique_networks:
                    if net_id == 0: continue # Should be excluded by non_zero_mask anyway
                    
                    mask = (voxel_networks == net_id)
                    if np.sum(mask) > 0:
                        net_z = abs_z[:, mask]
                        name = net_names.get(int(net_id), f"Network_{int(net_id)}")
                        # Clean name
                        name = name.replace(" ", "_").replace("-", "_")
                        features_df[f"net_mean_{name}"] = np.mean(net_z, axis=1)
                        
            else:
                 print(f"  WARNING: Feature count ({n_features}) does not match non-zero voxels in atlas ({n_voxels}). Skipping Voxel Aggregation.", flush=True)
                 
        except ImportError:
            print("  Error: nilearn not installed. Cannot load NIfTI atlas.", flush=True)
        except Exception as e:
            print(f"  Error processing NIfTI atlas: {e}", flush=True)

    elif args.atlas_labels and os.path.exists(args.atlas_labels):
        print(f"Loading atlas labels from {args.atlas_labels}...", flush=True)
        atlas_df = load_atlas_labels(args.atlas_labels)
        n_regions = len(atlas_df)
        print(f"  Found {n_regions} regions in atlas.", flush=True)
        
        # Determine if features are ROIs or Edges
        if n_features == n_regions:
            print("  Features correspond to Regions (ROI-wise). Aggregating by Network...", flush=True)
            networks = atlas_df['network'].unique()
            print(f"  Found networks: {networks}", flush=True)
            
            for net in networks:
                if pd.isna(net) or net == 'Unknown': continue
                mask = (atlas_df['network'] == net).values
                if np.sum(mask) > 0:
                    net_z = abs_z[:, mask]
                    features_df[f"net_mean_{net}"] = np.mean(net_z, axis=1)
                    
        else:
            # Check if Edges
            try:
                n_regions_calc = get_triu_mapping(n_features)
                if n_regions_calc == n_regions:
                    print(f"  Features correspond to Edges between {n_regions} regions.", flush=True)
                    
                    # Create Network Mapping for Edges
                    # We need to map each feature index k -> (i, j) -> (Net_i, Net_j)
                    # This loop is slow in Python, we need to vectorize or group smartly.
                    
                    print("  Mapping edges to networks (this may take a moment)...", flush=True)
                    
                    # Pre-compute network indices
                    net_to_indices = {}
                    networks = [n for n in atlas_df['network'].unique() if pd.notna(n) and n != 'Unknown']
                    for net in networks:
                        net_to_indices[net] = set(atlas_df[atlas_df['network'] == net].index)
                        
                    # Indices for upper triangle
                    # k = 0..M-1
                    # We can iterate over network PAIRS
                    
                    # Generate all pairs of indices (i, j) for upper triangle
                    # This is O(N^2), ~40k for N=200, ~300k for N=780. Fast enough.
                    
                    triu_indices = np.triu_indices(n_regions, k=1)
                    src_indices = triu_indices[0]
                    dst_indices = triu_indices[1]
                    
                    # Vectorized mapping
                    # Map region index -> network name
                    region_nets = atlas_df['network'].values
                    
                    src_nets = region_nets[src_indices]
                    dst_nets = region_nets[dst_indices]
                    
                    # For each network (Within-Network)
                    for net in networks:
                        # Find edges where both src and dst are in net
                        mask = (src_nets == net) & (dst_nets == net)
                        count = np.sum(mask)
                        if count > 0:
                            # Extract columns
                            net_z = abs_z[:, mask]
                            features_df[f"net_mean_{net}"] = np.mean(net_z, axis=1)
                            # features_df[f"net_max_{net}"] = np.max(net_z, axis=1) # Optional
                            
                    # For Between-Network (optional, maybe too many features? 7x7=49)
                    # Let's add them for major networks
                    # Or maybe just "DMN_vs_Salience"
                    
                else:
                    print(f"  WARNING: Feature count {n_features} does not match edges for {n_regions} regions (Expected {n_regions*(n_regions-1)//2}). Skipping Network Aggregation.", flush=True)
                    
            except ValueError:
                print(f"  WARNING: Feature count {n_features} is not a valid upper triangle size. Skipping Network Aggregation.", flush=True)
                
    else:
        if args.atlas_labels:
            print(f"Atlas file not found: {args.atlas_labels}", flush=True)
        else:
            print("No atlas labels provided. Skipping Network Aggregation.", flush=True)
            
    # Save
    out_path = os.path.join(args.output_dir, "advanced_features.csv")
    
    # Add Subject IDs if available
    summary_path = os.path.join(args.results_dir, "subjects_summary.csv")
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        # Check length
        if len(summary_df) == len(features_df):
            # Prepend ID columns
            id_cols = [c for c in summary_df.columns if 'ID' in c or 'GROUP' in c or 'SITE' in c]
            features_df = pd.concat([summary_df[id_cols], features_df], axis=1)
    
    features_df.to_csv(out_path, index=False)
    print(f"Saved advanced features to {out_path}", flush=True)
    print("Columns:", list(features_df.columns), flush=True)

if __name__ == "__main__":
    main()
