import numpy as np
import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description="Analyze Z-score outliers and generate probability maps")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing z_scores.npy")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save analysis results")
    parser.add_argument("--threshold", type=float, default=1.96, help="Z-score threshold for outliers")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Data
    z_path = os.path.join(args.results_dir, "z_scores.npy")
    harm_path = os.path.join(args.results_dir, "z_scores_harmonized.npy")
    
    if os.path.exists(harm_path):
        print(f"Loading harmonized Z-scores from {harm_path}", flush=True)
        z = np.load(harm_path)
    elif os.path.exists(z_path):
        print(f"Loading Z-scores from {z_path}", flush=True)
        z = np.load(z_path)
    else:
        raise FileNotFoundError(f"No z_scores.npy found in {args.results_dir}")
        
    print(f"Z-scores shape: {z.shape}", flush=True)
    n_subjects, n_features = z.shape
    
    pheno_path = os.path.join(args.results_dir, "subjects_summary.csv")
    if not os.path.exists(pheno_path):
        raise FileNotFoundError(f"subjects_summary.csv not found in {args.results_dir}")
        
    df = pd.read_csv(pheno_path)
    
    if len(df) != n_subjects:
        print(f"Warning: Phenotype rows ({len(df)}) != Z-scores rows ({n_subjects}). Truncating to minimum.", flush=True)
        min_len = min(len(df), n_subjects)
        z = z[:min_len]
        df = df.iloc[:min_len]
        
    # 2. Identify Groups
    # Assuming DX_GROUP: 1=ASD, 2=TD (ABIDE standard)
    if "DX_GROUP" not in df.columns:
        raise ValueError("DX_GROUP not in phenotype")
        
    # Standardize to 1=ASD, 0=TD
    # If 1 and 2 exist, assume 1=ASD, 2=TD (Wait, ABIDE is usually 1=Autism, 2=Control)
    # Let's check values
    classes = sorted(df["DX_GROUP"].unique())
    print(f"Classes found: {classes}", flush=True)
    
    if 1 in classes and 2 in classes:
        # ABIDE: 1=ASD, 2=TD
        df["is_asd"] = (df["DX_GROUP"] == 1).astype(int)
        print("Mapped: 1 -> ASD, 2 -> TD", flush=True)
    elif 0 in classes and 1 in classes:
        # Assuming 1=ASD, 0=TD
        df["is_asd"] = (df["DX_GROUP"] == 1).astype(int)
        print("Mapped: 1 -> ASD, 0 -> TD", flush=True)
    else:
        print(f"Warning: Unknown classes {classes}. Treating {classes[0]} as TD, {classes[-1]} as ASD.")
        df["is_asd"] = (df["DX_GROUP"] == classes[-1]).astype(int)

    asd_indices = df[df["is_asd"] == 1].index
    td_indices = df[df["is_asd"] == 0].index
    
    print(f"ASD subjects: {len(asd_indices)}, TD subjects: {len(td_indices)}", flush=True)
    
    # 3. Calculate Outliers
    threshold = args.threshold
    outliers = np.abs(z) > threshold
    
    # Per-subject counts
    counts = np.sum(outliers, axis=1)
    df["recalc_outlier_count"] = counts
    
    print("\n--- Outlier Count Stats ---", flush=True)
    print(f"Mean Outliers per Subject: {np.mean(counts):.2f} (Total Features: {n_features})", flush=True)
    print(f"Median Outliers per Subject: {np.median(counts):.2f}", flush=True)
    print(f"Min: {np.min(counts)}, Max: {np.max(counts)}", flush=True)
    
    mean_asd = np.mean(counts[asd_indices])
    mean_td = np.mean(counts[td_indices])
    print(f"Mean ASD: {mean_asd:.2f}, Mean TD: {mean_td:.2f}", flush=True)
    
    # Check for Inlier/Outlier Flip
    inliers = np.abs(z) <= threshold
    inlier_counts = np.sum(inliers, axis=1)
    print(f"Mean Inliers per Subject: {np.mean(inlier_counts):.2f}", flush=True)
    
    if np.mean(counts) > (n_features * 0.5):
        print("\nWARNING: More than 50% of features are outliers! You might be counting Inliers or model is poor.", flush=True)
        
    # 4. Probability Maps (Frequency per Feature)
    print("\nCalculating Probability Maps...", flush=True)
    
    freq_asd = np.mean(outliers[asd_indices], axis=0)
    freq_td = np.mean(outliers[td_indices], axis=0)
    
    # --- ADDED: Filter Broken Features ---
    freq_all = np.mean(outliers, axis=0)
    broken_mask = freq_all > 0.90
    n_broken = np.sum(broken_mask)
    print(f"\n--- Broken Feature Detection ---", flush=True)
    print(f"Found {n_broken} features that are outliers in >90% of ALL subjects.", flush=True)
    if n_broken > 0:
        print("These are likely artifacts (e.g., constant 0 in training, leading to tiny std).", flush=True)
        print("Recalculating counts excluding these features...", flush=True)
        
        valid_mask = ~broken_mask
        clean_outliers = outliers[:, valid_mask]
        clean_counts = np.sum(clean_outliers, axis=1)
        df["clean_outlier_count"] = clean_counts
        
        print(f"Cleaned Mean Outliers: {np.mean(clean_counts):.2f} (Total Valid Features: {np.sum(valid_mask)})", flush=True)
        
        mean_asd_clean = np.mean(clean_counts[asd_indices])
        mean_td_clean = np.mean(clean_counts[td_indices])
        print(f"Cleaned Mean ASD: {mean_asd_clean:.2f}, Cleaned Mean TD: {mean_td_clean:.2f}", flush=True)
        
        # Plot Cleaned Counts
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x="is_asd", y="clean_outlier_count")
        plt.xticks([0, 1], ["TD", "ASD"])
        plt.title(f"Outlier Count (Excluding {n_broken} Broken Features)")
        plt.savefig(os.path.join(args.output_dir, "clean_outlier_boxplot.png"))
        plt.close()
    # -------------------------------------
    
    diff_map = freq_asd - freq_td
    
    # Save Map Data
    map_data = pd.DataFrame({
        "feature_idx": range(n_features),
        "freq_asd": freq_asd,
        "freq_td": freq_td,
        "diff": diff_map,
        "abs_diff": np.abs(diff_map)
    })
    
    map_file = os.path.join(args.output_dir, "outlier_probability_map.csv")
    map_data.to_csv(map_file, index=False)
    print(f"Saved probability map to {map_file}", flush=True)
    
    # 5. Visualization
    print("Generating plots...", flush=True)
    
    # A. Frequency Comparison Plot (Sorted by Difference)
    # Sort features by difference to see separation
    sorted_idx = np.argsort(diff_map) # Ascending
    
    plt.figure(figsize=(12, 6))
    # Plot top 100 features where ASD > TD
    top_asd_features = sorted_idx[-100:]
    plt.plot(freq_asd[top_asd_features], label="ASD freq", color="red", alpha=0.7)
    plt.plot(freq_td[top_asd_features], label="TD freq", color="blue", alpha=0.7)
    plt.title("Top 100 Discriminative Features (ASD > TD)")
    plt.xlabel("Feature Rank")
    plt.ylabel("Outlier Frequency")
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "top_features_asd_gt_td.png"))
    plt.close()
    
    # B. Histogram of Frequencies
    plt.figure(figsize=(10, 6))
    sns.histplot(freq_asd, color="red", label="ASD", kde=True, element="step", alpha=0.3)
    sns.histplot(freq_td, color="blue", label="TD", kde=True, element="step", alpha=0.3)
    plt.title("Distribution of Outlier Frequencies Across All Features")
    plt.xlabel("Frequency (Fraction of Subjects with |Z| > 1.96)")
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "feature_frequency_dist.png"))
    plt.close()
    
    # C. Recalculated Boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="is_asd", y="recalc_outlier_count")
    plt.xticks([0, 1], ["TD", "ASD"])
    plt.title("Recalculated Outlier Count")
    plt.savefig(os.path.join(args.output_dir, "recalc_outlier_boxplot.png"))
    plt.close()

if __name__ == "__main__":
    main()
