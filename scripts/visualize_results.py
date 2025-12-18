import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description="Visualize Normative Model Results")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing results files (subjects_summary.csv, advanced_features.csv)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots")
    parser.add_argument("--advanced_features_csv", type=str, default="", help="Path to advanced_features.csv (optional overrides default path)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Summary
    summary_path = os.path.join(args.results_dir, "subjects_summary.csv")
    if not os.path.exists(summary_path):
        print(f"Summary file not found at {summary_path}. Trying to regenerate from phenotype...", flush=True)
        # Fallback logic if needed, but for now assume it exists as per previous step
        sys.exit(1)
        
    df = pd.read_csv(summary_path)
    
    print(f"Loaded {len(df)} subjects.", flush=True)
    
    # Ensure DX_GROUP is readable
    # Assuming 1=ASD, 2=TD based on ABIDE convention usually
    # If not mapped, map it
    if "DX_GROUP" in df.columns:
        df["Diagnosis"] = df["DX_GROUP"].map({1: "ASD", 2: "TD"})
    
    # Try to merge advanced features if present
    adv_path = args.advanced_features_csv or os.path.join(args.results_dir, "advanced_features", "advanced_features.csv")
    adv_df = None
    if os.path.exists(adv_path):
        print(f"Loading advanced features from {adv_path}...", flush=True)
        try:
            adv_df = pd.read_csv(adv_path)
            # Merge on SUB_ID if available, otherwise row order
            if "SUB_ID" in adv_df.columns and "SUB_ID" in df.columns:
                df = pd.merge(df, adv_df, on="SUB_ID", how="left")
            else:
                # Align by index length
                if len(adv_df) == len(df):
                    df = pd.concat([df.reset_index(drop=True), adv_df.reset_index(drop=True)], axis=1)
                else:
                    print("Warning: advanced_features.csv length mismatch; skipping merge.", flush=True)
                    adv_df = None
        except Exception as e:
            print(f"Failed to load advanced features: {e}", flush=True)
            adv_df = None
    else:
        print(f"advanced_features.csv not found at {adv_path}. Skipping advanced plots.", flush=True)
    
    # 1. Histogram of Mean Absolute Deviation
    print("Generating deviation histogram...", flush=True)
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="mean_abs_z", hue="Diagnosis", kde=True, bins=30, alpha=0.6)
    plt.title("Distribution of Global Deviations (Mean Abs Z-Score)")
    plt.xlabel("Mean |Z|")
    plt.ylabel("Count")
    plt.savefig(os.path.join(args.output_dir, "deviation_histogram.png"))
    plt.close()
    
    # 2. Boxplot by Site
    print("Generating site comparison...", flush=True)
    plt.figure(figsize=(14, 8))
    # Sort sites by median deviation
    site_order = df.groupby("SITE_ID")["mean_abs_z"].median().sort_values().index
    sns.boxplot(data=df, x="SITE_ID", y="mean_abs_z", order=site_order)
    plt.xticks(rotation=45, ha="right")
    plt.title("Global Deviation by Site")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "site_deviation_boxplot.png"))
    plt.close()
    
    # 3. Violin Plot of Deviation by Diagnosis
    print("Generating diagnosis comparison...", flush=True)
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df, x="Diagnosis", y="mean_abs_z", inner="quartile")
    plt.title("Global Deviation: ASD vs TD")
    plt.savefig(os.path.join(args.output_dir, "diagnosis_deviation_violin.png"))
    plt.close()
    
    # 4. Age vs Deviation Scatter
    if "AGE_AT_SCAN" in df.columns:
        print("Generating Age vs Deviation scatter...", flush=True)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x="AGE_AT_SCAN", y="mean_abs_z", hue="Diagnosis", alpha=0.6)
        sns.regplot(data=df, x="AGE_AT_SCAN", y="mean_abs_z", scatter=False, color="black")
        plt.title("Global Deviation vs Age")
        plt.savefig(os.path.join(args.output_dir, "age_deviation_scatter.png"))
        plt.close()
        
    # 5. Outlier Count Analysis
    if "outlier_count_total" in df.columns:
        print("Generating Outlier Count analysis...", flush=True)
        
        # Boxplot
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x="Diagnosis", y="outlier_count_total")
        plt.title("Number of Extreme Deviations (|Z| > 1.96)")
        plt.ylabel("Count of Significant Regions")
        plt.savefig(os.path.join(args.output_dir, "outlier_count_boxplot.png"))
        plt.close()
        
        # Histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x="outlier_count_total", hue="Diagnosis", kde=True, bins=30, alpha=0.6)
        plt.title("Distribution of Extreme Deviations")
        plt.xlabel("Count of |Z| > 1.96")
        plt.savefig(os.path.join(args.output_dir, "outlier_count_histogram.png"))
        plt.close()
        
    # 6. Network-level comparisons (requires advanced features)
    if adv_df is not None:
        print("Generating network-level comparisons...", flush=True)
        # Find network columns
        net_cols = [c for c in df.columns if c.startswith("net_mean_")]
        if net_cols:
            # Barplot of mean network deviations by Diagnosis
            net_means = df.groupby("Diagnosis")[net_cols].mean().T
            plt.figure(figsize=(12, 8))
            net_means.plot(kind="bar")
            plt.title("Mean Network Deviations by Diagnosis")
            plt.ylabel("Mean |Z| (Top-10%)")
            plt.xlabel("Network")
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "network_mean_bar_by_diagnosis.png"))
            plt.close()
            
            # Heatmap of subject-wise network deviations (subset for readability)
            sample_df = df[net_cols + (["Diagnosis"] if "Diagnosis" in df.columns else [])].copy()
            # Map Diagnosis to numeric for sorting (ASD first)
            if "Diagnosis" in sample_df.columns:
                sample_df["DiagCode"] = sample_df["Diagnosis"].map({"ASD": 0, "TD": 1})
                sample_df = sample_df.sort_values("DiagCode")
                sample_df = sample_df.drop(columns=["DiagCode"])
            # Take first 100 subjects for heatmap
            heat_df = sample_df[net_cols].head(100)
            plt.figure(figsize=(12, 10))
            sns.heatmap(heat_df, cmap="viridis")
            plt.title("Subject-wise Network Deviations (Top-100 subjects)")
            plt.xlabel("Network")
            plt.ylabel("Subject")
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "network_subject_heatmap.png"))
            plt.close()
        else:
            print("No 'net_mean_' columns found in advanced features; skipping network plots.", flush=True)
    
    print(f"Plots saved to {args.output_dir}", flush=True)

if __name__ == "__main__":
    main()
