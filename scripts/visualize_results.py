import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description="Visualize Normative Model Results")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing z_scores.npy and subjects_summary.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots")
    
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
        
    print(f"Plots saved to {args.output_dir}", flush=True)

if __name__ == "__main__":
    main()
