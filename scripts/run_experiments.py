import os
import json
import argparse
from asd_pipeline.config import load_config
from scripts.run_pipeline import main as pipeline_main


def run_config(config_path: str):
    cfg = load_config(config_path)
    os.makedirs(cfg.get("output_dir", "experiment_results"), exist_ok=True)
    cv_list = cfg.get("cv_strategies", ["stratified"]) 
    results = {}
    for cv in cv_list:
        args = [
            "--phenotype", cfg["phenotype"],
            "--nifti_dir", cfg["nifti_dir"],
            "--atlas", cfg["atlas"],
            "--atlas_type", cfg.get("atlas_type", "labels"),
            "--site_col", cfg.get("site_col", "SITE"),
            "--label_col", cfg.get("label_col", "DX_GROUP"),
            "--age_col", cfg.get("age_col", "AGE_AT_SCAN"),
            "--sex_col", cfg.get("sex_col", "SEX"),
            "--tr", str(cfg.get("tr", 2.0)),
            "--window_size", str(cfg.get("window_size", 50)),
            "--step", str(cfg.get("step", 10)),
            "--n_states", str(cfg.get("n_states", 5)),
            "--cv_strategy", cv,
            "--tune_models",
            "--output", os.path.join(cfg.get("output_dir", "experiment_results"), f"results_{cv}.json"),
        ]
        os.environ["TRAEPY_ARGS"] = " ".join(args)
        pipeline_main()
        with open(os.path.join(cfg.get("output_dir", "experiment_results"), f"results_{cv}.json")) as f:
            results[cv] = json.load(f)
    with open(os.path.join(cfg.get("output_dir", "experiment_results"), "summary.json"), "w") as f:
        json.dump(results, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run_config(args.config)


if __name__ == "__main__":
    main()
