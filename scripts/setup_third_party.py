import os
import sys
import argparse
import shutil
import subprocess


def install_pcntoolkit():
    try:
        import pcntoolkit  # noqa: F401
        return True
    except Exception:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pcntoolkit"])
            return True
        except Exception:
            return False


def ensure_bna_assets(src_dir: str, dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    src_atlas = os.path.join(src_dir, "atlas", "Full_brain_atlas_thr0-2mm", "fullbrain_atlas_thr0-2mm.nii.gz")
    src_labels = os.path.join(src_dir, "atlas", "BNA_subregions.xlsx")
    dst_atlas = os.path.join(dest_dir, "fullbrain_atlas_thr0-2mm.nii.gz")
    dst_labels = os.path.join(dest_dir, "BNA_subregions.xlsx")
    copied = []
    if os.path.exists(src_atlas):
        shutil.copy2(src_atlas, dst_atlas)
        copied.append(dst_atlas)
    if os.path.exists(src_labels):
        shutil.copy2(src_labels, dst_labels)
        copied.append(dst_labels)
    return copied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bna_src", type=str, default=os.path.expanduser("~/Downloads/Autism-Connectome-Analysis-master"))
    ap.add_argument("--dest", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "autism_connectome"))
    args = ap.parse_args()

    ok_pcn = install_pcntoolkit()
    copied = ensure_bna_assets(args.bna_src, args.dest)
    print({"pcntoolkit_installed": ok_pcn, "bna_copied": copied, "dest": args.dest})


if __name__ == "__main__":
    main()
