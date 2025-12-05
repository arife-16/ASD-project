import os
import sys
import argparse
import shutil
import subprocess
import requests


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


def download_if_url(name: str, url: str, dest_path: str):
    if not url:
        return False
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bna_src", type=str, default=os.path.expanduser("~/Downloads/Autism-Connectome-Analysis-master"))
    ap.add_argument("--dest", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "autism_connectome"))
    args = ap.parse_args()

    ok_pcn = install_pcntoolkit()
    copied = ensure_bna_assets(args.bna_src, args.dest)
    atlas_url = os.environ.get("BNA_ATLAS_URL", "")
    labels_url = os.environ.get("BNA_LABELS_URL", "")
    atlas_path = os.path.join(args.dest, "fullbrain_atlas_thr0-2mm.nii.gz")
    labels_path = os.path.join(args.dest, "BNA_subregions.xlsx")
    if not os.path.exists(atlas_path) and atlas_url:
        if download_if_url("atlas", atlas_url, atlas_path):
            copied.append(atlas_path)
    if not os.path.exists(labels_path) and labels_url:
        if download_if_url("labels", labels_url, labels_path):
            copied.append(labels_path)
    print({
        "pcntoolkit_installed": ok_pcn,
        "bna_copied": copied,
        "dest": args.dest,
        "atlas_present": os.path.exists(atlas_path),
        "labels_present": os.path.exists(labels_path),
    })


if __name__ == "__main__":
    main()
