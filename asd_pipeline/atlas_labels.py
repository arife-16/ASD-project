import os
import pandas as pd
from typing import List, Optional, Tuple


def load_bna_labels(xlsx_path: str) -> Tuple[List[str], Optional[List[str]]]:
    if not xlsx_path or not os.path.exists(xlsx_path):
        return [], None
    df = pd.read_excel(xlsx_path)
    cols = df.columns.str.lower()
    name_col = None
    net_col = None
    for c in df.columns:
        cl = c.lower()
        if "name" in cl or "roi" in cl:
            name_col = c if name_col is None else name_col
        if "network" in cl or "module" in cl:
            net_col = c
    if name_col is None:
        name_col = df.columns[0]
    names = df[name_col].astype(str).tolist()
    networks = df[net_col].astype(str).tolist() if net_col else None
    return names, networks


def load_cc_labels(tsv_path: str) -> Tuple[List[str], Optional[List[str]]]:
    if not tsv_path or not os.path.exists(tsv_path):
        return [], None
    df = pd.read_csv(tsv_path, sep="\t")
    if "label" in df.columns:
        labels = df["label"].astype(str).tolist()
    else:
        col = df.columns[1]
        labels = df[col].astype(str).tolist()
    if len(labels) > 0 and labels[0].lower() == "background":
        labels = labels[1:]
    return labels, None
