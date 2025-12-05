import os
import numpy as np
import matplotlib.pyplot as plt


def save_histogram(values: np.ndarray, title: str, path: str):
    plt.figure(figsize=(6,4))
    plt.hist(values, bins=40, alpha=0.8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_bar(names: list, scores: list, title: str, path: str, top_k: int = 20):
    idx = np.argsort(np.abs(scores))[::-1][:top_k]
    sel_names = [names[i] for i in idx]
    sel_scores = [scores[i] for i in idx]
    plt.figure(figsize=(8,6))
    plt.barh(range(len(sel_scores)), sel_scores)
    plt.yticks(range(len(sel_scores)), sel_names)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_waterfall(values: np.ndarray, title: str, path: str):
    order = np.argsort(values)
    v = values[order]
    plt.figure(figsize=(8,4))
    plt.plot(v, marker=".")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
