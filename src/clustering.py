"""
Stage 3 — Leiden clustering, UMAP embedding, and marker-based cell-type annotation.

Input:  data/processed/adata_preprocessed.h5ad
Output: data/processed/adata_clustered.h5ad
        figures/umap_*.png
"""
import argparse

import numpy as np
import pandas as pd
import scanpy as sc

from utils import FIGURES, MARKER_GENES, load_checkpoint, save_checkpoint

sc.settings.figdir = FIGURES
sc.settings.verbosity = 1


def cluster(adata, resolution=1.0, random_state=0):
    sc.tl.leiden(adata, resolution=resolution, random_state=random_state, key_added="leiden")
    sc.tl.umap(adata, random_state=random_state)
    print(f"[cluster] found {adata.obs['leiden'].nunique()} Leiden clusters at resolution={resolution}")
    return adata


def _safe_key(state: str) -> str:
    """h5ad/HDF5 keys cannot contain '/', which several of our state labels do
    (e.g. 'Naive/Stem-memory'). Use a sanitized key internally for obs columns."""
    return f"score_{state.replace('/', '_')}"


def score_cell_states(adata):
    """Score each cluster against curated marker sets and assign the best-matching
    biological label — a lightweight, transparent alternative to a reference-based
    classifier, appropriate for a focused CD8 T-cell dataset."""
    key_to_state = {}
    for state, genes in MARKER_GENES.items():
        present = [g for g in genes if g in adata.var_names]
        if present:
            key = _safe_key(state)
            sc.tl.score_genes(adata, present, score_name=key)
            key_to_state[key] = state

    score_cols = [k for k in key_to_state if k in adata.obs]
    cluster_scores = adata.obs.groupby("leiden")[score_cols].mean()
    best_key = cluster_scores.idxmax(axis=1)
    best_state = best_key.map(key_to_state)

    adata.obs["cell_type"] = adata.obs["leiden"].map(best_state).astype("category")
    print("[annotate] cluster -> cell type mapping:")
    print(best_state.to_string())
    return adata


def plot_results(adata):
    color_keys = [k for k in ["leiden", "cell_type", "patient", "source", "day_post_infusion"] if k in adata.obs]
    sc.pl.umap(adata, color=color_keys, save="_overview.png", show=False, wspace=0.4)

    present_markers = [g for genes in MARKER_GENES.values() for g in genes if g in adata.var_names]
    if present_markers:
        sc.pl.dotplot(
            adata, present_markers, groupby="leiden", standard_scale="var",
            save="_markers_by_cluster.png", show=False,
        )
    return adata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=float, default=1.0)
    args = parser.parse_args()

    adata = load_checkpoint("adata_preprocessed")
    adata = cluster(adata, resolution=args.resolution)
    adata = score_cell_states(adata)
    adata = plot_results(adata)

    save_checkpoint(adata, "adata_clustered")


if __name__ == "__main__":
    main()
