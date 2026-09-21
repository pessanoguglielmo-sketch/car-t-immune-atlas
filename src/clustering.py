"""
Stage 3 — Leiden clustering, UMAP embedding, and marker-based cell-type annotation.

Input:  data/processed/adata_preprocessed.h5ad
Output: data/processed/adata_clustered.h5ad
        data/processed/cluster_signature_zscores.csv
        figures/umap_*.png
"""
import argparse

import numpy as np
import pandas as pd
import scanpy as sc

from utils import DATA_PROCESSED, FIGURES, MARKER_GENES, load_checkpoint, save_checkpoint

sc.settings.figdir = FIGURES
sc.settings.verbosity = 1


def cluster(adata, resolution=1.0, random_state=0):
    # Leiden = community detection on the neighbor graph; higher resolution -> more clusters.
    sc.tl.leiden(adata, resolution=resolution, random_state=random_state, key_added="leiden")
    # UMAP = 2D projection of the same neighbor graph, for plotting.
    sc.tl.umap(adata, random_state=random_state)
    print(f"[cluster] found {adata.obs['leiden'].nunique()} Leiden clusters at resolution={resolution}")
    return adata


def _safe_key(state: str) -> str:
    """h5ad/HDF5 keys cannot contain '/', which several of our state labels do
    (e.g. 'Naive/Stem-memory'). Use a sanitized key internally for obs columns."""
    return f"score_{state.replace('/', '_')}"


def score_cell_states(adata):
    """Score every cell against each curated marker set, average the scores per
    Leiden cluster, then label each cluster with the state whose signature is
    highest *relative to the other clusters*.

    Raw scores are not comparable between signatures (e.g. cytotoxic genes are
    expressed far more strongly than naive-state genes), so taking the raw argmax
    labels almost everything as the strongest signature. Z-scoring each signature
    across clusters first puts all signatures on the same scale.
    """
    key_to_state = {}
    for state, genes in MARKER_GENES.items():
        present = [g for g in genes if g in adata.var_names]
        missing = [g for g in genes if g not in adata.var_names]
        if missing:
            print(f"[annotate] {state}: genes not in dataset: {missing}")
        if present:
            key = _safe_key(state)
            sc.tl.score_genes(adata, present, score_name=key)
            key_to_state[key] = state

    score_cols = [k for k in key_to_state if k in adata.obs]
    cluster_scores = adata.obs.groupby("leiden", observed=True)[score_cols].mean()

    # z-score each signature across clusters (columns), then compare within a cluster
    std = cluster_scores.std(axis=0, ddof=0).replace(0, 1.0)
    z = (cluster_scores - cluster_scores.mean(axis=0)) / std
    z.columns = [key_to_state[c] for c in z.columns]

    best_state = z.idxmax(axis=1)
    adata.obs["cell_type"] = adata.obs["leiden"].map(best_state).astype("category")

    z.round(2).to_csv(DATA_PROCESSED / "cluster_signature_zscores.csv")
    print("[annotate] signature z-scores per cluster (rows=cluster):")
    print(z.round(2).to_string())
    print("[annotate] cluster -> cell type mapping:")
    print(best_state.to_string())
    print("[annotate] cells per cell type:")
    print(adata.obs["cell_type"].value_counts().to_string())
    return adata


def plot_results(adata):
    color_keys = [k for k in ["leiden", "cell_type", "patient", "source", "day_post_infusion"] if k in adata.obs]
    sc.pl.umap(adata, color=color_keys, save="_overview.png", show=False, wspace=0.4)

    present_markers = []
    for genes in MARKER_GENES.values():
        for g in genes:
            if g in adata.var_names and g not in present_markers:
                present_markers.append(g)
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