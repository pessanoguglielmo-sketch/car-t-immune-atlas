"""
Stage 5 — Trajectory / cell-state inference.

Models CAR-T cell differentiation as a trajectory rooted in the
Naive/Stem-memory cluster and progressing toward Effector/Cytotoxic and
Exhausted states, using:
  - PAGA (partition-based graph abstraction) for the coarse cluster graph
  - Diffusion pseudotime (DPT) for a continuous per-cell ordering

Input:  data/processed/adata_de.h5ad
Output: data/processed/adata_trajectory.h5ad
        figures/paga_*.png, figures/dpt_*.png
"""
import argparse

import numpy as np
import scanpy as sc

from utils import DATA_PROCESSED, FIGURES, load_checkpoint, save_checkpoint

sc.settings.figdir = FIGURES
sc.settings.verbosity = 1

ROOT_STATE_CANDIDATES = ["Naive/Stem-memory", "Central memory"]


def pick_root_cell(adata, groupby="cell_type"):
    """Pick a root cell for DPT: the cell closest to the centroid of the
    most naive/stem-like available cluster."""
    root_label = next((s for s in ROOT_STATE_CANDIDATES if s in adata.obs[groupby].unique()), None)
    if root_label is None:
        root_label = adata.obs[groupby].value_counts().idxmax()
        print(f"[trajectory] no naive/stem cluster found; defaulting root to largest cluster '{root_label}'")
    else:
        print(f"[trajectory] rooting pseudotime in '{root_label}'")

    mask = (adata.obs[groupby] == root_label).values
    pca = adata.obsm["X_pca"]
    centroid = pca[mask].mean(axis=0)
    dists = np.linalg.norm(pca[mask] - centroid, axis=1)
    root_idx_local = np.argmin(dists)
    root_idx_global = np.where(mask)[0][root_idx_local]
    return root_idx_global, root_label


def run_paga(adata, groupby="cell_type"):
    sc.tl.paga(adata, groups=groupby)
    from scipy.sparse import csr_matrix
    for k in ("connectivities", "connectivities_tree"): 
        if k in adata.uns["paga"]:
            adata.uns["paga"][k] = csr_matrix(adata.uns["paga"][k])
    sc.pl.paga(adata, color=[groupby], save="_celltype_graph.png", show=False, threshold=0.1)
    return adata


def run_dpt(adata, groupby="cell_type"):
    root_idx, root_label = pick_root_cell(adata, groupby=groupby)
    adata.uns["iroot"] = int(root_idx)

    sc.tl.diffmap(adata)
    sc.tl.dpt(adata)

    color_keys = [k for k in ["dpt_pseudotime", groupby, "day_post_infusion"] if k in adata.obs]
    sc.pl.umap(adata, color=color_keys, save="_pseudotime.png", show=False, wspace=0.4)

    # PAGA-initialized UMAP layout often gives a cleaner trajectory picture
    sc.tl.umap(adata, init_pos="paga")
    sc.pl.umap(adata, color=color_keys, save="_pseudotime_paga_init.png", show=False, wspace=0.4)

    return adata, root_label


def summarize_trajectory(adata, groupby="cell_type"):
    summary = (
        adata.obs.groupby(groupby)["dpt_pseudotime"]
        .agg(["mean", "median", "std", "count"])
        .sort_values("mean")
    )
    print("[trajectory] mean pseudotime by cell state (ordered root -> terminal):")
    print(summary.to_string())
    summary.to_csv(DATA_PROCESSED / "dpt_summary_by_state.csv")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groupby", default="cell_type", choices=["cell_type", "leiden"])
    args = parser.parse_args()

    adata = load_checkpoint("adata_de")
    adata = run_paga(adata, groupby=args.groupby)
    adata, root_label = run_dpt(adata, groupby=args.groupby)
    summarize_trajectory(adata, groupby=args.groupby)

    save_checkpoint(adata, "adata_trajectory")
    print(f"[trajectory] done. Root state: {root_label}")


if __name__ == "__main__":
    main()
