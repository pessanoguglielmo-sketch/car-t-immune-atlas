#!/usr/bin/env python3
"""Run the full CAR-T scRNA-seq pipeline end to end.

Usage:
    python run_pipeline.py                # real GEO download
    python run_pipeline.py --synthetic     # synthetic data (offline / smoke test)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import clustering
import differential_expression
import download_data
import preprocessing
import trajectory
from utils import save_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--resolution", type=float, default=1.0, help="Leiden clustering resolution")
    args = parser.parse_args()

    print("=" * 70)
    print("STAGE 1/5 — download & assemble")
    print("=" * 70)
    adata = download_data.download_synthetic() if args.synthetic else download_data.download_real()
    save_checkpoint(adata, "adata_raw")

    print("=" * 70)
    print("STAGE 2/5 — QC, normalization, integration")
    print("=" * 70)
    adata = preprocessing.run_qc(adata)
    adata = preprocessing.normalize(adata)
    adata = preprocessing.select_hvgs_and_reduce(adata)
    if "patient" in adata.obs:
        adata = preprocessing.integrate_batches(adata)
    import scanpy as sc
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    save_checkpoint(adata, "adata_preprocessed")

    print("=" * 70)
    print("STAGE 3/5 — clustering & annotation")
    print("=" * 70)
    adata = clustering.cluster(adata, resolution=args.resolution)
    adata = clustering.score_cell_states(adata)
    adata = clustering.plot_results(adata)
    save_checkpoint(adata, "adata_clustered")

    print("=" * 70)
    print("STAGE 4/5 — differential expression")
    print("=" * 70)
    adata, _ = differential_expression.de_by_cluster(adata, groupby="cell_type")
    adata, _ = differential_expression.de_ip_vs_blood(adata)
    save_checkpoint(adata, "adata_de")

    print("=" * 70)
    print("STAGE 5/5 — trajectory / pseudotime")
    print("=" * 70)
    adata = trajectory.run_paga(adata, groupby="cell_type")
    adata, root_label = trajectory.run_dpt(adata, groupby="cell_type")
    trajectory.summarize_trajectory(adata, groupby="cell_type")
    save_checkpoint(adata, "adata_trajectory")

    print("\nPipeline complete. Final AnnData: data/processed/adata_trajectory.h5ad")
    print("Figures written to: figures/")


if __name__ == "__main__":
    main()
