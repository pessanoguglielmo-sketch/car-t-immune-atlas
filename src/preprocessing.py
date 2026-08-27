"""
Stage 2 — QC filtering, normalization, HVG selection, PCA, batch integration.

Input:  data/processed/adata_raw.h5ad   (from download_data.py)
Output: data/processed/adata_preprocessed.h5ad
"""
import argparse

import scanpy as sc

from utils import FIGURES, load_checkpoint, qc_summary, save_checkpoint

sc.settings.figdir = FIGURES
sc.settings.verbosity = 1


def run_qc(adata, min_genes=200, min_cells=3, max_pct_mt=15.0, max_genes=6000):
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    qc_summary(adata, "(before filtering)")

    sc.pl.violin(
        adata,
        ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        jitter=0.4,
        multi_panel=True,
        save="_qc_prefilter.png",
        show=False,
    )

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    adata = adata[adata.obs.n_genes_by_counts < max_genes, :].copy()
    if "pct_counts_mt" in adata.obs:
        adata = adata[adata.obs.pct_counts_mt < max_pct_mt, :].copy()

    qc_summary(adata, "(after filtering)")
    return adata


def normalize(adata):
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata  # keep full log-normalized matrix for DE / plotting later
    return adata


def select_hvgs_and_reduce(adata, n_top_genes=2000, batch_key="patient"):
    sc.pp.highly_variable_genes(
        adata, n_top_genes=n_top_genes, flavor="seurat_v3", layer="counts", batch_key=batch_key
    )
    adata_hvg = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, n_comps=30, svd_solver="arpack")

    # Copy PCA back onto the full-gene AnnData so downstream steps have all genes
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
    adata.uns["pca"] = adata_hvg.uns["pca"]
    adata.varm = adata_hvg.varm if adata_hvg.n_vars == adata.n_vars else adata.varm
    return adata


def integrate_batches(adata, batch_key="patient"):
    """Harmony integration across patients so clusters reflect T-cell state,
    not patient-of-origin."""
    try:
        import scanpy.external as sce

        sce.pp.harmony_integrate(adata, batch_key)
        adata.obsm["X_pca_uncorrected"] = adata.obsm["X_pca"]
        adata.obsm["X_pca"] = adata.obsm["X_pca_harmony"]
        print("[integrate] Harmony batch correction applied on:", batch_key)
    except Exception as e:
        print(f"[integrate] Harmony unavailable or failed ({e}); continuing without it.")
    return adata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-genes", type=int, default=200)
    parser.add_argument("--max-pct-mt", type=float, default=15.0)
    parser.add_argument("--n-hvgs", type=int, default=2000)
    parser.add_argument("--no-integration", action="store_true")
    args = parser.parse_args()

    adata = load_checkpoint("adata_raw")

    adata = run_qc(adata, min_genes=args.min_genes, max_pct_mt=args.max_pct_mt)
    adata = normalize(adata)
    adata = select_hvgs_and_reduce(adata, n_top_genes=args.n_hvgs)

    if not args.no_integration and "patient" in adata.obs:
        adata = integrate_batches(adata)

    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)

    save_checkpoint(adata, "adata_preprocessed")


if __name__ == "__main__":
    main()
