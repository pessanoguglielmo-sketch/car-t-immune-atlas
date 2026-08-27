"""
End-to-end smoke test on synthetic data — validates the whole pipeline
(download -> preprocessing -> clustering -> DE -> trajectory) runs without
error and produces sane outputs. Does not require network access.

Run with:
    pytest tests/test_pipeline_smoke.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import clustering
import differential_expression
import download_data
import preprocessing
import scanpy as sc
import trajectory


@pytest.fixture(scope="module")
def raw_adata():
    return download_data.download_synthetic(n_patients=3, cells_per_sample=150, n_genes=500, seed=1)


@pytest.fixture(scope="module")
def preprocessed_adata(raw_adata):
    adata = preprocessing.run_qc(raw_adata.copy(), min_genes=5, max_pct_mt=100.0, max_genes=100000)
    adata = preprocessing.normalize(adata)
    adata = preprocessing.select_hvgs_and_reduce(adata, n_top_genes=200)
    adata = preprocessing.integrate_batches(adata)
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=min(20, adata.obsm["X_pca"].shape[1]))
    return adata


@pytest.fixture(scope="module")
def clustered_adata(preprocessed_adata):
    adata = clustering.cluster(preprocessed_adata.copy(), resolution=1.0)
    adata = clustering.score_cell_states(adata)
    return adata


def test_download_synthetic_shape(raw_adata):
    assert raw_adata.n_obs > 0
    assert raw_adata.n_vars > 0
    assert {"patient", "source", "day_post_infusion"}.issubset(raw_adata.obs.columns)


def test_preprocessing_produces_pca_and_neighbors(preprocessed_adata):
    assert "X_pca" in preprocessed_adata.obsm
    assert "neighbors" in preprocessed_adata.uns


def test_clustering_assigns_clusters_and_cell_types(clustered_adata):
    assert "leiden" in clustered_adata.obs
    assert clustered_adata.obs["leiden"].nunique() >= 2
    assert "cell_type" in clustered_adata.obs
    assert clustered_adata.obs["cell_type"].notna().all()


def test_differential_expression_runs(clustered_adata):
    adata, df = differential_expression.de_by_cluster(clustered_adata.copy(), groupby="cell_type")
    assert df is not None and len(df) > 0
    assert {"gene", "logfoldchange", "pval_adj"}.issubset(df.columns)

    adata, df_ip = differential_expression.de_ip_vs_blood(adata)
    assert df_ip is not None and len(df_ip) > 0


def test_trajectory_runs_and_assigns_pseudotime(clustered_adata):
    adata, df = differential_expression.de_by_cluster(clustered_adata.copy(), groupby="cell_type")
    adata = trajectory.run_paga(adata, groupby="cell_type")
    adata, root_label = trajectory.run_dpt(adata, groupby="cell_type")
    assert "dpt_pseudotime" in adata.obs
    assert adata.obs["dpt_pseudotime"].between(0, 1).all()
    assert root_label is not None
