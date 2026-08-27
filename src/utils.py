"""Shared utilities for the CAR-T scRNA-seq pipeline."""
from pathlib import Path
import scanpy as sc

# ---- Project paths -------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"

for _p in (DATA_RAW, DATA_PROCESSED, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# ---- GEO series info -------------------------------------------------------
GEO_ACCESSION = "GSE125881"
GEO_FTP_SUPPL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE125nnn/"
    f"{GEO_ACCESSION}/suppl/"
)

# ---- Canonical marker genes for CAR-T / CD8 T-cell state annotation -------
# Used by clustering.py to label Leiden clusters biologically.
MARKER_GENES = {
    "Naive/Stem-memory": ["CCR7", "SELL", "TCF7", "LEF1", "IL7R", "CD27", "CD28"],
    "Central memory": ["CCR7", "SELL", "IL7R", "CD27", "GZMK"],
    "Effector memory": ["GZMK", "GZMA", "CCL5", "KLRG1", "CX3CR1"],
    "Effector/Cytotoxic": ["GZMB", "PRF1", "GNLY", "NKG7", "FGFBP2", "FCGR3A"],
    "Exhausted": ["PDCD1", "HAVCR2", "LAG3", "TIGIT", "TOX", "CTLA4", "ENTPD1"],
    "Proliferating": ["MKI67", "TOP2A", "PCNA", "TYMS"],
    "Regulatory-like": ["FOXP3", "IL2RA", "IKZF2"],
}


def qc_summary(adata, label=""):
    """Print a quick QC summary for an AnnData object."""
    print(f"--- QC summary {label} ---")
    print(f"cells: {adata.n_obs}, genes: {adata.n_vars}")
    if "pct_counts_mt" in adata.obs:
        print(f"median %MT: {adata.obs['pct_counts_mt'].median():.2f}")
    if "n_genes_by_counts" in adata.obs:
        print(f"median genes/cell: {adata.obs['n_genes_by_counts'].median():.0f}")


def save_checkpoint(adata, name):
    path = DATA_PROCESSED / f"{name}.h5ad"
    adata.write_h5ad(path)
    print(f"[checkpoint] wrote {path}")
    return path


def load_checkpoint(name):
    path = DATA_PROCESSED / f"{name}.h5ad"
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint {path} not found. Run the previous pipeline stage first."
        )
    return sc.read_h5ad(path)
