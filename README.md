# CAR-T Cell Single-Cell Immunotherapy Atlas

A reproducible **Scanpy**-based pipeline for analyzing single-cell RNA-seq (scRNA-seq)
data from CD19-directed CAR-T cell immunotherapy, covering QC, clustering, cell-type
annotation, differential gene expression, and pseudotime trajectory inference.

## Dataset

**GEO accession: [GSE125881](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE125881)**

> Sheih A, Voillet V, Hanafi LA, et al. *Clonal kinetics and single-cell
> transcriptional profiling of CAR-T cells in patients undergoing CD19 CAR-T
> immunotherapy.* **Nature Communications** 2020;11:219.
> [doi:10.1038/s41467-019-13880-1](https://doi.org/10.1038/s41467-019-13880-1)

The dataset contains 10x Genomics scRNA-seq profiles of **CD8+ CD19-CAR-T cells**
sorted from:
- **Infusion products (IP)** — the manufactured cell product before infusion
- **Post-infusion peripheral blood** — collected at multiple timepoints after
  patients received CAR-T therapy

This makes it well suited for exactly the three analysis goals of this project:
clustering distinct T-cell states (naive/memory, effector, exhausted, proliferating),
differential expression between infusion product vs. post-infusion / responders vs.
non-responders, and trajectory/pseudotime inference of T-cell differentiation and
exhaustion after adoptive transfer.

## Pipeline overview

```
raw GEO matrices
      │
      ▼
01  download_data.py        pull + assemble GSE125881 sample matrices from GEO
      │
      ▼
02  preprocessing.py        QC filtering, normalization, HVG selection, PCA, batch
      │                     integration (Harmony), neighbors graph
      ▼
03  clustering.py           Leiden clustering, UMAP, marker-gene based cell-type
      │                     annotation (naive/CM, effector, exhausted, proliferating…)
      ▼
04  differential_expression.py
      │                     rank_genes_groups (Wilcoxon) per cluster and per
      │                     contrast (IP vs. post-infusion), volcano + dotplots
      ▼
05  trajectory.py           diffusion pseudotime (DPT) + PAGA graph rooted in the
                             naive/stem-like cluster to model the differentiation /
                             exhaustion trajectory
```

Each stage reads/writes a single `AnnData` object (`data/processed/*.h5ad`), so
stages can be re-run independently once upstream artifacts exist.

## Project layout

```
car-t-immune-atlas/
├── README.md
├── environment.yml          # conda env (recommended)
├── requirements.txt         # pip alternative
├── run_pipeline.py          # orchestrates all 5 stages end to end
├── src/
│   ├── download_data.py
│   ├── preprocessing.py
│   ├── clustering.py
│   ├── differential_expression.py
│   ├── trajectory.py
│   └── utils.py
├── data/
│   ├── raw/                 # downloaded GEO files land here
│   └── processed/           # intermediate .h5ad checkpoints
├── figures/                 # all plots written here (UMAP, DE, trajectory)
├── tests/
│   └── test_pipeline_smoke.py   # synthetic-data smoke test (no download needed)
└── notebooks/
    └── 01_exploratory_analysis.ipynb
```

## Setup

```bash
git clone https://github.com/<your-username>/car-t-immune-atlas.git
cd car-t-immune-atlas

conda env create -f environment.yml
conda activate car-t-atlas
# or: pip install -r requirements.txt
```

## Running

```bash
# Full pipeline, stage by stage
python src/download_data.py
python src/preprocessing.py
python src/clustering.py
python src/differential_expression.py
python src/trajectory.py

# Or all at once
python run_pipeline.py
```

Outputs:
- `data/processed/adata_clustered.h5ad` — annotated AnnData ready for downstream use
- `figures/` — QC plots, UMAPs colored by cluster/cell type/patient/timepoint,
  DE volcano + dotplots, PAGA graph, diffusion pseudotime UMAP

## Quick smoke test (no internet / no download required)

The pipeline logic can be validated end-to-end on a small synthetic dataset that
mimics the real data's structure:

```bash
pytest tests/test_pipeline_smoke.py -v
```

This is useful for verifying your environment before committing to the full
~30k-cell real download.

## Biological questions this pipeline answers

1. **What T-cell states are present in a CAR-T product and after infusion?**
   (naive/stem-memory, central/effector memory, effector, exhausted, cycling)
2. **What genes distinguish infusion-product cells from cells that persisted
   and expanded post-infusion?** (differential expression, e.g. cytotoxicity /
   proliferation programs enriched in expanding clusters)
3. **What is the differentiation trajectory of a CAR-T cell after adoptive
   transfer**, from a stem/naive-like root toward effector and exhausted states?

## Notes on scaling to the full GEO dataset

`src/download_data.py` pulls the supplementary matrices GEO hosts for this series
and assembles a merged `AnnData` with per-cell metadata (patient, timepoint,
IP vs. blood) parsed from GEO sample titles. If GEO restructures file names,
adjust the parsing logic in `parse_sample_metadata()` in that script — GEO
supplementary file naming can change between series revisions.

## License

MIT — see `LICENSE`.
