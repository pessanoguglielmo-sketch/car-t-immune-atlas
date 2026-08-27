"""
Stage 4 — Differential gene expression.

Two kinds of contrasts, both standard for a CAR-T immunotherapy dataset:
  (a) marker DE:      each cluster / cell type vs. rest (Wilcoxon rank-sum)
  (b) biological DE:  Infusion Product vs. Post-infusion blood, pooling all
                       cells, to find genes associated with in vivo persistence
                       and activation after adoptive transfer.

Input:  data/processed/adata_clustered.h5ad
Output: data/processed/adata_de.h5ad
        figures/de_*.png
        data/processed/de_cluster_markers.csv
        data/processed/de_ip_vs_blood.csv
"""
import argparse

import pandas as pd
import scanpy as sc

from utils import DATA_PROCESSED, FIGURES, load_checkpoint, save_checkpoint

sc.settings.figdir = FIGURES
sc.settings.verbosity = 1


def de_by_cluster(adata, groupby="cell_type", method="wilcoxon"):
    sc.tl.rank_genes_groups(adata, groupby=groupby, method=method, key_added=f"rank_{groupby}")

    sc.pl.rank_genes_groups_dotplot(
        adata, key=f"rank_{groupby}", groupby=groupby, n_genes=5,
        save=f"_top_markers_{groupby}.png", show=False,
    )

    result = adata.uns[f"rank_{groupby}"]
    groups = result["names"].dtype.names
    rows = []
    for g in groups:
        for i in range(len(result["names"][g])):
            rows.append(
                {
                    "group": g,
                    "gene": result["names"][g][i],
                    "logfoldchange": result["logfoldchanges"][g][i],
                    "pval": result["pvals"][g][i],
                    "pval_adj": result["pvals_adj"][g][i],
                    "score": result["scores"][g][i],
                }
            )
    df = pd.DataFrame(rows)
    out = DATA_PROCESSED / f"de_{groupby}_markers.csv"
    df.to_csv(out, index=False)
    print(f"[DE] wrote {out} ({len(df)} rows)")
    return adata, df


def de_ip_vs_blood(adata, method="wilcoxon"):
    if "source" not in adata.obs or adata.obs["source"].nunique() < 2:
        print("[DE] 'source' column with >=2 groups not found — skipping IP-vs-blood contrast.")
        return adata, None

    sc.tl.rank_genes_groups(
        adata, groupby="source", groups=["Infusion Product"], reference="Post-infusion blood",
        method=method, key_added="rank_source",
    )
    result = adata.uns["rank_source"]
    df = pd.DataFrame(
        {
            "gene": result["names"]["Infusion Product"],
            "logfoldchange": result["logfoldchanges"]["Infusion Product"],
            "pval": result["pvals"]["Infusion Product"],
            "pval_adj": result["pvals_adj"]["Infusion Product"],
            "score": result["scores"]["Infusion Product"],
        }
    )
    out = DATA_PROCESSED / "de_ip_vs_blood.csv"
    df.to_csv(out, index=False)
    print(f"[DE] wrote {out} ({len(df)} rows)")

    sc.pl.rank_genes_groups(adata, key="rank_source", save="_ip_vs_blood.png", show=False)

    # simple volcano
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 5))
    neglog10p = -np.log10(df["pval_adj"].clip(lower=1e-300))
    ax.scatter(df["logfoldchange"], neglog10p, s=6, alpha=0.5, c="steelblue")
    sig = (df["pval_adj"] < 0.05) & (df["logfoldchange"].abs() > 1)
    ax.scatter(df.loc[sig, "logfoldchange"], neglog10p[sig], s=8, c="crimson")
    ax.set_xlabel("log2 fold change (Infusion Product vs. Post-infusion blood)")
    ax.set_ylabel("-log10 adjusted p-value")
    ax.set_title("IP vs. post-infusion blood — differential expression")
    fig.tight_layout()
    fig.savefig(FIGURES / "de_volcano_ip_vs_blood.png", dpi=150)
    plt.close(fig)

    return adata, df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groupby", default="cell_type", choices=["cell_type", "leiden"])
    args = parser.parse_args()

    adata = load_checkpoint("adata_clustered")
    adata, _ = de_by_cluster(adata, groupby=args.groupby)
    adata, _ = de_ip_vs_blood(adata)

    save_checkpoint(adata, "adata_de")


if __name__ == "__main__":
    main()
