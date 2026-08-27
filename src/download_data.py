"""
Stage 1 — Download & assemble GSE125881 (CD19 CAR-T scRNA-seq, Sheih et al. 2020).

Real download:
    python src/download_data.py
Synthetic (for offline dev / CI / smoke tests, mimics the same structure):
    python src/download_data.py --synthetic

GEO does not provide a single count matrix for this series — it hosts one
supplementary file set per GEO Sample (GSM), one GSM per patient/timepoint.
This script:
  1. Queries the series (via GEOparse) for its GSM list and titles.
  2. Downloads each GSM's supplementary 10x-style matrix (or raw counts table).
  3. Parses patient / timepoint / IP-vs-blood metadata out of each GSM title.
  4. Concatenates everything into one AnnData with rich `.obs` metadata and
     writes data/processed/adata_raw.h5ad.

NOTE: GEO occasionally changes supplementary file naming between series
revisions. If GEOparse's file list doesn't match the patterns below, inspect
the series page (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE125881)
and adjust `parse_sample_metadata()` / `read_supplementary_matrix()`.
"""
import argparse
import re
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from utils import DATA_RAW, GEO_ACCESSION, save_checkpoint

# Typical GSM title patterns for this series look like:
#   "IP_Pt12", "Pt12_d10_blood", "Pt7_IP", "Pt3_d90"
# i.e. patient id, whether it's the Infusion Product or a post-infusion blood
# draw, and (for blood draws) the day post-infusion.
TITLE_PATTERN = re.compile(
    r"(?P<patient>Pt\d+).*?(?:(?P<ip>IP)|d(?P<day>\d+))", re.IGNORECASE
)


def parse_sample_metadata(title: str) -> dict:
    """Extract patient / timepoint / sample-source metadata from a GSM title."""
    m = TITLE_PATTERN.search(title)
    if not m:
        return {"patient": "unknown", "source": "unknown", "day_post_infusion": np.nan}
    patient = m.group("patient")
    if m.group("ip"):
        return {"patient": patient, "source": "Infusion Product", "day_post_infusion": 0}
    day = int(m.group("day")) if m.group("day") else np.nan
    return {"patient": patient, "source": "Post-infusion blood", "day_post_infusion": day}


def download_real():
    try:
        import GEOparse
    except ImportError:
        sys.exit(
            "GEOparse is required for the real download. Install with:\n"
            "  pip install GEOparse"
        )

    print(f"Fetching series metadata for {GEO_ACCESSION} ...")
    gse = GEOparse.get_GEO(geo=GEO_ACCESSION, destdir=str(DATA_RAW))

    adatas = []
    for gsm_name, gsm in gse.gsms.items():
        title = gsm.metadata.get("title", [gsm_name])[0]
        print(f"  downloading {gsm_name} ({title}) ...")
        gsm.download_supplementary_files(directory=str(DATA_RAW / gsm_name))

        # Each GSM's supplementary files typically include a barcodes/features/
        # matrix triplet or a dense counts table — adjust to whichever GEO
        # actually ships for a given GSM.
        supp_dir = DATA_RAW / gsm_name
        counts_files = list(supp_dir.glob("*matrix*")) + list(supp_dir.glob("*counts*"))
        if not counts_files:
            print(f"    !! no recognizable count matrix for {gsm_name}, skipping")
            continue

        try:
            a = sc.read_10x_mtx(supp_dir) if any(
                f.name.endswith(".mtx") or f.name.endswith(".mtx.gz") for f in supp_dir.iterdir()
            ) else sc.read_csv(counts_files[0]).T
        except Exception as e:
            print(f"    !! failed to parse {gsm_name}: {e}")
            continue

        meta = parse_sample_metadata(title)
        for k, v in meta.items():
            a.obs[k] = v
        a.obs["gsm"] = gsm_name
        a.obs_names = [f"{gsm_name}_{bc}" for bc in a.obs_names]
        a.var_names_make_unique()
        adatas.append(a)

    if not adatas:
        sys.exit("No samples were successfully parsed — check GEO file structure.")

    adata = ad.concat(adatas, join="outer", index_unique=None)
    adata.var_names_make_unique()
    return adata


def download_synthetic(n_patients=6, cells_per_sample=400, n_genes=2000, seed=0):
    """
    Generate a synthetic dataset with the same *shape* of metadata as
    GSE125881 (patients x {IP, several post-infusion timepoints}) and gene
    expression structured around the marker genes used downstream, so the
    full pipeline (clustering / DE / trajectory) can be exercised without
    network access.
    """
    from utils import MARKER_GENES

    rng = np.random.default_rng(seed)
    marker_genes_flat = sorted({g for gs in MARKER_GENES.values() for g in gs})
    background_genes = [f"GENE{i}" for i in range(n_genes - len(marker_genes_flat))]
    genes = marker_genes_flat + background_genes

    states = list(MARKER_GENES.keys())
    timepoints = ["Infusion Product", "d10", "d30", "d90"]

    obs_rows = []
    X_rows = []
    for p in range(1, n_patients + 1):
        patient = f"Pt{p}"
        for tp in timepoints:
            source = "Infusion Product" if tp == "Infusion Product" else "Post-infusion blood"
            day = 0 if tp == "Infusion Product" else int(tp[1:])
            # differentiation drifts toward effector/exhausted over time
            state_weights = np.array(
                [
                    max(0.35 - 0.003 * day, 0.03),  # Naive/Stem-memory shrinks
                    0.15,
                    0.20,
                    0.15 + 0.001 * day,             # Effector/Cytotoxic grows slightly
                    0.10 + 0.004 * day,              # Exhausted grows with time
                    max(0.15 - 0.001 * day, 0.02),   # Proliferating shrinks
                    0.05,
                ]
            )
            state_weights = state_weights / state_weights.sum()

            for _ in range(cells_per_sample):
                cell_state = rng.choice(states, p=state_weights)
                counts = rng.negative_binomial(3, 0.4, size=len(genes)).astype(float)
                for g in MARKER_GENES[cell_state]:
                    idx = genes.index(g)
                    counts[idx] += rng.poisson(15)
                X_rows.append(counts)
                obs_rows.append(
                    {
                        "patient": patient,
                        "source": source,
                        "day_post_infusion": day,
                        "ground_truth_state": cell_state,
                        "gsm": f"SYN_{patient}_{tp}",
                    }
                )

    X = np.vstack(X_rows)
    obs = pd.DataFrame(obs_rows)
    obs.index = [f"cell_{i}" for i in range(len(obs))]
    var = pd.DataFrame(index=genes)
    adata = ad.AnnData(X=X, obs=obs, var=var)
    print(
        f"[synthetic] built {adata.n_obs} cells x {adata.n_vars} genes across "
        f"{n_patients} patients x {len(timepoints)} timepoints"
    )
    return adata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate a structurally-matched synthetic dataset instead of downloading from GEO.",
    )
    args = parser.parse_args()

    adata = download_synthetic() if args.synthetic else download_real()

    print(adata)
    save_checkpoint(adata, "adata_raw")


if __name__ == "__main__":
    main()
