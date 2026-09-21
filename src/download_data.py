"""
Stage 1 — Download & assemble GSE125881 (CD19 CAR-T scRNA-seq, Sheih et al. 2020).

Real download:
    python src/download_data.py
Synthetic (for offline dev / CI / smoke tests, mimics the same structure):
    python src/download_data.py --synthetic

How GEO hosts this series
-------------------------
The per-sample (GSM) records only carry TCR clonotype tables (the "VDJ"
samples); the "GE" samples have no files attached. The expression data lives
at the SERIES level in one combined dense CSV:

    GSE125881_raw.expMatrix.csv.gz     genes (rows) x cells (columns), ~62k cells
    GSE125881_RAW.tar                  per-sample VDJ files (not needed here)

Cell barcodes in that matrix carry no sample label, so this script assigns each
cell to a patient/timepoint using the barcode suffix (-1, -2, ... from Cell
Ranger aggregation) and validates the suffix -> sample mapping against the
barcodes in the per-sample VDJ clonotype files when those contain barcodes.

The matrix is a dense CSV, so it is streamed in row chunks straight into a
sparse matrix (a full pandas load would need many GB of RAM).
"""
import argparse
import re
import sys
import urllib.request

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from utils import DATA_RAW, GEO_ACCESSION, save_checkpoint

MATRIX_NAME = "GSE125881_raw.expMatrix.csv.gz"
MATRIX_URLS = [
    f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE125nnn/{GEO_ACCESSION}/suppl/{MATRIX_NAME}",
    f"ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE125nnn/{GEO_ACCESSION}/suppl/{MATRIX_NAME}",
]

# GSM titles in this series look like:
#   "CLL-1 - IP - GE", "CLL-1 - d21 - GE", "NHL-6 - d102 - VDJ"
TITLE_PATTERN = re.compile(
    r"^\s*(?P<patient>[A-Za-z]+-\d+)\s*-\s*(?P<tp>IP|d\d+)\s*-\s*(?P<assay>GE|VDJ)\s*$",
    re.IGNORECASE,
)


def parse_sample_metadata(title: str):
    """Parse patient / timepoint / assay out of a GSM title. Returns None if no match."""
    m = TITLE_PATTERN.match(title)
    if not m:
        return None
    patient = m.group("patient").upper()
    tp = m.group("tp")
    is_ip = tp.upper() == "IP"
    return {
        "patient": patient,
        "disease": patient.split("-")[0],
        "timepoint": "IP" if is_ip else tp.lower(),
        "source": "Infusion Product" if is_ip else "Post-infusion blood",
        "day_post_infusion": 0 if is_ip else int(tp[1:]),
        "assay": m.group("assay").upper(),
    }


def fetch_matrix():
    """Download the series-level expression matrix into data/raw (skipped if present)."""
    dest = DATA_RAW / MATRIX_NAME
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[matrix] using existing {dest}")
        return dest
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for url in MATRIX_URLS:
        try:
            print(f"[matrix] downloading {url} (this can take a few minutes) ...")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(dest)
            return dest
        except Exception as e:  # noqa: BLE001
            print(f"    !! failed: {e}")
    sys.exit(
        f"Could not download {MATRIX_NAME}. Download it manually in a browser from\n"
        f"  {MATRIX_URLS[0]}\nand save it to {dest}, then re-run."
    )


def load_matrix_sparse(path, chunksize=500):
    """Stream a dense genes x cells CSV(.gz) into a sparse cells x genes matrix."""
    header = pd.read_csv(path, nrows=0, index_col=0)
    cells = list(header.columns)
    if len(set(cells)) != len(cells):
        sys.exit("Cell barcodes in the matrix are not unique; cannot assign samples safely.")
    dtype = {c: np.float32 for c in cells}

    genes, blocks = [], []
    reader = pd.read_csv(path, index_col=0, dtype=dtype, chunksize=chunksize)
    for i, chunk in enumerate(reader):
        genes.extend(chunk.index.astype(str))
        blocks.append(sparse.csr_matrix(chunk.to_numpy()))
        if i % 10 == 0:
            print(f"    ... {len(genes)} genes read")
    print(f"[matrix] {len(genes)} genes x {len(cells)} cells")
    X = sparse.vstack(blocks, format="csr").T.tocsr()
    return X, cells, genes


def _split_barcode(name: str):
    """'AAACCTGAGCGTCTAT-3' -> ('AAACCTGAGCGTCTAT', 3). Suffix is None if absent."""
    m = re.match(r"^([ACGT]+)(?:-(\d+))?$", name)
    if not m:
        return name, None
    return m.group(1), (int(m.group(2)) if m.group(2) else None)


def read_vdj_barcodes(path):
    """Return the set of 16-mer barcodes in a VDJ file, or an empty set if it has none."""
    try:
        df = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return set()
    candidates = [df[c] for c in df.columns] + [df.index.to_series()]
    for col in candidates:
        seqs = col.astype(str).str.extract(r"^([ACGT]{16})", expand=False)
        if seqs.notna().mean() > 0.8:
            return set(seqs.dropna())
    return set()


def assign_samples(cells, ge_samples, vdj_barcodes):
    """
    Map each barcode suffix to a GE sample key.

    ge_samples: ordered list of (gsm_name, meta) for the GE samples
    vdj_barcodes: {(patient, timepoint): set of barcode sequences}
    Returns {suffix: (gsm_name, meta)}.
    """
    groups = {}
    for c in cells:
        seq, suf = _split_barcode(c)
        groups.setdefault(suf, set()).add(seq)

    if None in groups or len(groups) == 1:
        sys.exit(
            "Barcodes have no per-library suffix (-1, -2, ...), so cells cannot be "
            "assigned to samples from the matrix alone. Paste the barcode suffix check "
            "output and we will find another way to map cells to samples."
        )

    suffixes = sorted(groups)
    by_key = {(m["patient"], m["timepoint"]): (g, m) for g, m in ge_samples}
    mapping = {}

    # 1) validate/derive the mapping from VDJ barcodes when available
    usable = {k: v for k, v in vdj_barcodes.items() if v and k in by_key}
    if usable:
        print("[assign] suffix -> sample, by overlap with VDJ barcodes:")
        for s in suffixes:
            scores = {k: len(groups[s] & v) / len(v) for k, v in usable.items()}
            best = max(scores, key=scores.get)
            frac = scores[best]
            print(f"    -{s:<3} {len(groups[s]):>6} cells  best={best[0]} {best[1]:<5} overlap={frac:.2f}")
            if frac >= 0.3:
                mapping[s] = by_key[best]
        resolved_keys = [v[0] for v in mapping.values()]
        if len(mapping) == len(suffixes) and len(set(resolved_keys)) == len(resolved_keys):
            return mapping
        print("    !! VDJ overlap did not resolve every suffix uniquely; falling back to GSM order.")

    # 2) fallback: assume aggregation order == GSM order
    if suffixes == list(range(1, len(ge_samples) + 1)):
        print(
            "[assign] WARNING: assigning suffix -N to the N-th GE sample in GSM order. "
            "This is an assumption; check the results (e.g. IP samples should have mostly "
            "naive/stem-like cells)."
        )
        return {i + 1: ge_samples[i] for i in range(len(ge_samples))}

    sys.exit(
        f"Found barcode suffixes {suffixes} but {len(ge_samples)} GE samples, and no VDJ "
        "barcode overlap to resolve them. Cannot assign samples automatically."
    )


def download_real():
    try:
        import GEOparse
    except ImportError:
        sys.exit("GEOparse is required for the real download. Install with:\n  pip install GEOparse")

    print(f"Fetching series metadata for {GEO_ACCESSION} ...")
    gse = GEOparse.get_GEO(geo=GEO_ACCESSION, destdir=str(DATA_RAW), silent=True)

    ge_samples, vdj_gsms = [], {}
    for gsm_name, gsm in sorted(gse.gsms.items()):
        title = gsm.metadata.get("title", [gsm_name])[0]
        meta = parse_sample_metadata(title)
        if meta is None:
            print(f"  (ignoring {gsm_name}: unrecognized title '{title}')")
            continue
        if meta["assay"] == "GE":
            ge_samples.append((gsm_name, meta))
        else:
            vdj_gsms[(meta["patient"], meta["timepoint"])] = (gsm_name, gsm)
    if not ge_samples:
        sys.exit("No 'GE' samples found in the series metadata.")
    print(f"[samples] {len(ge_samples)} GE samples, {len(vdj_gsms)} VDJ samples")

    # VDJ clonotype tables (small) are used only to validate cell -> sample assignment
    vdj_barcodes = {}
    for key, (gsm_name, gsm) in vdj_gsms.items():
        d = DATA_RAW / gsm_name
        files = list(d.rglob("*.csv.gz"))
        if not files:
            try:
                gsm.download_supplementary_files(directory=str(d), download_sra=False)
                files = list(d.rglob("*.csv.gz"))
            except Exception as e:  # noqa: BLE001
                print(f"    !! could not fetch VDJ for {gsm_name}: {e}")
        vdj_barcodes[key] = read_vdj_barcodes(files[0]) if files else set()
    n_bc = sum(1 for v in vdj_barcodes.values() if v)
    print(f"[vdj] barcodes found in {n_bc}/{len(vdj_barcodes)} VDJ files")

    X, cells, genes = load_matrix_sparse(fetch_matrix())
    mapping = assign_samples(cells, ge_samples, vdj_barcodes)

    suffixes = [_split_barcode(c)[1] for c in cells]
    rows = []
    for suf in suffixes:
        gsm_name, meta = mapping[suf]
        rows.append(
            {
                "patient": meta["patient"],
                "disease": meta["disease"],
                "timepoint": meta["timepoint"],
                "source": meta["source"],
                "day_post_infusion": meta["day_post_infusion"],
                "gsm": gsm_name,
            }
        )
    obs = pd.DataFrame(rows, index=cells)
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
    adata.var_names_make_unique()
    print(adata)
    print(adata.obs.groupby(["patient", "timepoint"], observed=True).size().to_string())
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