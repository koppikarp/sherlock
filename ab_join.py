import pandas as pd
import urllib.request
import os

# ── 1. Download metadata ─────────────────────────────────────────────────────
metadata_file = "metadata.csv"
if not os.path.exists(metadata_file):
    print("Downloading metadata...")
    urllib.request.urlretrieve(
        "https://czi-subcell-public.s3.amazonaws.com/hpa-processed/cell_crops/metadata.csv",
        metadata_file
    )

# ── 2. Download test antibody list ───────────────────────────────────────────
test_ab_file = "test_antibodies.txt"
if not os.path.exists(test_ab_file):
    print("Downloading test antibody list...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/CellProfiling/subcell-embed/main/annotations/splits/test_antibodies.txt",
        test_ab_file
    )

test_ab_set = set(
    pd.read_csv(test_ab_file, header=None, names=["antibody_id"])["antibody_id"].str.strip()
)
print(f"Test antibodies: {len(test_ab_set)}")

# ── 3. Load data ─────────────────────────────────────────────────────────────
subcell_meta = pd.read_csv(metadata_file, index_col=0)
your_hpa = pd.read_csv("data/groundtruth/proteinatlas_filtered.csv")

# ── 4. Filter metadata to test-set antibodies BEFORE merging ─────────────────
# This is the key step — only keep cells whose antibody is in the held-out split
subcell_test = subcell_meta[subcell_meta["antibody"].isin(test_ab_set)].copy()
print(f"Cells from test-set antibodies: {len(subcell_test)}")
print(f"Unique test antibodies found in metadata: {subcell_test['antibody'].nunique()}")

# ── 5. Merge on Ensembl ID to get ground-truth labels ────────────────────────
merged = subcell_test.merge(
    your_hpa[["Gene", "Ensembl", "is_multi_localized", "Reliability (IF)"]],
    left_on="ensembl_ids",
    right_on="Ensembl",
    how="inner"
)
print(f"Cells after joining with HPA ground truth: {len(merged)}")
print(f"Unique antibodies retained: {merged['antibody'].nunique()}")
print(f"Unique genes retained:      {merged['gene_names'].nunique()}")
print(f"  Multi-localized proteins: {merged.groupby('gene_names')['is_multi_localized'].first().sum()}")
print(f"  Mono-localized proteins:  {(merged.groupby('gene_names')['is_multi_localized'].first() == 0).sum()}")

# ── 6. Reliability filter (optional but recommended) ─────────────────────────
# 'Supported' and 'Enhanced' are the two high-confidence tiers in HPA
# Dropping 'Uncertain' reduces label noise significantly
before = len(merged)
merged = merged[merged["Reliability (IF)"].isin(["Supported", "Enhanced"])]
print(f"After reliability filter: {len(merged)} cells ({before - len(merged)} dropped)")

# ── 7. Construct S3 paths ────────────────────────────────────────────────────
# Based on the metadata structure: if_plate_id / position / sample
# The cell crop filename pattern from CZI docs is:
# {if_plate_id}/{position}/{sample}/{cell_id}_cell_image.png
merged["s3_path"] = (
    "s3://czi-subcell-public/hpa-processed/cell_crops/"
    + merged["if_plate_id"].astype(str) + "/"
    + merged["position"].astype(str) + "/"
    + merged["sample"].astype(str) + "/"
    + merged["cell_id"].astype(int).astype(str) + "_cell_image.png"
)
merged["s3_folder_prefix"] = (
    "s3://czi-subcell-public/hpa-processed/cell_crops/"
    + merged["if_plate_id"].astype(str) + "/"
)

# ── 8. Clean up and save ─────────────────────────────────────────────────────
merged = merged.drop(columns=["Ensembl"])
merged.to_csv("cells_to_download.csv", index=False)

print(f"\nSaved {len(merged)} cell rows to cells_to_download.csv")
print(merged[["gene_names", "antibody", "if_plate_id", "s3_path", "is_multi_localized"]].head())

# ── 9. Also save a per-antibody summary for reference ────────────────────────
summary = (
    merged.groupby(["antibody", "gene_names", "is_multi_localized"])
    .agg(n_cells=("cell_id", "count"))
    .reset_index()
    .sort_values("n_cells", ascending=False)
)
summary.to_csv("test_set_summary.csv", index=False)
print(f"\nPer-antibody summary saved to test_set_summary.csv")
print(summary.head(10))