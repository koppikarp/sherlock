import pandas as pd
import urllib.request
import os

# ── 1. Download metadata and test splits ─────────────────────────────────────
metadata_file = "metadata.csv"
if not os.path.exists(metadata_file):
    print("Downloading metadata...")
    urllib.request.urlretrieve(
        "https://czi-subcell-public.s3.amazonaws.com/hpa-processed/cell_crops/metadata.csv",
        metadata_file
    )

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
print(f"Test antibodies total: {len(test_ab_set)}")

# ── 2. Load data and filter to test set ──────────────────────────────────────
subcell_meta = pd.read_csv(metadata_file, index_col=0)
your_hpa = pd.read_csv("data/groundtruth/proteinatlas_filtered.csv")

subcell_test = subcell_meta[subcell_meta["antibody"].isin(test_ab_set)].copy()
print(f"Cells from test-set antibodies: {len(subcell_test)}")

# ── 3. Merge ground truth and filter for reliability ─────────────────────────
merged = subcell_test.merge(
    your_hpa[["Gene", "Ensembl", "is_multi_localized", "Reliability (IF)"]],
    left_on="ensembl_ids",
    right_on="Ensembl",
    how="inner"
)

# Keep only high-confidence annotations
merged = merged[merged["Reliability (IF)"].isin(["Supported", "Enhanced"])]
print(f"Cells after reliability filter: {len(merged)}")

# ── 4. Apply Pilot constraints (Cap 30, Min 15) ──────────────────────────────
# Cap cells per antibody at 30 using a fixed seed for reproducibility
merged_capped = (
    merged.groupby("antibody", group_keys=False)
    .apply(lambda x: x.sample(min(len(x), 30), random_state=42))
    .reset_index(drop=True)
)

# Only keep antibodies with at least 15 cells
ab_counts = merged_capped.groupby("antibody").size()
valid_abs = ab_counts[ab_counts >= 15].index
merged_capped = merged_capped[merged_capped["antibody"].isin(valid_abs)]

# ── 5. Balanced sample (150 per class) ───────────────────────────────────────
ab_labels = (
    merged_capped.groupby("antibody")["is_multi_localized"]
    .first()
    .reset_index()
)

mono_population = ab_labels[ab_labels["is_multi_localized"] == 0]["antibody"]
multi_population = ab_labels[ab_labels["is_multi_localized"] == 1]["antibody"]

# Sample 150, or the maximum available if the strict filtering leaves fewer than 150
mono_sample_size = min(150, len(mono_population))
multi_sample_size = min(150, len(multi_population))

mono_abs = mono_population.sample(mono_sample_size, random_state=42)
multi_abs = multi_population.sample(multi_sample_size, random_state=42)
pilot_abs = set(mono_abs) | set(multi_abs)

pilot = merged_capped[merged_capped["antibody"].isin(pilot_abs)].copy()

# ── 6. Construct S3 paths for final dataset ──────────────────────────────────
pilot["s3_path"] = (
    "s3://czi-subcell-public/hpa-processed/cell_crops/"
    + pilot["if_plate_id"].astype(str) + "/"
    + pilot["position"].astype(str) + "/"
    + pilot["sample"].astype(str) + "/"
    + pilot["cell_id"].astype(int).astype(str) + "_cell_image.png"
)

pilot = pilot.drop(columns=["Ensembl"])

# ── 7. Output results ────────────────────────────────────────────────────────
print("\n--- Pilot Dataset Summary ---")
print(f"Total cells:             {len(pilot)}")
print(f"Total antibodies:        {pilot['antibody'].nunique()}")
print(f"  Multi-localized abs:   {pilot[pilot['is_multi_localized']==1]['antibody'].nunique()}")
print(f"  Mono-localized abs:    {pilot[pilot['is_multi_localized']==0]['antibody'].nunique()}")
print(f"Median cells per ab:     {pilot.groupby('antibody').size().median():.0f}")
print(f"Estimated download size: ~{len(pilot) * 0.2:.0f} MB")

pilot.to_csv("pilot_cells_to_download.csv", index=False)
print("\nSaved to pilot_cells_to_download.csv")

# Save a quick reference summary
summary = (
    pilot.groupby(["antibody", "gene_names", "is_multi_localized"])
    .agg(n_cells=("cell_id", "count"))
    .reset_index()
    .sort_values("n_cells", ascending=False)
)
summary.to_csv("pilot_test_set_summary.csv", index=False)