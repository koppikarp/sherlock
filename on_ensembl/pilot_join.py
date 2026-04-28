import pandas as pd
import random

import pandas as pd
import urllib.request
import os

# 1. Download metadata if you haven't already (updated the URL to the exact location)
metadata_file = "metadata.csv"
if not os.path.exists(metadata_file):
    print("Downloading metadata...")
    urllib.request.urlretrieve(
        "https://czi-subcell-public.s3.amazonaws.com/hpa-processed/cell_crops/metadata.csv",
        metadata_file
    )

# 2. Load the data
# subcell_meta index might come in as 'Unnamed: 0', so we can ignore it or let pandas handle it
subcell_meta = pd.read_csv(metadata_file)
your_hpa = pd.read_csv("data/groundtruth/proteinatlas_filtered.csv")

# 3. Merge on Ensembl IDs (Safest method)
# Using left_on and right_on because the column names differ between the two CSVs
merged = subcell_meta.merge(
    your_hpa[['Gene', 'Ensembl', 'is_multi_localized', 'Reliability (IF)']], 
    left_on='ensembl_ids', 
    right_on='Ensembl', 
    how='inner'
)

# 4. Construct the S3 Folder paths
# You noticed the S3 bucket has folders like '99/', '997/', etc. 
# These correspond directly to the 'if_plate_id'. 
merged['s3_folder_prefix'] = "s3://czi-subcell-public/hpa-processed/cell_crops/" + merged['if_plate_id'].astype(str) + "/"

# (Optional) Clean up redundant columns from the merge
merged = merged.drop(columns=['Ensembl']) 

# 5. Check your results and save
print(f"Successfully matched {len(merged)} individual cell crops!")
print(merged[['gene_names', 'if_plate_id', 's3_folder_prefix']].head())

# 1. Filter for Supported reliability (if not already done in the initial CSV)
df_supported = merged[merged['Reliability (IF)'] == 'Supported']

# 2. Count crops per gene and enforce the floor (LID needs >= 5)
crop_counts = df_supported['Gene'].value_counts()
valid_genes = crop_counts[crop_counts >= 5].index
df_valid = df_supported[df_supported['Gene'].isin(valid_genes)]

# 3. Split into mono and multi-localized pools
mono_pool = df_valid[df_valid['is_multi_localized'] == 0]
multi_pool = df_valid[df_valid['is_multi_localized'] == 1]

# 4. Sample 100 unique genes from each pool for the pilot
# (Using a seed for reproducibility so you get the same 200 genes if you rerun)
random.seed(42)
pilot_mono_genes = random.sample(list(mono_pool['Gene'].unique()), 100)
pilot_multi_genes = random.sample(list(multi_pool['Gene'].unique()), 100)

# Combine and filter the dataframe to just these 200 genes
pilot_genes = pilot_mono_genes + pilot_multi_genes
pilot_df = df_valid[df_valid['Gene'].isin(pilot_genes)]

# 5. Enforce the ceiling: Cap at max 15 crops per gene for balancing and speed
# groupby().head(n) takes the first n rows for each group
final_pilot_df = pilot_df.groupby('Gene').head(15)

print(f"Original dataset: {len(merged)} crops")
print(f"Pilot dataset: {len(final_pilot_df)} crops across {final_pilot_df['Gene'].nunique()} genes")

# Save this lean list to drive your download script
final_pilot_df.to_csv("pilot_cells_to_download.csv", index=False)