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

# Save this so your bash/download script can read it next
merged.to_csv("cells_to_download.csv", index=False)