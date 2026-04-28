import pandas as pd
import cv2
import numpy as np
from PIL import Image
import urllib.request
import os

# 1. Configuration
BASE_URL = "https://czi-subcell-public.s3.amazonaws.com/hpa-processed/cell_crops"
df = pd.read_csv("pilot_cells_to_download.csv")

# Ensure base directories exist
os.makedirs("images", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("tmp_downloads", exist_ok=True) # For the raw 4-channel PNGs

def to_8bit(arr):
    if arr.dtype == np.uint16:
        return (arr / 256).astype(np.uint8)
    return arr

def download_and_split(row, img_dir, prefix):
    # Construct S3 URL based on our known format
    cell_id = int(float(row['cell_id']))
    filename = f"{row['if_plate_id']}_{row['position']}_{row['sample']}_{cell_id}_cell_image.png"
    url = f"{BASE_URL}/{row['if_plate_id']}/{filename}"
    
    temp_path = f"tmp_downloads/{filename}"
    
    # Download
    if not os.path.exists(temp_path):
        urllib.request.urlretrieve(url, temp_path)
    
    # Read as BGRA and split
    img = cv2.imread(temp_path, -1)
    if img is None or img.shape[-1] != 4:
        raise ValueError(f"Invalid 4-channel image from {url}")
        
    paths = {}
    # SubCellPortable mapping: r=microtubule(B), y=er(G), b=dna(R), g=protein(A)
    for name, ch_idx in [('r', 0), ('y', 1), ('b', 2), ('g', 3)]:
        out_path = f"{img_dir}/{prefix}_{name}.png"
        Image.fromarray(to_8bit(img[:, :, ch_idx])).save(out_path)
        # Use absolute paths to prevent SubCellPortable routing errors
        paths[name] = os.path.abspath(out_path)
        
    # Optional: remove the temp 4-channel file to save space
    os.remove(temp_path) 
    return paths

# 2. Main Processing Loop
path_list_rows = []
print(f"Preparing SubCell CSV for {len(df)} images...")

for _, row in df.iterrows():
    gene = row['Gene']
    cell_id = int(float(row['cell_id']))
    
    # Create unique prefix: Gene_Plate_Position_Sample_CellID
    img_id = f"{row['if_plate_id']}_{row['position']}_{row['sample']}_{cell_id}"
    prefix = f"{gene}_{img_id}"
    
    img_dir = f"images/{gene}"
    out_dir = f"outputs/{gene}"
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    
    try:
        paths = download_and_split(row, img_dir, prefix)
        path_list_rows.append({
            'r_image': paths['r'],
            'y_image': paths['y'],
            'b_image': paths['b'],
            'g_image': paths['g'],
            'output_folder': os.path.abspath(out_dir),
            'output_prefix': prefix + '_'
        })
    except Exception as e:
        print(f"Failed {prefix}: {e}")

# 3. Save the final CSV for SubCellPortable
path_df = pd.DataFrame(path_list_rows)
path_df.to_csv('path_list.csv', index=False)
print(f"\nDone! Successfully generated path_list.csv with {len(path_df)} valid crop paths.")