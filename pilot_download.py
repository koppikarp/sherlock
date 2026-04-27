import pandas as pd
import urllib.request
import os

# Load your targeted pilot dataset
df = pd.read_csv("pilot_cells_to_download.csv")

# Grab just the first 3 rows for testing
test_df = df.head(3)

# Create a test directory
output_dir = "test_images"
os.makedirs(output_dir, exist_ok=True)

base_url = "https://czi-subcell-public.s3.amazonaws.com/hpa-processed/cell_crops"

print(f"Testing download for {len(test_df)} images...\n" + "-"*40)

for index, row in test_df.iterrows():
    # Safely convert cell_id (e.g., from '1.0' to '1')
    cell_id = int(float(row['cell_id'])) 
    
    # The format we verified from your 'aws s3 ls' output
    filename = f"{row['if_plate_id']}_{row['position']}_{row['sample']}_{cell_id}_cell_image.png"
    
    url = f"{base_url}/{row['if_plate_id']}/{filename}"
    local_path = os.path.join(output_dir, filename)
    
    print(f"Attempting: {filename}")
    print(f"Target URL: {url}")
    
    try:
        urllib.request.urlretrieve(url, local_path)
        print(f"✅ Success! Saved to {local_path}\n")
    except Exception as e:
        print(f"❌ Failed: {e}\n")

print("Test complete.")