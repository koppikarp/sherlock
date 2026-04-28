import cv2
import numpy as np
from PIL import Image
import os
import glob

def split_subcell_png(filepath, out_dir, prefix):
    # Read as 4-channel (BGRA), preserving 16-bit
    img = cv2.imread(filepath, -1)  # shape: (H, W, 4), BGRA order
    
    # Catch any accidentally downloaded masks or malformed images
    if img is None or len(img.shape) < 3 or img.shape[-1] != 4:
        print(f"  [!] Skipping {filepath}: Not a 4-channel image.")
        return None

    microtubule = img[:, :, 0]   # B channel (index 0)
    er          = img[:, :, 1]   # G channel (index 1)
    dna         = img[:, :, 2]   # R channel (index 2)
    protein     = img[:, :, 3]   # A channel (index 3)
    
    # Normalise to 8-bit for SubCellPortable if needed
    def to_8bit(arr):
        if arr.dtype == np.uint16:
            return (arr / 256).astype(np.uint8)
        return arr
    
    paths = {}
    # SubCellPortable suffixes: r=microtubule, y=er, b=dna, g=protein
    for name, data in [('r', microtubule), ('y', er), 
                       ('b', dna), ('g', protein)]:
        path = f"{out_dir}/{prefix}_{name}.png"
        Image.fromarray(to_8bit(data)).save(path)
        paths[name] = path
    
    return paths

# --- Execution Logic ---

input_dir = "test_images"
output_dir = "split_test_images"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Find all PNGs in the test directory
image_files = glob.glob(os.path.join(input_dir, "*.png"))

print(f"Found {len(image_files)} images to split in '{input_dir}'.\n" + "-"*40)

for filepath in image_files:
    # Extract the base filename without the extension (e.g., '1_E6_1_1_cell_image')
    filename = os.path.basename(filepath)
    prefix = os.path.splitext(filename)[0]
    
    print(f"Splitting {filename}...")
    saved_paths = split_subcell_png(filepath, output_dir, prefix)
    
    if saved_paths:
        print(f"  ✅ Saved 4 channels to {output_dir}/")

print("-" * 40 + "\nFinished splitting test images.")