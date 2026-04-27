import pandas as pd

def prep_hpa_ground_truth(file_path):
    # 1. Define the strictly necessary columns
    keep_cols = [
        'Gene', 
        'Ensembl', 
        'Uniprot', 
        'Reliability (IF)', 
        'Subcellular location'
    ]
    
    # 2. Load only the necessary columns to save memory
    df = pd.read_csv(file_path, sep='\t', usecols=keep_cols)
    
    # 3. Filter for high-confidence IF annotations
    quality_mask = df['Reliability (IF)'].isin(['Enhanced', 'Supported'])
    df_clean = df[quality_mask].copy()
    
    # 4. Drop rows missing localization data entirely
    df_clean = df_clean.dropna(subset=['Subcellular location'])
    
    # 5. Create the binary label: 0 for mono-localized, 1 for multi-localized
    df_clean['is_multi_localized'] = df_clean['Subcellular location'].apply(
        lambda x: 1 if ',' in str(x) else 0
    )
    
    # Optional: Reset index for cleaner mapping later
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean

# Execute
df_ground_truth = prep_hpa_ground_truth('proteinatlas.tsv')
df_ground_truth.to_csv('proteinatlas_filtered.csv', index=False)
print(f"Curated dataset size: {len(df_ground_truth)} proteins")
print(df_ground_truth['is_multi_localized'].value_counts())