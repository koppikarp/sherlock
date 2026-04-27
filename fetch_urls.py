import requests
import pandas as pd
import re
import time

df = pd.read_csv('data/groundtruth/proteinatlas_filtered.csv')
image_records = []

for _, row in df.iterrows():
    ensg = row['Ensembl']
    url = f"https://www.proteinatlas.org/{ensg}.json"
    
    try:
        r = requests.get(url, timeout=10)
        
        # Raise an error if the request itself failed (e.g., 404, 500)
        r.raise_for_status() 
        
        # Regex to find all IF composite image URLs directly from the raw text.
        # This handles various subdomains and optional yellow (ER) channels.
        pattern = r'(https?://[a-zA-Z0-9.-]*proteinatlas\.org/[^"\'\s]+_blue_red_green(?:_yellow)?\.jpg)'
        
        # Find all matches and remove duplicates (HPA sometimes lists the same URL twice)
        image_urls = list(set(re.findall(pattern, r.text)))
        
        if not image_urls:
            print(f"No IF images found for {ensg}")
            
        for img_url in image_urls:
            image_records.append({
                'Gene': row['Gene'],
                'Ensembl': ensg,
                'is_multi_localized': row['is_multi_localized'],
                'image_url': img_url
            })
            
        # Be polite to the HPA servers
        time.sleep(0.3)  
        
    except Exception as e:
        print(f"Failed request for {ensg}: {e}")

image_df = pd.DataFrame(image_records)
image_df.to_csv('hpa_image_urls.csv', index=False)

print(f"Extraction complete! Found {len(image_df)} unique images.")