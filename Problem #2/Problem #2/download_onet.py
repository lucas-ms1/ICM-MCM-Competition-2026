from pathlib import Path
import requests
import zipfile
import io

url = "https://onetcenter.org/dl_files/database/db_30_1_text.zip"
base_dir = Path(__file__).parent / "data"
base_dir.mkdir(parents=True, exist_ok=True)
output_path = base_dir / "db_30_1_text.zip"
extract_path = base_dir / "onet_db"

print(f"Downloading {url} to {output_path}...")
try:
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")

    print(f"Extracting to {extract_path}...")
    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete.")

except Exception as e:
    print(f"Error: {e}")
