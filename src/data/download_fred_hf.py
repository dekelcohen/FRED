import os
import zipfile
from glob import glob
from tqdm import tqdm
from huggingface_hub import snapshot_download

def download_and_extract_hf(repo_id="GabrieleMagrini/FRED", local_dir="FRED_HF"):
    print(f"Downloading '{repo_id}' from Hugging Face...")
    
    # 1. Download the raw dataset repository
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        ignore_patterns=["*.parquet", "*.git*"] # Skip auto-generated Hugging Face metadata files
    )
    
    # 2. Extract zipped sequences
    # Based on the repo structure: local_dir/train/0.zip, 1.zip...
    zip_files = glob(os.path.join(local_dir, "**", "*.zip"), recursive=True)
    
    if zip_files:
        print(f"\nFound {len(zip_files)} zip files. Extracting...")
        for zip_path in tqdm(zip_files, desc="Processing Zip Files"):
            
            # e.g., zip_path = "FRED_HF/train/218.zip"
            # extract_dir will become "FRED_HF/train/218"
            extract_dir = os.path.splitext(zip_path)[0] 
            
            # Create the sequence directory
            os.makedirs(extract_dir, exist_ok=True)
            
            try:
                # Extract the files (coordinates.txt, RGB folder, Event folder) into the new directory
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                    
                # Clean up: Delete the zip file after successful extraction to save disk space
                os.remove(zip_path)
                
            except zipfile.BadZipFile:
                print(f"\nWarning: {zip_path} is corrupt or not a valid zip file. Skipping.")
                
        print("\nExtraction complete! Data is correctly formatted for FREDDataset.")
    else:
        print("\nNo zip files found. Assuming folders are already extracted.")

if __name__ == "__main__":
    download_and_extract_hf()