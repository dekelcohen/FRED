# python src/data/download_fred_hf.py 
  # Download entire FRED train + test 
# python src/data/download_fred_hf.py --zips 100.zip 101.zip 110.zip
  # Download 100.zip + 101.zip video seqs for train 110.zip for test and https://huggingface.co/datasets/GabrieleMagrini/FRED/tree/main/train
  # Note: Can call download script multiple times - it merges all downloads in train/<seq num> and test/<seq num>
import os
import zipfile
import argparse
from glob import glob
from tqdm import tqdm
from huggingface_hub import snapshot_download

def download_and_extract_hf(repo_id="GabrieleMagrini/FRED", local_dir="FRED_HF", specific_zips=None):
    print(f"Connecting to '{repo_id}' on Hugging Face...")
    print(f"Target local directory: {os.path.abspath(local_dir)}")
    
    # Restrict download to specific zip files if provided
    allow_patterns = None
    if specific_zips:
        # **/* matches the file regardless of if it's in the train/ or test/ folder
        allow_patterns =[f"**/{z}" for z in specific_zips]
        print(f"Restricting download to only match: {allow_patterns}")
        
    # 1. Download the raw dataset repository
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=allow_patterns,
        ignore_patterns=["*.parquet", "*.git*"] # Skip auto-generated HF metadata files
    )
    
    # 2. Extract zipped sequences
    zip_files = glob(os.path.join(local_dir, "**", "*.zip"), recursive=True)
    
    if zip_files:
        print(f"\nFound {len(zip_files)} zip files. Extracting...")
        for zip_path in tqdm(zip_files, desc="Processing Zip Files"):
            
            # Example zip_path: "FRED_HF/train/100.zip"
            # Extract to the directory CONTAINING the zip: "FRED_HF/train"
            # Ex: extract_dir will become "FRED_HF/train/218"
            # Because the zip internally already contains the "100" folder.
            extract_dir = os.path.dirname(zip_path)
            
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Check the internal structure of the zip
                    # If the zip already has a top-level folder (e.g., "100/coordinates.txt")
                    # we extract it to "FRED_HF/train/". 
                    # If it doesn't (e.g., "coordinates.txt" at the root), we extract it to "FRED_HF/train/100/"
                    top_level_items = {os.path.normpath(name).split(os.sep)[0] for name in zip_ref.namelist()}
                    folder_name = os.path.basename(zip_path).replace(".zip", "")
                    
                    if folder_name not in top_level_items:
                        extract_dir = os.path.join(extract_dir, folder_name)
                        os.makedirs(extract_dir, exist_ok=True)
                        
                    zip_ref.extractall(extract_dir)
                    
                # Clean up: Delete the zip file after successful extraction
                os.remove(zip_path)
                
            except zipfile.BadZipFile:
                print(f"\nWarning: {zip_path} is corrupt or not a valid zip file. Skipping.")
                
        print("\nExtraction complete! Data is correctly formatted for FREDDataset.")
    else:
        print("\nNo zip files found. Either they were already extracted or the filename was incorrect.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FRED dataset from Hugging Face.")
    parser.add_argument(
        "--zips", 
        nargs='*', # Accepts zero or multiple arguments
        help="Specific zip filenames to download (e.g., 218.zip 1.zip). If omitted, downloads everything.", 
        default=None
    )
    args = parser.parse_args()

    download_and_extract_hf(specific_zips=args.zips)