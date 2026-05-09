import os
import json
import shutil
import argparse
import random
import copy
from tqdm import tqdm

# Import your custom dataset class
from data import FREDDataset 

def export_split_to_coco(split_name, output_folder, fred_dataset, max_bbox_size=None):
    if len(fred_dataset) == 0:
        print(f"\nSkipping '{split_name}' (0 images)...")
        return

    print(f"\nExporting '{split_name}' to '{output_folder}'...")
    os.makedirs(output_folder, exist_ok=True)
    
    excluded_seqs = set()
    
    # --- 1. FIRST PASS: Calculate mean bbox size to identify sequences to exclude ---
    if max_bbox_size is not None:
        print(f"Scanning '{split_name}' to find sequences with MEAN bbox area > {max_bbox_size}...")
        seq_stats = {}
        
        for idx in tqdm(range(len(fred_dataset)), desc="Scanning sequences"):
            _, target = fred_dataset[idx]
            
            # Navigate from FRED_HF/split/110/RGB/img.jpg ---> FRED_HF/split/110
            seq_name = os.path.dirname(os.path.dirname(target["image_path"]))
            
            if seq_name not in seq_stats:
                seq_stats[seq_name] = {"total_area": 0.0, "count": 0}
                
            for ann in target["annotations"]:
                seq_stats[seq_name]["total_area"] += ann["area"]
                seq_stats[seq_name]["count"] += 1

        for seq, stats in seq_stats.items():
            if stats["count"] > 0:
                mean_area = stats["total_area"] / stats["count"]
                if mean_area > max_bbox_size:
                    excluded_seqs.add(seq)
        
        if excluded_seqs:
            print(f"Excluded {len(excluded_seqs)} sequence folder(s) due to mean bbox area > {max_bbox_size}.")

    # --- 2. SECOND PASS: Export images and build COCO format ---
    coco_format = {
        "info": {"description": "FRED Dataset from HuggingFace for RF-DETR"},
        "licenses":[],
        "categories":[{"id": 0, "name": "drone", "supercategory": "none"}],
        "images": [],
        "annotations":[]
    }
    
    ann_id = 0
    images_exported = 0
    
    for idx in tqdm(range(len(fred_dataset)), desc=f"Processing {split_name}"):
        image, target = fred_dataset[idx]
        img_path = target["image_path"]
        
        seq_name = os.path.dirname(os.path.dirname(img_path))
        
        if max_bbox_size is not None and seq_name in excluded_seqs:
            continue
            
        img_name = os.path.basename(img_path)
        dest_path = os.path.join(output_folder, img_name)
        
        shutil.copy(img_path, dest_path)
        
        width, height = image.size
        coco_format["images"].append({
            "id": target["image_id"],
            "file_name": img_name,
            "width": width,
            "height": height
        })
        
        for ann in target["annotations"]:
            coco_format["annotations"].append({
                "id": ann_id,
                "image_id": target["image_id"],
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],  
                "area": ann["area"],
                "iscrowd": ann["iscrowd"]
            })
            ann_id += 1
            
        images_exported += 1
            
    json_path = os.path.join(output_folder, "_annotations.coco.json")
    with open(json_path, "w") as f:
        json.dump(coco_format, f)
        
    print(f"Finished exporting '{split_name}'. Included: {images_exported} images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare FRED Dataset for RF-DETR in COCO format")
    parser.add_argument("--max-bbox-size", type=float, default=None, 
                        help="Maximum MEAN bounding box area in pixels. If the mean bbox in a video sequence exceeds this, the entire sequence is excluded.")
    parser.add_argument("--dataset-path", type=str, default="./FRED_HF",
                        help="Path to the raw FRED dataset directory (default: ./FRED_HF)")
    # New Arguments for Splitting
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of the train set to extract as a validation set (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic sequence splitting")
    
    args = parser.parse_args()
    
    # Set seed for sequence shuffling
    random.seed(args.seed)

    CLASS_TO_ID = {'drone': 0}
    OUTPUT_DIR = 'rfdetr_dataset/'
    
    print("Loading raw datasets into memory...")
    
    # Load the datasets natively
    train_dataset_full = FREDDataset(split='train', dataset_path=args.dataset_path, class_to_id=CLASS_TO_ID, modality='rgb')
    test_dataset = FREDDataset(split='test', dataset_path=args.dataset_path, class_to_id=CLASS_TO_ID, modality='rgb')
    
    # ==========================================
    # Sequence-level Train/Validation Split
    # ==========================================
    all_train_paths = train_dataset_full.valid_image_paths
    
    # Identify unique sequence folders using the path structure
    unique_seqs = list(set([os.path.dirname(os.path.dirname(p['rgb'])) for p in all_train_paths]))
    unique_seqs.sort() # Sort to ensure deterministic behavior before shuffling
    random.shuffle(unique_seqs)
    
    # Calculate how many sequences belong to the validation set
    num_val_seqs = max(1, int(len(unique_seqs) * args.val_split)) if args.val_split > 0 else 0
    val_seqs = set(unique_seqs[:num_val_seqs])
    train_seqs = set(unique_seqs[num_val_seqs:])
    
    print(f"\n[Split Info] Total unique training sequences found: {len(unique_seqs)}")
    print(f"[Split Info] Splitting into -> Train sequences: {len(train_seqs)} | Val sequences: {len(val_seqs)}")
    
    # Filter the actual file paths
    train_paths_only =[p for p in all_train_paths if os.path.dirname(os.path.dirname(p['rgb'])) in train_seqs]
    val_paths_only =[p for p in all_train_paths if os.path.dirname(os.path.dirname(p['rgb'])) in val_seqs]
    
    # Create distinct datasets via shallow copy to reuse initialized data/caches
    train_dataset = copy.copy(train_dataset_full)
    valid_dataset = copy.copy(train_dataset_full)
    
    # Override the path lists to reflect the new splits
    train_dataset.valid_image_paths = train_paths_only
    valid_dataset.valid_image_paths = val_paths_only
    
    # ==========================================
    # Exporting
    # ==========================================
    export_split_to_coco("train", os.path.join(OUTPUT_DIR, "train"), train_dataset, max_bbox_size=args.max_bbox_size)
    export_split_to_coco("valid", os.path.join(OUTPUT_DIR, "valid"), valid_dataset, max_bbox_size=args.max_bbox_size) 
    export_split_to_coco("test",  os.path.join(OUTPUT_DIR, "test"),  test_dataset,  max_bbox_size=args.max_bbox_size)
    
    print("\nDataset preparation complete! Ready for training.")