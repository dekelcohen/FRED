import os
import json
import shutil
from tqdm import tqdm

# Import your custom dataset class
# Ensure your original code is saved as data.py in the same directory
from data import FREDDataset 

def export_split_to_coco(split_name, output_folder, fred_dataset):
    print(f"Exporting '{split_name}' to '{output_folder}'...")
    os.makedirs(output_folder, exist_ok=True)
    
    coco_format = {
        "info": {"description": "FRED Dataset from HuggingFace for RF-DETR"},
        "licenses": [],
        "categories":[{"id": 0, "name": "drone", "supercategory": "none"}],
        "images": [],
        "annotations":[]
    }
    
    ann_id = 0
    for idx in tqdm(range(len(fred_dataset)), desc=f"Processing {split_name}"):
        image, target = fred_dataset[idx]
        img_path = target["image_path"]
        
        # Rename image to avoid identical filenames across different sequences
        img_name = f"{split_name}_{idx}.jpg"
        dest_path = os.path.join(output_folder, img_name)
        
        # 1. Copy image
        shutil.copy(img_path, dest_path)
        
        # 2. Add to COCO images list
        width, height = image.size
        coco_format["images"].append({
            "id": target["image_id"],
            "file_name": img_name,
            "width": width,
            "height": height
        })
        
        # 3. Add to COCO annotations list
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
            
    # Save the annotations file with the specific name rfdetr expects
    json_path = os.path.join(output_folder, "_annotations.coco.json")
    with open(json_path, "w") as f:
        json.dump(coco_format, f)

if __name__ == "__main__":
    CLASS_TO_ID = {'drone': 0}
    RAW_DATASET_PATH = 'FRED_HF/' # Downloaded in Step 1
    OUTPUT_DIR = 'rfdetr_dataset/'
    
    # Load dataset using your class
    print("Loading raw datasets into memory...")
    train_dataset = FREDDataset(split='train', dataset_path=RAW_DATASET_PATH, class_to_id=CLASS_TO_ID, modality='rgb')
    test_dataset = FREDDataset(split='test', dataset_path=RAW_DATASET_PATH, class_to_id=CLASS_TO_ID, modality='rgb')
    
    # Export to RF-DETR structure
    # Note: We duplicate 'test' as 'valid' because RF-DETR expects a validation split during training
    export_split_to_coco("train", os.path.join(OUTPUT_DIR, "train"), train_dataset)
    export_split_to_coco("valid", os.path.join(OUTPUT_DIR, "valid"), test_dataset) 
    export_split_to_coco("test", os.path.join(OUTPUT_DIR, "test"), test_dataset)
    
    print("\nDataset preparation complete! Ready for training.")