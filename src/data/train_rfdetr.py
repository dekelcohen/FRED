from rfdetr import RFDETRLarge

def main():
    DATASET_DIR = "rfdetr_dataset"
    OUTPUT_DIR = "rfdetr_fred_output"
    
    print("Initializing RF-DETR Base Model...")
    # Downloads the pretrained Roboflow weights automatically
    model = RFDETRLarge() 
    
    print(f"Starting training on dataset at {DATASET_DIR}...")
    #  RF-DETR uses PyTorch Lightning under the hood, you can pass standard Lightning Trainer arguments directly into model.train() via kwargs
    model.train(
        dataset_dir=DATASET_DIR, 
        epochs=10,               # Adjust based on your time constraints
        batch_size=16,            # Adjust based on GPU VRAM
        grad_accum_steps=1,      # Helps simulate larger batch sizes
        lr=1e-4,
        output_dir=OUTPUT_DIR,
        device="cuda"            # Change to 'cpu' or 'mps' if needed
    )

    print("\nTraining complete! Running inference on a test image...")
    
    # The training loop saves the best checkpoint here:
    best_weights_path = f"{OUTPUT_DIR}/checkpoint_best_total.pth"
    best_model = RFDETRLarge(pretrain_weights=best_weights_path)
    
    # Run prediction on the first test image
    test_img_path = f"{DATASET_DIR}/test/test_0.jpg"
    detections = best_model.predict(test_img_path)
    
    print("\nDetections:")
    print(detections)

if __name__ == "__main__":
    main()