from pathlib import Path

RAW_DATA_DIR = Path("data/raw/Chicken-fecal-images")
CLASSES = ["Coccidiosis", "Healthy"] 

def main():
    print("Validating dataset...\n")

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError("data/raw folder not found")

    for cls in CLASSES:
        class_dir = RAW_DATA_DIR / cls

        if not class_dir.exists():
            raise FileNotFoundError(f"Missing folder: {class_dir}")

        images = list(class_dir.glob("*"))
        if len(images) == 0:
            raise ValueError(f"No images found in {class_dir}")

        print(f"{cls}: {len(images)} images")

    print("\nDataset validation passed")

if __name__ == "__main__":
    main()
