from pathlib import Path
import random
import shutil

RAW_DIR = Path("data/raw/Chicken-fecal-images")
OUT_DIR = Path("data/split")

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
SEED = 42

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> list[Path]:
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]


def ensure_dirs(class_name: str):
    for split in ["train", "val", "test"]:
        (OUT_DIR / split / class_name).mkdir(parents=True, exist_ok=True)


def split_list(items: list[Path], train_r: float, val_r: float):
    n = len(items)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test


def copy_files(files: list[Path], dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dest_dir / f.name)


def main():
    random.seed(SEED)

    if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-9:
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0")

    classes = ["Healthy", "Coccidiosis"]

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {RAW_DIR}")

    print("Creating splits...\n")

    for cls in classes:
        class_dir = RAW_DIR / cls
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing folder: {class_dir}")

        images = list_images(class_dir)
        if len(images) == 0:
            raise ValueError(f"No images found in {class_dir}")

        random.shuffle(images)
        train, val, test = split_list(images, TRAIN_RATIO, VAL_RATIO)

        ensure_dirs(cls)
        copy_files(train, OUT_DIR / "train" / cls)
        copy_files(val, OUT_DIR / "val" / cls)
        copy_files(test, OUT_DIR / "test" / cls)

        print(f"{cls}: total={len(images)} | train={len(train)} val={len(val)} test={len(test)}")

    print("\nSplits created in data/split/")


if __name__ == "__main__":
    main()
