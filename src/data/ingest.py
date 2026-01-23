import os
import zipfile
import urllib.request
from pathlib import Path

ZIP_URL = "https://github.com/ibrahimshaharyar/Chicken-Disease-Classification/raw/refs/heads/main/Chicken-fecal-images.zip"

OUT_DIR = Path("data/raw")
ZIP_PATH = OUT_DIR / "Chicken-fecal-images.zip"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset to: {ZIP_PATH}")
    urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)

    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(OUT_DIR)

    print("Done ✅")

if __name__ == "__main__":
    main()
