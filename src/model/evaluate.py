from pathlib import Path
import json
import tensorflow as tf
import numpy as np

from src.utils.common import read_yaml

TEST_DIR = Path("data/split/test")
MODEL_PATH = Path("artifacts/model/best_model.keras")
OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    params = read_yaml("configs/params.yaml")
    image_size = int(params["image_size"])
    batch_size = int(params["batch_size"])

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="binary",
        shuffle=False
    )

    AUTOTUNE = tf.data.AUTOTUNE
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    model = tf.keras.models.load_model(MODEL_PATH)

    results = model.evaluate(test_ds, return_dict=True)
    metrics = {k: float(v) for k, v in results.items()}

    # Save metrics
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nTest metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
