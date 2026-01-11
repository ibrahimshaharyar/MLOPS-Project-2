from pathlib import Path
import tensorflow as tf

MODEL_PATH = Path("artifacts/model/best_model.keras")
EXPORT_DIR = Path("artifacts/exported_model")

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)

    # Make sure export directory exists (and is clean)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Export in TensorFlow SavedModel format (folder)
    model.export(str(EXPORT_DIR))

    print(f"Exported model to: {EXPORT_DIR.resolve()}")

if __name__ == "__main__":
    main()
