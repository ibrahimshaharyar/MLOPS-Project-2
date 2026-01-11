from pathlib import Path
import tensorflow as tf

from src.utils.common import read_yaml
from src.model.build import build_model

TRAIN_DIR = Path("data/split/train")
VAL_DIR = Path("data/split/val")
OUT_DIR = Path("artifacts/model")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    params = read_yaml("configs/params.yaml")
    image_size = int(params["image_size"])
    batch_size = int(params["batch_size"])
    epochs = int(params["epochs"])
    lr = float(params["learning_rate"])
    seed = int(params.get("seed", 42))

    # IMPORTANT: Keras expects subfolder names as class labels
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="binary",
        shuffle=True,
        seed=seed
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="binary",
        shuffle=False
    )

    # Performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    model = build_model(image_size=image_size, learning_rate=lr)

    checkpoint_path = OUT_DIR / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    
    model.save(OUT_DIR / "final_model.keras")
    print(f"\nSaved best model to: {checkpoint_path}")
    print(f"Saved final model to: {OUT_DIR / 'final_model.keras'}")

if __name__ == "__main__":
    main()
