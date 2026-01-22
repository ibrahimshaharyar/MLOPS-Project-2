from pathlib import Path
import tensorflow as tf

from src.utils.common import read_yaml
from src.utils.image_ops import load_and_prepare_image

MODEL_PATH = Path("artifacts/model/best_model.keras")

CLASS_POS = "Healthy"
CLASS_NEG = "Coccidiosis"

class Predictor:
    def __init__(self):
        params = read_yaml("configs/params.yaml")
        self.image_size = int(params["image_size"])

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        self.model = tf.keras.models.load_model(MODEL_PATH)

    def predict(self, image_path: str):
        x = load_and_prepare_image(image_path, self.image_size)
        prob_pos = float(self.model.predict(x, verbose=0)[0][0])  # sigmoid output

        if prob_pos >= 0.5:
            label = CLASS_POS
            confidence = prob_pos
        else:
            label = CLASS_NEG
            confidence = 1.0 - prob_pos

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "prob_healthy": round(prob_pos, 4),
        }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to an image file")
    args = parser.parse_args()

    pred = Predictor()
    result = pred.predict(args.image)
    print(result)


if __name__ == "__main__":
    main()
