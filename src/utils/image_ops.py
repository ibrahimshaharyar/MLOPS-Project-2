import tensorflow as tf

def load_and_prepare_image(image_path: str, image_size: int):
    img_bytes = tf.io.read_file(image_path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, (image_size, image_size))
    img = tf.cast(img, tf.float32)  # keep float, model handles preprocess_input inside
    img = tf.expand_dims(img, axis=0)  # (1, H, W, 3)
    return img
