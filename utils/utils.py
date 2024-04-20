import tensorflow as tf


def load_image(image_path, img_shape=(224, 224)):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_shape)
    return img
