import tensorflow as tf


def load_image(image, image_shape=(224, 224)):
    if isinstance(image, str) or isinstance(image, tf.Tensor):
        image = tf.io.read_file(image)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, image_shape)
    return image
