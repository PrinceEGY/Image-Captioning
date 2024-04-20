import keras
import tensorflow as tf


def masked_loss(labels, preds):
    loss_fn = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=None
    )
    loss = loss_fn(labels, preds)

    mask = (labels != 0) & (loss < 1e8)
    mask = tf.cast(mask, loss.dtype)

    loss = loss * mask
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss
