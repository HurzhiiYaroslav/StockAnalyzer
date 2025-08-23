import tensorflow as tf
import numpy as np
from tensorflow import keras

def to_coral_format(y_scalar, num_classes: int):
    """Converts a vector of scalar ranks into the CORAL format for model training using vectorized operations."""
    y_scalar = np.asarray(y_scalar)
    levels = np.arange(1, num_classes)
    coral_y = (y_scalar[:, np.newaxis] > levels).astype(int)
    return coral_y

class CoralLoss(keras.losses.Loss):
    def __init__(self, name="coral_loss", **kwargs):
        super().__init__(name=name, **kwargs)
    def call(self, y_true_coral, y_pred_logits):
        """
        Calculates the loss for a CORAL model, which is the mean of 
        binary cross-entropy across all outputs.
        """
        return tf.reduce_mean(
            keras.losses.binary_crossentropy(y_true_coral, y_pred_logits, from_logits=True)
        )
    
coral_loss = CoralLoss()
keras.utils.get_custom_objects().update({'CoralLoss': CoralLoss})

@tf.function
def coral_logits_to_probs(logits):
    """Converts the output logits from a CORAL model into probabilities using a sigmoid function."""
    return tf.math.sigmoid(logits)

@tf.function
def coral_probs_to_rank(probs):
    """Converts probabilities from a CORAL model into a final scalar rank."""
    return tf.reduce_sum(tf.cast(probs > 0.5, dtype=tf.int32), axis=1) + 1

@tf.function
def coral_probs_to_soft_rank(probs):
    """
    Converts probabilities from a CORAL model into a final "soft" scalar rank
    by summing the probabilities.
    """
    expected_rank_float = tf.reduce_sum(tf.cast(probs, dtype=tf.float32), axis=1) + 1.0
    return tf.round(expected_rank_float)
