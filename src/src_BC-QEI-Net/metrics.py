# -----------------------------------------------------------------------------
# Metrics file
# Author: Xavier Beltran Urbano and Frederik Hartmann
# Date Created: 12-01-2024
# -----------------------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from keras import backend as K
from tensorflow.keras.metrics import MeanSquaredError
from sklearn.metrics import balanced_accuracy_score

def weighted_binary_crossentropy(weights={0: 0.655737705, 1: 0.344262295}):
    def loss(y_true, y_pred):
        # Ensure weights are properly handled
        weight_0 = tf.cast(weights[0], dtype=tf.float32)
        weight_1 = tf.cast(weights[1], dtype=tf.float32)
        
        # Convert y_true to float32
        y_true = tf.cast(y_true, dtype=tf.float32)
        
        # Clip predictions to avoid log(0) errors
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        
        # Calculate binary crossentropy with weights
        bce_loss = - (weight_1 * y_true * tf.math.log(y_pred) + weight_0 * (1 - y_true) * tf.math.log(1 - y_pred))
        
        # Return the mean loss
        return tf.reduce_mean(bce_loss)
    return loss

