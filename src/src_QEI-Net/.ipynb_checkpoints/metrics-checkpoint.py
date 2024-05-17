# -----------------------------------------------------------------------------
# Metrics file
# Author: Xavier Beltran Urbano and Frederik Hartmann
# Date Created: 12-01-2024
# -----------------------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from keras import backend as K
from tensorflow.keras.metrics import mean_squared_error

def MSE(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return mean_squared_error(y_true, y_pred)
    
def MSE_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return mean_squared_error(y_true, y_pred)
    
def Pred(y_true, y_pred):
    return y_pred
    
def Rat(y_true, y_pred):
    return y_true

# Define a custom RMSE metric
def RMSE(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))