# -----------------------------------------------------------------------------
# Metrics file
# Author: Xavier Beltran Urbano and Frederik Hartmann
# Date Created: 12-01-2024
# -----------------------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from keras import backend as K
from tensorflow.keras.metrics import mean_squared_error
from sklearn.metrics import balanced_accuracy_score

def MSE(y_true, y_pred):
    #tf.print("Rating:", y_true, summarize=-1)  # summarize=-1 prints all elements
    #tf.print("Pred:", y_pred, summarize=-1)
    return mean_squared_error(y_true, y_pred)
    
def MSE_loss(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)
    
def Pred(y_true, y_pred):
    return y_pred
    
def Rat(y_true, y_pred):
    return y_true

# Define a custom RMSE metric
def RMSE(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

def weighted_categorical_crossentropy(weights):
    weights = tf.constant(weights, dtype=tf.float32)
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        loss = y_true * tf.math.log(y_pred) * weights
        loss = -tf.reduce_sum(loss, axis=-1)
        return loss
    return loss

def balanced_accuracy(y_true, y_pred):
    print(type(y_true))
    print(type(y_pred))
    return balanced_accuracy_score(y_true, y_pred)

import tensorflow as tf

def macro_accuracy(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)), axis=0)
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)), axis=0)
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)), axis=0)

    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())

    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

    macro_f1 = tf.reduce_mean(f1)
    return macro_f1
