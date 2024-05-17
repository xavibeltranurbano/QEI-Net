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
    #tf.print("Rating:", y_true, summarize=-1)  # summarize=-1 prints all elements
    #tf.print("Pred:", y_pred, summarize=-1)
    return mean_squared_error(y_true, y_pred)
    
def MSE_loss(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)
    
def Pred(y_true, y_pred):
    return y_pred
    
def Rat(y_true, y_pred):
    return y_true