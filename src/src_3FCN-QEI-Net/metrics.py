# -----------------------------------------------------------------------------
# Metrics file
# Author: Xavier Beltran Urbano and Frederik Hartmann
# Date Created: 12-01-2024
# -----------------------------------------------------------------------------

from tensorflow.keras.metrics import mean_squared_error

def MSE(y_true, y_pred):
    # Return the Mean Squared Error between true and predicted values
    return mean_squared_error(y_true, y_pred)

def MSE_loss(y_true, y_pred):
    # Return the Mean Squared Error loss between true and predicted values
    return mean_squared_error(y_true, y_pred)

def Pred(y_true, y_pred):
    # Return the predicted values
    return y_pred

def Rat(y_true, y_pred):
    # Return the true values
    return y_true
