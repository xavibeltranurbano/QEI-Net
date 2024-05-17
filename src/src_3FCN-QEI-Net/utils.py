# -----------------------------------------------------------------------------
# Utils file
# Author: Xavier Beltran Urbano
# Date Created: 22-02-2024
# -----------------------------------------------------------------------------

import os
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

class Utils:
    def __init__(self):
        pass

    @staticmethod
    def allCallbacks(networkName,currentFold):
        # Save weights of each epoch
        pathWeights=f"/home/xurbano/QEI-ASL/results/{networkName}/{currentFold}"
        checkpoint_path = pathWeights+"/Best_Model.keras"
        model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=0)

        # Reduce learning rate
        reduce_lr_callback = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.1,        # Factor by which the learning rate will be reduced. new_lr = lr * factor
        patience=10,        # Number of epochs with no improvement after which learning rate will be reduced.
        min_lr=0.0000000000001,    # Lower bound on the learning rate.
        verbose=1)

        # Early stopping callback
        early_stopping_callback = EarlyStopping(
        monitor='val_loss', 
        patience=30,       # Number of epochs with no improvement after which training will be stopped.
        verbose=1)

        return model_checkpoint_callback,reduce_lr_callback,early_stopping_callback

    @staticmethod
    def save_training_plots(history, file_path):
        # Extract loss and Dice coefficient from the 'history' object
        train_loss = history.history['loss']#[10:-1]
        val_loss = history.history['val_loss']#[10:-1]
        MSE = history.history['mse']#[10:-1]
        val_MSE = history.history['val_mse']#[10:-1]

        # Determine the actual number of epochs
        epochs = len(train_loss)
        
        # Create subplots for Loss and Dice coefficient
        plt.figure(figsize=(12, 5))

        # Plot training and validation Loss
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs + 1), train_loss, label='Training Loss')
        plt.plot(range(1, epochs + 1), val_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # Plot training and validation Dice coefficient
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs + 1), MSE, label='Training MSE')
        plt.plot(range(1, epochs + 1), val_MSE, label='Validation MSE')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.title('Training and Validation MSE')
        plt.legend()

        # Save the plots to the specified file path
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
    
    
    
    