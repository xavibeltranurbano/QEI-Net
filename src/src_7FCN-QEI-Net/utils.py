# -----------------------------------------------------------------------------
# Utils file
# Author: Xavier Beltran Urbano
# Date Created: 22-02-2024
# -----------------------------------------------------------------------------

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


class Utils:
    def __init__(self):
        # Initialize the utility class
        pass

    @staticmethod
    def allCallbacks(networkName, currentFold):
        # Create and return callbacks for model checkpointing, learning rate reduction, and early stopping
        pathWeights = f"/results/{networkName}/{currentFold}"
        checkpoint_path = pathWeights + "/Best_Model.keras"
        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=0
        )

        reduce_lr_callback = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,  # Factor by which the learning rate will be reduced. new_lr = lr * factor
            patience=10,  # Number of epochs with no improvement after which learning rate will be reduced.
            min_lr=0.0000000000001,  # Lower bound on the learning rate.
            verbose=1
        )

        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=30,  # Number of epochs with no improvement after which training will be stopped.
            verbose=1
        )

        return model_checkpoint_callback, reduce_lr_callback, early_stopping_callback

    @staticmethod
    def save_training_plots(history, file_path):
        # Save training and validation loss and MSE plots
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        MSE = history.history['mse']
        val_MSE = history.history['val_mse']

        epochs = len(train_loss)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs + 1), train_loss, label='Training Loss')
        plt.plot(range(1, epochs + 1), val_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs + 1), MSE, label='Training MSE')
        plt.plot(range(1, epochs + 1), val_MSE, label='Validation MSE')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.title('Training and Validation MSE')
        plt.legend()

        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
