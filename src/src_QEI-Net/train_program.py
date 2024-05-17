import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.optimizers import Adam, SGD
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from network import QEI_Net
from metrics import MSE, MSE_loss, Pred, Rat, RMSE
from utils import Utils
from configuration import Configuration
from predict_Test import predict_test


class Trainer:
    def __init__(self, config, networkName, params, epochs):
        self.config = config
        self.epochs=epochs
        self.networkName = networkName
        self.params = params
        self.currentlr=0.0001
        self.model = self.initialize_model()
        self.callbacks = Utils.allCallbacks(networkName, params['currentFold'])
        self.history = {'loss': [], 'mse': [], 'val_loss': [], 'val_mse': []}
        self.best_val_loss = float('inf')  # Initialize as instance variable
        self.epochs_no_improve1 = 0
        self.epochs_no_improve2 = 0
        self.patience1 = 10
        self.patience2 = 30

    def initialize_model(self):
        K.clear_session()
        network = QEI_Net(imgSize=self.params['targetSize'])
        model = network.get_model()
        model.compile(
            optimizer=Adam(learning_rate=self.currentlr),
            loss='mean_squared_error',
            metrics=[MSE, Pred, Rat]
        )
        return model

    def run(self):
        trainGenerator, valGenerator = self.config.createAllDataGenerators()
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            # Training phase
            epoch_losses = []
            epoch_mses = []
            for x_batch, y_batch in trainGenerator:
                metrics = self.model.train_on_batch(x_batch, y_batch)
                epoch_losses.append(metrics[0])
                epoch_mses.append(metrics[1])

            avg_loss = np.mean(epoch_losses)
            avg_mse = np.mean(epoch_mses)
            self.history['loss'].append(avg_loss)
            self.history['mse'].append(avg_mse)

            # Validation phase
            val_loss, val_mse = self.validate(valGenerator)
            self.history['val_loss'].append(val_loss)
            self.history['val_mse'].append(val_mse)

            # Callbacks handling
            earlyStop=self.handle_callbacks(epoch, val_loss)

            # Print epoch summary
            print(f'Epoch {epoch+1}: Loss = {avg_loss}, MSE = {avg_mse}, Val Loss = {val_loss}, Val MSE = {val_mse}')
            if earlyStop: break

        # Save training plots
        Utils.save_training_plots(self.history, f"/home/xurbano/QEI-ASL/results/{self.networkName}/{self.params['currentFold']}/training_plots.png")

    def handle_callbacks(self, epoch, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_no_improve1=0
            self.epochs_no_improve2 = 0
            # Save the model manually
            model_path = f"/home/xurbano/QEI-ASL/results/{self.networkName}/{self.params['currentFold']}/Best_Model.keras"
            self.model.save(model_path)
        else:
            self.epochs_no_improve1 += 1
            self.epochs_no_improve2 += 1

        if self.epochs_no_improve1 >= self.patience1:
            self.epochs_no_improve1=0
            self.model.optimizer.lr.assign(self.currentlr*0.1)
            
        if self.epochs_no_improve2>=self.patience2:
            print("Early stopping triggered")
            return True
        else:
            return False

    def validate(self, generator):
        total_loss = 0
        total_mse = 0
        count = 0
        for x_val, y_val in generator:
            loss,mse,_,_ = self.model.evaluate(x_val, y_val, verbose=0)
            total_loss += loss
            total_mse += mse
            count += 1
        avg_loss = total_loss / count
        avg_mse = total_mse / count
        return avg_loss, avg_mse


if __name__ == "__main__":
    imgPath = '/home/xurbano/QEI-ASL/data_final'
    networkName = "QEI-NET_CNN_final"
    seed = 48
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Set the number of folds for cross-validation
    num_folds = 5  # Adjust the number of folds here

    # Create main directory for this experiment
    os.makedirs(f"/home/xurbano/QEI-ASL/results/{networkName}", exist_ok=True)

    # Loop through each fold
    for i in range(1, num_folds + 1):
        print("\n******************************************")
        print(f"----------Current Fold: {i}----------")
        
        # Parameters of the training for each fold
        params = {
            'pathData': imgPath,
            'targetSize': (64, 64, 32, 1),
            'batchSize': 2,  # 20 works well, adjust based on hardware capability
            'currentFold': i
        }
        
        # Create folder for this specific fold
        os.makedirs(f"/home/xurbano/QEI-ASL/results/{networkName}/{i}", exist_ok=True)
        
        # Configuration of the experiment
        config = Configuration(**params)
        print(config.returnVal_IDS())

        # Initialize and run the training process
        trainer = Trainer(config, networkName, params, epochs=10)
        trainer.run()

        # Optional: Perform any post-training analysis or predictions
    predict_test(networkName)  


