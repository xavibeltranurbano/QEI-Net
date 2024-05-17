# -----------------------------------------------------------------------------
# Main file for 3FCN-QEI-Net
# Author: Xavier Beltran Urbano
# Date Created: 21-02-2024
# -----------------------------------------------------------------------------

from network import FCDN_QEI
from keras import backend as K
from keras.optimizers import Adam
from metrics import MSE, Pred, Rat
import os
from utils import Utils
from configuration import Configuration
import tensorflow as tf
import numpy as np
import random
from predict_Test import predict_test


def run_program(config, networkName, params):
    # Set TensorFlow logging level to filter out unnecessary logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Clear any existing TensorFlow session to prevent resource clutter
    K.clear_session()

    # Initialize utilities for the experiment
    utils = Utils()

    # Generate data for training and validation
    trainGenerator, valGenerator = config.createAllDataGenerators()

    # Create and compile the neural network model
    network = FCDN_QEI(nFeatures=params['targetSize'])
    model = network.get_model()
    model.summary()
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=[MSE, Pred, Rat])

    # Specify the GPU device for training
    with tf.device('/GPU:0'):
        # Setup training callbacks for saving model, reducing learning rate, and early stopping
        callbacks = utils.allCallbacks(networkName, params['currentFold'])
        epochs = 4000

        # Train the model
        history = model.fit(trainGenerator, validation_data=valGenerator, epochs=epochs, verbose=1, callbacks=callbacks)

        # Save training history plots
        utils.save_training_plots(history,
                                  f"/home/xurbano/QEI-ASL/results/{networkName}/{params['currentFold']}/training_plots.png")

        # Evaluate model on validation data and print results
        loss, acc, _, _ = model.evaluate(valGenerator, verbose=1)
        print(f"\nVal: MSE= {acc}, Loss= {loss}")


if __name__ == "__main__":
    # Setup path and network configuration
    imgPath = '/home/xurbano/QEI-ASL/data_final'
    networkName = "QEI-NET_3_features"
    seed = 48

    # Set seeds for reproducibility
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Create the main results directory for the experiment
    os.makedirs(f"/home/xurbano/QEI-ASL/results/{networkName}", exist_ok=True)

    # Perform experiment for each fold (here only one fold as example)
    for i in range(1, 2):
        print("\n******************************************")
        print(f"----------Current Fold: {i}----------")

        # Parameters of the training
        params = {
            'pathData': imgPath,
            'targetSize': (3,),
            'batchSize': 256,
            'currentFold': i
        }

        # Ensure a unique folder is created for each fold
        os.makedirs(f"/home/xurbano/QEI-ASL/results/{networkName}/{i}", exist_ok=True)

        # Configuration setup for this fold
        config = Configuration(**params)

        # Run the training experiment
        run_program(config, networkName, params)

    # Additional post-training analysis
    predict_test()
