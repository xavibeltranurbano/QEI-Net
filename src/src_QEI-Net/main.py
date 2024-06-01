# -----------------------------------------------------------------------------
# Main file for extracting features
# Author: Xavier Beltran Urbano
# Date Created: 21-02-2024
# -----------------------------------------------------------------------------

from network import QEI_Net
from keras import backend as K
from keras.optimizers import Adam
from metrics import MSE, Pred, Rat
import os
from utils import Utils
from configuration import Configuration
import tensorflow as tf
from predict_Test import predict_test
import numpy as np
import random
from tensorflow.keras.losses import MeanSquaredError


def run_program(config, networkName, params):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    K.clear_session()  # Clear any existing TensorFlow session
    utils = Utils()
    trainGenerator, valGenerator = config.createAllDataGenerators()  # Generate the IDs for train, val, and test
    network = QEI_Net(imgSize=params['targetSize'])
    model = network.get_model()
    model.summary()
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=[MSE, Pred, Rat])
    with tf.device('/GPU:0'):
        model_checkpoint_callback, reduce_lr_callback, early_stopping_callback = utils.allCallbacks(networkName, params[
            'currentFold'])
        epochs = 400
        history = model.fit(trainGenerator, validation_data=valGenerator, epochs=epochs, verbose=1,
                            callbacks=[model_checkpoint_callback, reduce_lr_callback, early_stopping_callback])
        utils.save_training_plots(history,
                                  f"/home/xurbano/QEI-ASL/results/{networkName}/{params['currentFold']}/training_plots.png")
        loss, acc, _, _ = model.evaluate(valGenerator, verbose=1)
        print(f"\nVal: MSE= {acc}, Loss= {loss}")


if __name__ == "__main__":
    imgPath = '/home/xurbano/QEI-ASL/data_final'
    networkName = "QEI-Net"
    seed = 48
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    os.makedirs(f"/home/xurbano/QEI-ASL/results/{networkName}", exist_ok=True)  # Create folder for this experiment

    for i in range(1, 6):
        print("\n******************************************")
        print(f"----------Current Fold: {i}----------")
        params = {
            'pathData': imgPath,
            'targetSize': (64, 64, 32, 1),
            'batchSize': 32,  # 20 works well
            'currentFold': i
        }

        os.makedirs(f"/home/xurbano/QEI-ASL/results/{networkName}/{i}",
                    exist_ok=True)  # Create folder for this experiment

        config = Configuration(**params)  # Configuration of the experiment
        print(config.returnVal_IDS())
        run_program(config, networkName, params)  # Run experiment

    predict_test(networkName)
