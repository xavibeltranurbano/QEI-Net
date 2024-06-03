# -----------------------------------------------------------------------------
# Main file for extracting features
# Author: Xavier Beltran Urbano
# Date Created: 21-02-2024
# -----------------------------------------------------------------------------

from network import MSC_QEI_Net
from keras import backend as K
from keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalFocalCrossentropy
import os
from utils import Utils
from configuration import Configuration
import tensorflow as tf
from predict_Test import predict_test
import numpy as np
import random

def run_program(config, networkName, params, rater):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    K.clear_session()  # Clear any existing TensorFlow session
    utils = Utils()
    trainGenerator, valGenerator = config.createAllDataGenerators(rater)  # Generate the IDs for train, val, and test
    network = MSC_QEI_Net(imgSize=params['targetSize'])
    model = network.get_model()
    model.compile(optimizer=Adam(learning_rate=0.00001), loss=CategoricalFocalCrossentropy(), metrics=['accuracy'])
    with tf.device('/GPU:0'):
        model_checkpoint_callback, reduce_lr_callback, early_stopping_callback = utils.allCallbacks(networkName, params['currentFold'], rater)
        epochs = 400
        history = model.fit(trainGenerator, validation_data=valGenerator, epochs=epochs, verbose=1, callbacks=[model_checkpoint_callback, reduce_lr_callback, early_stopping_callback])
        utils.save_training_plots(history, f"/results/{networkName}/{rater}/{params['currentFold']}/training_plots.png")
        loss, acc = model.evaluate(valGenerator, verbose=1)
        print(f"\nVal: MSE= {acc}, Loss= {loss}")

if __name__ == "__main__":
    imgPath = '/data_final'
    networkName = "MSC-QEI-Net"
    seed = 48
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    raters = ['JD', 'SD','ZW']
    os.makedirs(f"/results/{networkName}", exist_ok=True)  # Create folder for this experiment
    for i in range(1, 6):
        print("\n******************************************")
        print(f"----------Current Fold: {i}----------")
        params = {
            'pathData': imgPath,
            'targetSize': (64, 64, 32, 1),
            'batchSize': 64,
            'currentFold': i
        }
        config = Configuration(**params)
        print(config.returnVal_IDS())
        for rater in raters:
            print(f"\nRater {rater}")
            os.makedirs(f"/results/{networkName}/{rater}/{i}", exist_ok=True)  # Create folder for this experiment
            run_program(config, networkName, params, rater)  # Run experiment
    predict_test(networkName)
