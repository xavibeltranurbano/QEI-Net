# -----------------------------------------------------------------------------
# Predict the test/val data once the model is trained
# Author: Xavier Beltran Urbano 
# Date Created: 21-02-2024
# -----------------------------------------------------------------------------

from keras.utils import Sequence
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import mean_squared_error
import os
from configuration import Configuration
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random



def scatterPlot(ratings, predictions, currentFold, mse):
    os.makedirs(f"/home/xurbano/QEI-ASL/results/QEI-NET_3_features_Sudipto/plotsValidation/ScatterPlot", exist_ok=True)
    
    plt.scatter(ratings, predictions, color='blue', label='Data points')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Perfect prediction')  # Add a diagonal dashed line for reference
    
    plt.xlabel('Ratings')
    plt.ylabel('Predictions')
    plt.title('Scatter Plot of the QEI-ASL')
    plt.legend([f'Predicted QEI (MSE {mse:.4f})','Perfect QEI'])
    
    save_path = f'/home/xurbano/QEI-ASL/results/QEI-NET_3_features_Sudipto/plotsValidation/ScatterPlot/{currentFold}.png'
    plt.savefig(save_path)
    plt.close()

    
def predict_test():
    tf.random.set_seed(48)
    np.random.seed(48)
    random.seed(48)
    path = '/home/xurbano/QEI-ASL/data_final'
    allRatings = []
    allPredictions = []
    features = pd.read_excel("/home/xurbano/QEI-ASL/src_3_features_Sudipto/computed_QEI.xlsx")
    
    with tf.device('/GPU:0'):
        for currentFold in range(1, 6):
            # Parameters of the training
            params = {
                'pathData': path,
                'targetSize': (3,),
                'batchSize': 1,
                'currentFold': currentFold
            }
            config = Configuration(**params)
            valIDs = config.returnVal_IDS()
            print(valIDs)
            print(f"\n------------------FOLD {currentFold}------------------")
            ratings = []
            predictions = []
            for ID in valIDs:
                feature = features.loc[features['ID'] == ID]
                x = feature['QEI_Sudipto'].values[0]
                y = feature['Ratings'].values[0]
                predictions.append(x)
                ratings.append(y)
                allRatings.append(y)
                allPredictions.append(x)
            
            print(ratings)
            print(np.asarray(ratings).shape)
            print(np.asarray(predictions).shape)
            
            # Calculate MSE for the current fold
            mse = mean_squared_error(ratings, predictions).numpy()
            print(f"Fold {currentFold} MSE: {mse}")
            
            # Plot the results
            scatterPlot(ratings, predictions, currentFold, mse)
        
        # Calculate overall MSE
        overall_mse = mean_squared_error(allRatings, allPredictions).numpy()
        print(f"MEAN MSE: {overall_mse}")

if __name__ == "__main__":
    predict_test()

            
if __name__ == "__main__":
    predict_test()