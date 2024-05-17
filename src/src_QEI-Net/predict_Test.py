# -----------------------------------------------------------------------------
# Predict the test/val data once the model is trained
# Author: Xavier Beltran Urbano
# Date Created: 21-02-2024
# -----------------------------------------------------------------------------

from keras.utils import Sequence
import numpy as np
import tensorflow as tf
from ASL_Preprocessing import Preprocessing
from tensorflow.keras.metrics import mean_squared_error
import os
from keras import backend as K
from configuration import Configuration
from network import QEI_Net
from keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from metrics import MSE, RMSE, Pred, Rat
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random


def scatterPlot(ratings, predictions, currentFold, mse, networkName):
    # Create scatter plot of ratings vs predictions
    os.makedirs(f"/home/xurbano/QEI-ASL/results/{networkName}/plotsValidation/ScatterPlot", exist_ok=True)
    plt.scatter(ratings, predictions, color='blue', label='Data points')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--',
             label='Perfect prediction')  # Add a diagonal dashed line for reference
    plt.xlabel('Ratings')
    plt.ylabel('Predictions')
    plt.title('Scatter Plot of the QEI-ASL')
    plt.legend([f'Predicted QEI (MSE {mse:.4f})', 'Perfect QEI'])
    save_path = f'/home/xurbano/QEI-ASL/results/{networkName}/plotsValidation/ScatterPlot/{currentFold}.png'
    plt.savefig(save_path)
    plt.close()


def compute_mse(y_true, y_pred):
    # Computr MSE of the rating and predictions
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    squared_differences = tf.square(y_true - y_pred)
    mse = tf.reduce_mean(squared_differences)
    return mse


def save_to_excel(fold_results, networkName, currentFold, writer):
    # Create DataFrame from the fold results
    df = pd.DataFrame(fold_results, columns=['Name', 'Rating', 'Prediction', 'MSE'])

    # Calculate the mean MSE and create a new DataFrame for it
    mean_mse = {'Name': 'Mean MSE', 'Rating': None, 'Prediction': None, 'MSE': df['MSE'].mean()}
    mean_mse_df = pd.DataFrame([mean_mse])

    # Concatenate the original DataFrame with the mean MSE DataFrame
    df_final = pd.concat([df, mean_mse_df], ignore_index=True)

    # Write the DataFrame to a new sheet in the Excel file, named by the currentFold
    sheet_name = f'Fold_{currentFold}'
    df_final.to_excel(writer, sheet_name=sheet_name, index=False)


def predict_test(networkName):
    # Predict the test set
    seed = 48
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    K.clear_session()
    path = '/home/xurbano/QEI-ASL/data_final'
    allMSE = []
    # Initialize the ExcelWriter for the single Excel file
    excel_path = f'/home/xurbano/QEI-ASL/results/{networkName}/All_Folds_Predictions.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        with tf.device('/GPU:0'):
            for currentFold in range(1, 2):
                params = {
                    'pathData': path,
                    'targetSize': (64, 64, 32, 1),
                    'batchSize': 30,
                    'currentFold': currentFold
                }
                preprocessing = Preprocessing(data_aug=False, norm_intensity=True, imageSize=(64, 64, 32))
                config = Configuration(**params)
                valIDs = config.returnVal_IDS()
                network = QEI_Net(imgSize=params['targetSize'])
                model = network.get_model()
                model.load_weights(f"/home/xurbano/QEI-ASL/results/{networkName}/{currentFold}/Best_Model.keras")

                fold_results = []
                for ID in valIDs:
                    x, y = preprocessing.process_case(ID, path)
                    x = np.expand_dims(x, axis=0)
                    prediction = model.predict(x)[0]
                    mse = compute_mse(y, prediction).numpy()  # Make sure to convert tensor to numpy value
                    fold_results.append([ID, y, prediction[0], mse])
                    allMSE.append(mse)

                    # Save fold results to Excel
                save_to_excel(fold_results, networkName, currentFold, writer)
                ratings, predictions, mse_values = [item[1] for item in fold_results], [item[2] for item in
                                                                                        fold_results], [item[3] for item
                                                                                                        in fold_results]
                scatterPlot(ratings, predictions, currentFold, np.mean(mse_values), networkName)

        print(f"Prediction and MSE results saved for each fold in the '{networkName}' directory.")
        print(f"MEAN AVG MSE: {np.mean(allMSE)}")


if __name__ == "__main__":
    predict_test('QEI-Net')
