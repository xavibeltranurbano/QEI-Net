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
from keras import backend as K
from configuration import Configuration
from network import FCDN_QEI
from keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from metrics import MSE, Pred, Rat
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


def compute_se(y_true, y_pred):
    # Computr MSE of the rating and predictions
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    squared_difference = tf.square(y_true - y_pred)
    return squared_difference


def save_to_excel(fold_results, networkName, currentFold, writer):
    # Create DataFrame from the fold results
    df = pd.DataFrame(fold_results, columns=['Name', 'Rating', 'Prediction', 'SE'])
    mean_mse = {'Name': 'Mean MSE', 'Rating': None, 'Prediction': None, 'SE': df['SE'].mean()}
    mean_mse_df = pd.DataFrame([mean_mse])
    df_final = pd.concat([df, mean_mse_df], ignore_index=True)
    sheet_name = f'Fold_{currentFold}'
    df_final.to_excel(writer, sheet_name=sheet_name, index=False)

def readAnnotation(ratings_excel, ID):
    # Read the annotation rating for the given ID
    matched_row = ratings_excel[ratings_excel.iloc[:, 0] == ID]
    rating = (matched_row.iloc[-1, -1] - 1.0) / 3.0
    return rating

def predict_test(networkName):
    # Predict the test set
    seed = 48
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    K.clear_session()
    path = '/home/xurbano/QEI-ASL/data_final'
    allSE = []
    # Initialize the ExcelWriter for the single Excel file
    excel_path = f'/home/xurbano/QEI-ASL/results/{networkName}/All_Folds_Predictions.xlsx'
    features = pd.read_excel("/home/xurbano/QEI-ASL/new_code/src_7FCN-QEI-Net/computed_features.xlsx")
    ratings_excel = pd.read_excel("/home/xurbano/QEI-ASL/data_final/Ratings.xlsx")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        with tf.device('/GPU:0'):
            for currentFold in range(1, 6):
                params = {
                    'pathData': path,
                    'targetSize': (14,),
                    'batchSize': 64,
                    'currentFold': currentFold
                }
                config = Configuration(**params)
                valIDs = config.returnVal_IDS()
                network = FCDN_QEI(nFeatures=params['targetSize'])
                model = network.get_model()
                model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=[MSE, Pred, Rat])
                model.load_weights(f"/home/xurbano/QEI-ASL/results/7FCN-QEI-Net/{currentFold}/Best_Model.keras")

                fold_results = []
                for ID in valIDs:
                    x = features.loc[features['ID'] == ID]
                    x = x.iloc[:, 1:]  # Drop the ID column
                    y = readAnnotation(ratings_excel, ID)
                    x = tf.convert_to_tensor(x, dtype=tf.float32)
                    x_tensor = tf.reshape(x, (1, -1))  # Reshape to (1, num_features)
                    prediction = model.predict(x_tensor)[0]
                    squared_difference = compute_se(y, prediction).numpy()[0]  # Make sure to convert tensor to numpy value
                    fold_results.append([ID, y, prediction[0], squared_difference])
                    allSE.append(squared_difference)

                    # Save fold results to Excel
                save_to_excel(fold_results, networkName, currentFold, writer)
                ratings, predictions, se_values = [item[1] for item in fold_results], [item[2] for item in
                                                                                        fold_results], [item[3] for item
                                                                                                        in fold_results]
                scatterPlot(ratings, predictions, currentFold, np.mean(se_values), networkName)

        print(f"Prediction and MSE results saved for each fold in the '{networkName}' directory.")
        print(f"MEAN AVG MSE: {np.mean(allSE)}")


if __name__ == "__main__":
    predict_test('7FCN-QEI-Net')
