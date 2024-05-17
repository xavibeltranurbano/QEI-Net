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


def scatterPlot(ratings, predictions, currentFold, mse):
    # Create scatter plot of ratings vs predictions
    os.makedirs(f"/home/xurbano/QEI-ASL/results/QEI-NET_3_features/plotsValidation/ScatterPlot", exist_ok=True)

    plt.scatter(ratings, predictions, color='blue', label='Data points')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--',
             label='Perfect prediction')  # Add a diagonal dashed line for reference

    plt.xlabel('Ratings')
    plt.ylabel('Predictions')
    plt.title('Scatter Plot of the QEI-ASL')
    plt.legend([f'Predicted QEI (MSE {mse:.4f})', 'Perfect QEI'])

    save_path = f'/home/xurbano/QEI-ASL/results/QEI-NET_3_features/plotsValidation/ScatterPlot/{currentFold}.png'
    plt.savefig(save_path)
    plt.close()


def readAnnotation(ratings_excel, ID):
    # Read the annotation rating for the given ID
    matched_row = ratings_excel[ratings_excel.iloc[:, 0] == ID]
    rating = (matched_row.iloc[-1, -1] - 1.0) / 3.0
    return rating


def save_to_excel(fold_results, networkName, currentFold, writer):
    # Save fold results to Excel
    df = pd.DataFrame(fold_results, columns=['Name', 'Rating', 'Prediction', 'MSE'])
    mean_mse = {'Name': 'Mean MSE', 'Rating': None, 'Prediction': None, 'MSE': df['MSE'].mean()}
    mean_mse_df = pd.DataFrame([mean_mse])
    df_final = pd.concat([df, mean_mse_df], ignore_index=True)
    sheet_name = f'Fold_{currentFold}'
    df_final.to_excel(writer, sheet_name=sheet_name, index=False)


def predict_test():
    # Predict and save the test/validation data
    ratings_excel = pd.read_excel("/home/xurbano/QEI-ASL/data_final/Ratings.xlsx")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    K.clear_session()
    path = '/home/xurbano/QEI-ASL/data_final'
    allRatings = []
    allMSE = []
    features = pd.read_excel("/home/xurbano/QEI-ASL/src_3_features/computed_features.xlsx")

    # Initialize the ExcelWriter for the single Excel file
    excel_path = f'/home/xurbano/QEI-ASL/results/QEI-NET_3_features/All_Folds_Predictions.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        with tf.device('/GPU:0'):
            for currentFold in range(1, 2):
                # Parameters of the training
                params = {
                    'pathData': path,
                    'targetSize': (3,),
                    'batchSize': 1,
                    'currentFold': currentFold
                }
                config = Configuration(**params)
                valIDs = config.returnVal_IDS()
                network = FCDN_QEI(nFeatures=params['targetSize'])
                model = network.get_model()
                model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=[MSE, Pred, Rat])
                model.load_weights(f"/home/xurbano/QEI-ASL/results/QEI-NET_3_features/{currentFold}/Best_Model.keras")

                print(f"\n------------------FOLD {currentFold}------------------")

                fold_results = []
                ratings = []
                predictions = []
                MSE_vec = []

                for ID in valIDs:
                    x = features.loc[features['ID'] == ID]
                    x = x.iloc[:, 1:]  # Drop the ID column
                    y = readAnnotation(ratings_excel, ID)
                    x = tf.convert_to_tensor(x, dtype=tf.float32)
                    x_tensor = tf.reshape(x, (1, -1))  # Reshape to (1, num_features)
                    prediction = model.predict(x_tensor)[0]
                    predictions.append(prediction[0])
                    ratings.append(y)
                    allRatings.append(y)

                    y_tensor = tf.convert_to_tensor([y], dtype=tf.float32)  # Ensure y is a tensor
                    prediction_tensor = tf.convert_to_tensor([prediction[0]],
                                                             dtype=tf.float32)  # Ensure prediction is a tensor
                    mse = mean_squared_error(y_tensor, prediction_tensor).numpy()  # Convert tensor to numpy value

                    allMSE.append(mse)
                    MSE_vec.append(mse)

                    fold_results.append([ID, y, prediction[0], mse])

                # Save fold results to Excel
                save_to_excel(fold_results, '3FCN-QEI-Net', currentFold, writer)

                # Plot the results
                scatterPlot(ratings, predictions, currentFold, np.mean(MSE_vec))

        print(f"Prediction and MSE results saved for each fold in the '3FCN-QEI-Net' directory.")
        print(f"MEAN MSE: {np.mean(allMSE)}")


if __name__ == "__main__":
    predict_test()
