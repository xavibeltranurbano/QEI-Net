# -----------------------------------------------------------------------------
# Predict the test/val data once the model is trained
# Author: Xavier Beltran Urbano
# Date Created: 21-02-2024
# -----------------------------------------------------------------------------

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from ASL_Preprocessing import Preprocessing
from configuration import Configuration
from network import MSC_QEI_Net
import matplotlib.pyplot as plt


def scatterPlot(ratings, predictions, currentFold, mse,networkName):
    # Create scatter plot of ratings vs predictions
    os.makedirs(f"/results/{networkName}/plotsValidation/ScatterPlot",
                exist_ok=True)
    plt.scatter(ratings, predictions, color='blue', label='Data points')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--',
             label='Perfect prediction')  # Add a diagonal dashed line for reference
    plt.xlabel('Ratings')
    plt.ylabel('Predictions')
    plt.title('Scatter Plot of the QEI-ASL')
    plt.legend([f'Predicted QEI (MSE {mse:.4f})', 'Perfect QEI'])
    save_path = f'/results/{networkName}/plotsValidation/ScatterPlot/{currentFold}.png'
    plt.savefig(save_path)
    plt.close()


def reorder_columns(df):
    # Reorder columns to move specific columns to the end
    cols_to_move = ['Rating', 'Prediction', 'SquaredError']
    for col in cols_to_move:
        if col not in df.columns:
            print(f"DataFrame does not have the required column: {col}")
            return df

    cols = [col for col in df.columns if col not in cols_to_move]
    new_order = cols + cols_to_move
    return df[new_order]


def save_to_excel(all_results, networkName):
    # Save results to an Excel file
    excel_path = f'/results/{networkName}/All_Folds_Predictions.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for fold, data in all_results.items():
            if not data:  # Check if data list is empty
                print(f"No data collected for fold {fold}. Skipping...")
                continue
            df = pd.DataFrame(data)
            df = reorder_columns(df)  # Reorder columns before adding the average MSE
            scatterPlot(df['Rating'], df['Prediction'], fold, np.mean(df['SE']),networkName)
            if 'SE' in df.columns:  # Check if the 'MSE' column exists
                mean_mse = pd.DataFrame(
                    {'Name': ['Average MSE'], 'Rating': [None], 'Prediction': [None], 'SE': [df['SE'].mean()]})
                df = pd.concat([df, mean_mse], ignore_index=True)

            sheet_name = f'Fold_{fold}'
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def predict_test(networkName):
    # Predict the test set
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    K.clear_session()
    path = '/data_final'
    raters = ['JD', 'SD','ZW']
    seed = 48
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    all_results = {fold: [] for fold in range(1, 6)}

    with tf.device('/GPU:0'):
        for currentFold in range(1, 6):
            params = {
                'pathData': path,
                'targetSize': (64, 64, 32, 1),
                'batchSize': 64,
                'currentFold': currentFold
            }
            config = Configuration(**params)
            for rater in raters:
                preprocessing = Preprocessing(data_aug=False, norm_intensity=True, imageSize=(64, 64, 32), rater=rater)
                valIDs = config.returnVal_IDS()
                network = MSC_QEI_Net(imgSize=params['targetSize'])
                model = network.get_model()
                model.load_weights(
                    f"/results/{networkName}/{rater}/{currentFold}/Best_Model.keras")

                for ID in valIDs:
                    x, y = preprocessing.process_case(ID, path)
                    x = np.expand_dims(x, axis=0)
                    prediction = model.predict(x)
                    # Calculate weighted average prediction
                    weighted_avg_prediction = np.dot(prediction[0], np.arange(1, prediction.shape[1] + 1))
                    print(prediction[0],weighted_avg_prediction)
                    predicted_class = weighted_avg_prediction

                    result = next((item for item in all_results[currentFold] if item['Name'] == ID), None)
                    if not result:
                        result = {'Name': ID, 'Rating': [], 'Prediction': [], 'SE': []}
                        all_results[currentFold].append(result)

                    result[f'GT {rater}'] = y
                    result[f'Pred {rater}'] = predicted_class
                    result['Rating'].append(y)
                    result['Prediction'].append(predicted_class)

            # Final computation for each ID in the current fold
            for result in all_results[currentFold]:
                mean_rating = (np.mean(result['Rating']) - 1.0) / 3.0
                mean_prediction = (np.mean(result['Prediction']) - 1.0) / 3.0
                squared_error = (mean_rating - mean_prediction) ** 2

                result['Rating'] = mean_rating
                result['Prediction'] = mean_prediction
                result['SE'] = squared_error

    save_to_excel(all_results,networkName)


if __name__ == "__main__":
    predict_test('MSC-QEI-Net')