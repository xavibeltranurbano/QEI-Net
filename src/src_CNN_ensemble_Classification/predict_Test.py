import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from ASL_Preprocessing import Preprocessing
from configuration import Configuration
from network import QEI_Net
from tensorflow.keras.losses import MeanSquaredError, CategoricalFocalCrossentropy
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import mean_squared_error

def scatterPlot(ratings, predictions, currentFold, mse):
    os.makedirs(f"/home/xurbano/QEI-ASL/results/QEI-NET_CNN_ensemble_Classification_final/plotsValidation/ScatterPlot", exist_ok=True)
    plt.scatter(ratings, predictions, color='blue', label='Data points')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Perfect prediction')  # Add a diagonal dashed line for reference 
    plt.xlabel('Ratings')
    plt.ylabel('Predictions')
    plt.title('Scatter Plot of the QEI-ASL')
    plt.legend([f'Predicted QEI (MSE {mse:.4f})','Perfect QEI'])
    save_path = f'/home/xurbano/QEI-ASL/results/QEI-NET_CNN_ensemble_Classification_final/plotsValidation/ScatterPlot/{currentFold}.png'
    plt.savefig(save_path)
    plt.close()

def reorder_columns(df):
    # Ensure the specific columns are in the DataFrame
    cols_to_move = ['Rating', 'Final Pred', 'MSE']
    for col in cols_to_move:
        if col not in df.columns:
            print(f"DataFrame does not have the required column: {col}")
            return df

    # Columns to keep in front
    cols = [col for col in df.columns if col not in cols_to_move]
    # New order: columns to keep in front + columns to move at the end
    new_order = cols + cols_to_move

    # Return the DataFrame with columns reordered
    return df[new_order]



def save_to_excel(all_results):
    excel_path = f'/home/xurbano/QEI-ASL/results/QEI-NET_CNN_ensemble_Classification_final/All_Folds_Predictions.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for fold, data in all_results.items():
            if not data:  # Check if data list is empty
                print(f"No data collected for fold {fold}. Skipping...")
                continue
            df = pd.DataFrame(data)
            df = reorder_columns(df)  # Reorder columns before adding the average MSE
            scatterPlot(df['Rating'], df['Final Pred'], fold, np.mean(df['MSE']))
            if 'MSE' in df.columns:  # Check if the 'MSE' column exists
                mean_mse = pd.DataFrame({'ID': ['Average MSE'], 'Rating': [None], 'Final Pred': [None], 'MSE': [df['MSE'].mean()]})
                df = pd.concat([df, mean_mse], ignore_index=True)

            sheet_name = f'Fold_{fold}'
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def predict_test():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    K.clear_session()
    path = '/home/xurbano/QEI-ASL/data_final'
    raters = ['JD']# ['Ali', 'JD', 'RW']
    seed = 48
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    all_results = {fold: [] for fold in range(1, 6)}
    
    with tf.device('/GPU:0'):
        for currentFold in range(1, 2):
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
                network = QEI_Net(imgSize=params['targetSize'])
                model = network.get_model()
                model.load_weights(f"/home/xurbano/QEI-ASL/results/QEI-NET_CNN_ensemble_Classification_final/{rater}/{currentFold}/Best_Model.keras")
                
                for ID in valIDs:
                    x, y = preprocessing.process_case(ID, path)
                    x = np.expand_dims(x, axis=0)
                    prediction = model.predict(x)
                    predicted_class = np.argmax(prediction[0]) + 1
                    
                    result = next((item for item in all_results[currentFold] if item['ID'] == ID), None)
                    if not result:
                        result = {'ID': ID, 'Rating': [], 'Final Pred': []}
                        all_results[currentFold].append(result)
                    
                    result[f'GT {rater}'] = y
                    result[f'Pred {rater}'] = predicted_class
                    result['Rating'].append(y)
                    result['Final Pred'].append(predicted_class)
            
            # Final computation for each ID in the current fold
            for result in all_results[currentFold]:
                result['Rating'] = (np.mean(result['Rating'])-1)/3.0
                result['Final Pred'] = (np.mean(result['Final Pred'])-1)/3.0
                result['MSE'] = mean_squared_error([result['Rating']], [result['Final Pred']]).numpy()  #                
    save_to_excel(all_results)
    

if __name__ == "__main__":
    predict_test()
