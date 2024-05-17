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
from metrics import MSE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import random


def scatterPlot(ratings, predictions, currentFold, mse):
    os.makedirs(f"/home/xurbano/QEI-ASL/results/QEI-NET_CNN_ensemble_Classification/plotsValidation/ConfusionMatrix", exist_ok=True)
    plt.scatter(ratings, predictions, color='blue', label='Data points')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Perfect prediction')  # Add a diagonal dashed line for reference 
    plt.xlabel('Ratings')
    plt.ylabel('Predictions')
    plt.title('Scatter Plot of the QEI-ASL')
    plt.legend([f'Predicted QEI (MSE {mse:.4f})','Perfect QEI'])
    save_path = f'/home/xurbano/QEI-ASL/results/QEI-NET_CNN_ensemble_Classification/{rater}/plotsValidation/ConfusionMatrix/{currentFold}.png'
    plt.savefig(save_path)
    plt.close()

def confusionMatrix(ratings, predictions, currentFold,rater):
    os.makedirs(f"/home/xurbano/QEI-ASL/results/QEI-NET_CNN_ensemble_Classification/{rater}/{currentFold}/plotsValidation/ConfusionMatrix", exist_ok=True)
    save_path = f'/home/xurbano/QEI-ASL/results/QEI-NET_CNN_ensemble_Classification/{rater}/{currentFold}/plotsValidation/ConfusionMatrix/{currentFold}.png'
    # Plotting the confusion matrix for the current fold
    cm = confusion_matrix(ratings, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix for Fold {currentFold}")
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(save_path)
    plt.close()


def readAnnotation(ratings,ID):
        matched_row = ratings[ratings.iloc[:, 0] == ID]
        rating = matched_row.iloc[-1, -1]/4.0
        return rating

def scatterPlot(ratings, predictions, currentFold, mse,networkName):
    os.makedirs(f"/home/xurbano/QEI-ASL/results/{networkName}/plotsValidation/ScatterPlot", exist_ok=True)
    plt.scatter(ratings, predictions, color='blue', label='Data points')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Perfect prediction')  # Add a diagonal dashed line for reference 
    plt.xlabel('Ratings')
    plt.ylabel('Predictions')
    plt.title('Scatter Plot of the QEI-ASL')
    plt.legend([f'Predicted QEI (MSE {mse:.4f})','Perfect QEI'])
    save_path = f'/home/xurbano/QEI-ASL/results/{networkName}/plotsValidation/ScatterPlot/{currentFold}.png'
    plt.savefig(save_path)
    plt.close()

        
def compute_mse(y_true, y_pred):
    # Ensure the inputs are tensors. If they're numpy arrays or lists, they'll be converted to tensors.
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    # Calculate the squared differences between the true and predicted values
    squared_differences = tf.square(y_true - y_pred)
    # Compute the mean of the squared differences
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



def compute_MSE():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
    K.clear_session()
    path='/home/xurbano/QEI-ASL/data_v2'
    allRatings=[]
    allPredictions=[]
    rater1,rater2,rater3='Ali', 'JD', 'RW'
    seed=48
    tf.random.set_seed(48)
    np.random.seed(48)
    random.seed(48)
    networkName="QEI-NET_CNN_ensemble_Classification"
    ratings_EXCEL = pd.read_excel("/home/xurbano/QEI-ASL/data_v2/Ratings.xlsx")
    excel_path = f'/home/xurbano/QEI-ASL/results/{networkName}/All_Folds_Predictions.xlsx'
    allMSE=[]
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        with tf.device('/GPU:0'):
            for currentFold in range(1, 6):
                # Parameters of the training
                params = {
                    'pathData': path,
                    'targetSize': (64, 64, 32, 1),
                    'batchSize': 64,
                    'currentFold': currentFold,
                    'rater':'Ali'
                }
                # Configuration of the experiment
                preprocessing = Preprocessing(data_aug=False, norm_intensity=True, imageSize=(64, 64, 32),rater='Ali')
                config = Configuration(**params)
                valIDs = config.returnVal_IDS()
                network = QEI_Net(imgSize=params['targetSize'])
                model1 = network.get_model()
                model1.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
                model1.load_weights(f"/home/xurbano/QEI-ASL/results/QEI-NET_CNN_ensemble_Classification/{rater1}/{currentFold}/Best_Model.keras")
                model2 = network.get_model()
                model2.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
                model2.load_weights(f"/home/xurbano/QEI-ASL/results/QEI-NET_CNN_ensemble_Classification/{rater2}/{currentFold}/Best_Model.keras")
                model3 = network.get_model()
                model3.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
                model3.load_weights(f"/home/xurbano/QEI-ASL/results/QEI-NET_CNN_ensemble_Classification/{rater3}/{currentFold}/Best_Model.keras")
                print(f"\n------------------FOLD {currentFold}------------------")
                ratings = []
                predictions = []
                fold_results = []
                for ID in valIDs:
                    predicted_class=0
                    x, _ = preprocessing.process_case(ID, path)
                    y=readAnnotation(ratings_EXCEL,ID)
                    x = np.expand_dims(x, axis=0)
                    for model in [model1,model2,model3]:
                        prediction = model.predict(x)
                        print(np.argmax(prediction[0])+1)
                        predicted_class += np.argmax(prediction[0])+1
                        
                    ratings.append(y)
                    predicted_class=predicted_class/12
                    predictions.append(predicted_class)
                    mse = compute_mse(y, predicted_class).numpy()  # Make sure to convert tensor to numpy value
                    fold_results.append([ID, y,predicted_class, mse])
                    allMSE.append(mse)  
                        
                # Save fold results to Excel
                save_to_excel(fold_results, networkName, currentFold, writer)
                ratings, predictions, mse_values = [item[1] for item in fold_results], [item[2] for item in fold_results], [item[3] for item in fold_results]
                scatterPlot(ratings, predictions, currentFold, np.mean(mse_values), networkName)
                
    
        # Calculate and print overall accuracy
        print(f"Prediction and MSE results saved for each fold in the '{networkName}' directory.")
        print(f"MEAN AVG MSE: {np.mean(allMSE)}")
            
            
if __name__ == "__main__":
    compute_MSE()