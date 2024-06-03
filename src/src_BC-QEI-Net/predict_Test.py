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
from tensorflow.keras.metrics import AUC
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix


def compute_metrics(y_true, y_pred):
    #Compute metrics for the val data
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    youden_index = tpr - fpr
    best_threshold = thresholds[np.argmax(youden_index)]
    y_pred_binary = (y_pred >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    best_fpr = fpr[np.argmax(youden_index)]
    best_tpr = tpr[np.argmax(youden_index)]
    return sensitivity, specificity, youden_index.max(), auc_score, best_threshold, fpr, tpr, best_fpr, best_tpr

def plot_roc_curve(y_true, y_pred, networkName):
    os.makedirs(f"/results/{networkName}/plotsValidation/ROC_Curve", exist_ok=True)
    sensitivity, specificity, youden_index, auc_score, best_threshold, fpr, tpr, best_fpr, best_tpr = compute_metrics(y_true, y_pred)
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.scatter(best_fpr, best_tpr, color='green', label=f'Youden Index (FPR={best_fpr:.2f}, TPR={best_tpr:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    save_path = f'/results/{networkName}/plotsValidation/ROC_Curve/AllFolds.png'
    plt.savefig(save_path)
    plt.close()
    return auc_score

def reorder_columns(df):
    # Reorder columns to move specific columns to the end
    cols_to_move = ['Rating', 'Prediction', 'Youden Index', 'Sensitivity', 'Specificity', 'AUC']
    for col in cols_to_move:
        if col not in df.columns:
            print(f"DataFrame does not have the required column: {col}")
            return df

    cols = [col for col in df.columns if col not in cols_to_move]
    new_order = cols + cols_to_move
    return df[new_order]

def save_to_excel(fold_results, networkName, writer):
    # Create DataFrame from the fold results
    df = pd.DataFrame(fold_results, columns=['Name', 'Rating', 'Prediction', 'Youden Index', 'Sensitivity', 'Specificity', 'AUC'])
    mean_metrics = {
        'Name': 'Mean Metrics',
        'Rating': None,
        'Prediction': None,
        'Youden Index': df['Youden Index'].mean(),
        'Sensitivity': df['Sensitivity'].mean(),
        'Specificity': df['Specificity'].mean(),
        'AUC': df['AUC'].mean()
    }
    mean_metrics_df = pd.DataFrame([mean_metrics])
    df_final = pd.concat([df, mean_metrics_df], ignore_index=True)
    df_final.to_excel(writer, sheet_name='AllFolds', index=False)

    
def predict_test(networkName):
    # Predict the test set
    seed = 48
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    K.clear_session()
    path = '/data_final'
    all_fold_results = []
    all_y_true = []
    all_y_pred = []
    # Initialize the ExcelWriter for the single Excel file
    excel_path = f'/results/{networkName}/All_Folds_Predictions.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        with tf.device('/GPU:0'):
            for currentFold in range(1, 6):
                params = {
                    'pathData': path,
                    'targetSize': (64, 64, 32, 1),
                    'batchSize': 30,
                    'currentFold': currentFold
                }
                preprocessing = Preprocessing(data_aug=False, norm_intensity=True, imageSize=(64, 64, 32))
                config = Configuration(**params)
                valIDs = config.returnVal_IDS()
                network = MSC_QEI_Net(imgSize=params['targetSize'])
                model = network.get_model()
                model.load_weights(f"/results/{networkName}/{currentFold}/Best_Model.keras")

                fold_results = []
                for ID in valIDs:
                    x, y = preprocessing.process_case(ID, path)
                    x = np.expand_dims(x, axis=0)
                    prediction = model.predict(x).flatten()
                    all_y_true.append(y)
                    all_y_pred.append(prediction)
                    fold_results.append([ID, y, prediction, None, None, None, None])
                
                all_fold_results.extend(fold_results)
        
        # Compute metrics and plots after concatenating all fold data
        sensitivity, specificity, youden_index, auc_score, best_threshold, fpr, tpr, best_fpr, best_tpr = compute_metrics(all_y_true, all_y_pred)

        # Update fold results with computed metrics
        for result in all_fold_results:
            result[3] = youden_index
            result[4] = sensitivity
            result[5] = specificity
            result[6] = auc_score

        # Save all fold results to Excel
        save_to_excel(all_fold_results, networkName, writer)
        
        # Generate ROC curve plot
        plot_roc_curve(all_y_true, all_y_pred, networkName)

        print(f"Mean Sensitivity: {sensitivity:.4f}, Mean Specificity: {specificity:.4f}, Mean Youden Index: {youden_index:.4f}, Mean AUC: {auc_score:.4f}")

if __name__ == "__main__":
    predict_test("BC-QEI-Net")