# -----------------------------------------------------------------------------
# Main file
# Author: Xavier Beltran Urbano 
# Date Created: 21-02-2024
# -----------------------------------------------------------------------------

import os
from configuration import Configuration
import os
from features_QEI import Features_QEI
from sympy import symbols, exp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from predict_Test import predict_test
import tensorflow as tf
import random

# Functions to compute QEI
def fun1(x, xdata):
    return np.exp(-x[0] * (xdata) ** x[1])
    
def fun2(x, xdata):
    return 1 - np.exp(-x[0] * (xdata) ** x[1])
    
def QE_formula(p_ss, D, p_nGMCBF, x1,x2,x3):
    return (fun1(x1,D)*fun1(x2,p_nGMCBF)*fun2(x3,p_ss))**(1/3)

def scatterPlot(ratings, predictions,path):
    plt.scatter(ratings, predictions, color='blue', label='Data points')
    plt.plot([0, 1], [0, 1], color='red')#, label='Line from (0,0) to (4,4)')
    plt.xlabel('Ratings')
    plt.ylabel('Predictions')
    plt.title('Scatter Plot of the QEI-ASL')
    save_path = f'{path}/ScatterPlot.png'
    plt.savefig(save_path)
    plt.close()
    
def run_program(config,networkName, params, features):
    # Define the symbols
    p_ss, D, p_nGMCBF = symbols('p_ss D p_nGMCBF')
    # Predefined parameters
    x1 = [0.0544, 0.9272]
    x2 = [2.8478, 0.5196]
    x3 = [3.0126, 2.4419]
    # Compute QEI
    features['QEI_Sudipto'] = features.apply(lambda row: QE_formula(row['Structural_Similarity'], row['Spatial_Variability'], row['Negative_GM_CBF'], x1,x2,x3), axis=1)
    # Compare the results with the ratings
    ratings = pd.read_excel('/home/xurbano/QEI-ASL/data_v2/Ratings.xlsx')
    merged_df = pd.merge(features, ratings, left_on="ID", right_on="IDS", how="left")
    features['Ratings'] = merged_df.iloc[:, -1]/4.00
    # Calculate Mean Squared Error (MSE) of the QEI and the ratings
    mse = ((features.iloc[:, 4] - features.iloc[:, 5]) ** 2)
    features['MSE'] = mse
    #print(f" Mean of the MSE {np.mean(features['MSE'])}")
    # Save the updated dataframe to the first Excel file
    features.to_excel("/home/xurbano/QEI-ASL/src_3_features_Sudipto/computed_QEI.xlsx", index=False)
    scatterPlot(features['Ratings'],features['QEI_Sudipto'],f"/home/xurbano/QEI-ASL/results/{networkName}")
    
if __name__ == "__main__":
    imgPath = '/home/xurbano/QEI-ASL/data_v2'
    networkName="QEI-NET_3_features_Sudipto"
    tf.random.set_seed(48)
    np.random.seed(48)
    random.seed(48)
    # Create folder for this experiment
    os.makedirs(f"/home/xurbano/QEI-ASL/results/{networkName}", exist_ok=True)
    QEI_features=Features_QEI(imgPath)
    params={
            'pathData':imgPath,
            'targetSize':(64,64,32,1),
            'batchSize':5,
            'currentFold':1
        }
    # Configuration of the experiment
    config=Configuration(**params)
    features=QEI_features.execute(config.returnIDS())
    run_program(config,networkName,params,features)
    predict_test()
