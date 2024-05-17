import pandas as pd
import numpy as np
from sympy import symbols, exp
import matplotlib.pyplot as plt

def scatterPlot(ratings, predictions):
    plt.scatter(ratings, predictions, color='blue', label='Data points')
    plt.plot([0, 1], [0, 1], color='red')#, label='Line from (0,0) to (4,4)')
    plt.xlabel('Ratings')
    plt.ylabel('Predictions')
    plt.title('Scatter Plot of the QEI-ASL')
    save_path = f'/home/xurbano/QEI-ASL/src_3_features/ScatterPlot.png'
    plt.savefig(save_path)
    plt.close()
    
# Define the symbols
p_ss, D, p_nGMCBF = symbols('p_ss D p_nGMCBF')

def fun1(x, xdata):
    return np.exp(-x[0] * (xdata) ** x[1])

def fun2(x, xdata):
    return 1 - np.exp(-x[0] * (xdata) ** x[1])
    
# Define the QE function using the formula provided
def QE_formula(p_ss, D, p_nGMCBF, x1,x2,x3):
    return (fun1(x1,D)*fun1(x2,p_nGMCBF)*fun2(x3,p_ss))**(1/3)

# Load data from Excel
data = pd.read_excel('/home/xurbano/QEI-ASL/src_3_features_Sudipto/computed_features.xlsx')

x1 = [0.0544, 0.9272]
x2 = [2.8478, 0.5196]
x3 = [3.0126, 2.4419]

# Compute QE for each row and store the results in a new column 'QE'
data['QEI_Sudipto'] = data.apply(lambda row: QE_formula(row['Structural_Similarity'], row['Spatial_Variability'], row['Negative_GM_CBF'], x1,x2,x3), axis=1)

df2 = pd.read_excel('/home/xurbano/QEI-ASL/data_final/Ratings.xlsx')
# Merge the two dataframes on the ID column
merged_df = pd.merge(data, df2, left_on="ID", right_on="IDS", how="left")

print(data)
# Update the first Excel file with the value from the last column of the second Excel file
data['Ratings'] = (merged_df.iloc[:, -1] -1)/3.00
print(data)
# Calculate Mean Squared Error (MSE) between column 5 and column 6
mse = ((data.iloc[:, 4] - data.iloc[:, 5]) ** 2)

# Add a new column with the calculated MSE to the DataFrame
data['MSE'] = mse
print(np.mean(data['MSE']))
# Save the updated dataframe to the first Excel file
data.to_excel("/home/xurbano/QEI-ASL/src_3_features_Sudipto/computed_QEI.xlsx", index=False)

scatterPlot(data['QEI_Sudipto'],data['Ratings'])