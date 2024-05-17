import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy
from scipy.stats import pearsonr,kurtosis, entropy
import glob
import sys
sys.path.append('/home/xurbano/QEI-ASL/src/')
from configuration import Configuration
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sympy import symbols, exp
import statistics
from scipy.stats import gmean

class Features_QEI:
    def __init__(self, path):
        self.path=path

    def readNiftiFile(self, path):
        return nib.load(path).get_fdata()
 
        
    def readImages(self,imagePath):
        self.CBF=self.readNiftiFile(os.path.join(imagePath,"CBF_Map_smoothed.nii"))
        self.GM_prob=self.readNiftiFile(os.path.join(imagePath,"GM_prob.nii"))
        self.WM_prob=self.readNiftiFile(os.path.join(imagePath,"WM_prob.nii"))
        self.CSF_prob=self.readNiftiFile(os.path.join(imagePath,"CSF_prob.nii"))
        self.GM_Mask=np.where(self.GM_prob>0.9,1,0)
        self.CSF_Mask=np.where(self.CSF_prob>0.9,1,0)
        self.WM_Mask=np.where(self.WM_prob>0.9,1,0)
        
#Structural_Similarity	Negative_GM_CBF	Spatial_Variability
    
    def Structural_Similarity(self):
        spCBF=self.GM_prob*2.5 + self.WM_prob
        mask_non_zero_CBF = self.CBF != 0
        mask_non_nan_CBF = ~np.isnan(self.CBF)
        mask_non_nan_spCBF = ~np.isnan(spCBF)
        final_mask = mask_non_zero_CBF & mask_non_nan_CBF & mask_non_nan_spCBF
        Pearson_Correlation,_=pearsonr(spCBF[final_mask].flatten(),self.CBF[final_mask].flatten())
        if Pearson_Correlation<0: return 0
        else: return Pearson_Correlation
        
    def Spatial_Variability(self):
        numberGM_voxels, numberWM_voxels, numberCSF_voxels = np.sum(self.GM_Mask.flatten()), np.sum(self.WM_Mask.flatten()), np.sum(self.CSF_Mask.flatten())
        varGM, varWM, varCSF = np.var(self.CBF[self.GM_Mask==1].flatten()), np.var(self.CBF[self.WM_Mask==1].flatten()), np.var(self.CBF[self.CSF_Mask==1].flatten())
        pooledVariance = ((numberGM_voxels-1)*varGM + (numberWM_voxels-1)*varWM + (numberCSF_voxels-1)*varCSF) / (numberGM_voxels + numberWM_voxels + numberCSF_voxels -3)
        CV=pooledVariance/abs(np.mean(self.CBF[self.GM_Mask==1].flatten()))
        return CV

    def Negative_GM_CBF(self):
        totalVoxels=np.sum(self.GM_Mask==1)
        negativeVoxels=np.sum(self.CBF[self.GM_Mask==1]<0)
        Percentage_Negative_GM=(negativeVoxels/totalVoxels)
        return Percentage_Negative_GM
    
    def computeFeatures(self,ID):
        # Call all feature extraction methods
        structural_similarity = self.Structural_Similarity()
        negative_GM = self.Negative_GM_CBF()
        spatial_variability = self.Spatial_Variability()
        features = [ID, structural_similarity, negative_GM, spatial_variability]
        return features

    def normalizeFeatures(self, df):
        features = df.drop(columns=['ID'])
        # Ensure all feature columns are numeric
        features = features.apply(pd.to_numeric, errors='coerce')
        # Initialize MinMaxScaler
        scaler = MinMaxScaler()
        # Fit and transform the features
        features_normalized_array = scaler.fit_transform(features)
        # Convert back to DataFrame
        features_normalized = pd.DataFrame(features_normalized_array, columns=features.columns)
        # Add the 'ID' column back
        df_normalized = pd.concat([df[['ID']].reset_index(drop=True), features_normalized.reset_index(drop=True)], axis=1)
        return df_normalized
        
    def execute(self, listIDs):
        allFeatures=[]
        for ID in listIDs:
            newImagePath=os.path.join(self.path, ID)
            self.readImages(newImagePath)
            allFeatures.append(self.computeFeatures(ID))
        # Write to Excel
        df = pd.DataFrame(allFeatures, columns=["ID", "Structural_Similarity", "Negative_GM_CBF", 
                                                  "Spatial_Variability"])
        # Preprocess features
        df_normalized = self.normalizeFeatures(df)
        
        df_normalized.to_excel("/home/xurbano/QEI-ASL/src_3_features_Sudipto/computed_features.xlsx", index=False)
        return df_normalized

if __name__ == "__main__":
    imgPath = '/home/xurbano/QEI-ASL/data_final'
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
    print(features.head())
    
    