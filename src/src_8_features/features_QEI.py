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

class Features_QEI:
    def __init__(self, path):
        self.path=path

    def readNiftiFile(self, path):
        return nib.load(path).get_fdata()

    def readImages(self,imagePath):
        self.sCBF=self.readNiftiFile(os.path.join(imagePath,"CBF_Map_smoothed.nii"))
        self.CBF=self.readNiftiFile(os.path.join(imagePath,"CBF_Map.nii"))
        self.GM_prob=self.readNiftiFile(os.path.join(imagePath,"GM_prob.nii"))
        self.WM_prob=self.readNiftiFile(os.path.join(imagePath,"WM_prob.nii"))
        self.CSF_prob=self.readNiftiFile(os.path.join(imagePath,"CSF_prob.nii"))
        self.GM_Mask=np.where(self.GM_prob>0.9,1,0)
        self.CSF_Mask=np.where(self.CSF_prob>0.9,1,0)
        self.WM_Mask=np.where(self.WM_prob>0.9,1,0)
        
    def SNR(self):
        mean_GM=np.mean(self.CBF[self.GM_Mask==1])
        std_CSF=np.std(self.CBF[self.CSF_Mask==1])
        SNR=mean_GM/std_CSF
        return SNR

    def Structural_Similarity(self):
        spCBF=self.GM_prob*2.5 + self.WM_prob
        mask_non_zero_CBF = self.sCBF != 0
        mask_non_nan_CBF = ~np.isnan(self.sCBF)
        mask_non_nan_spCBF = ~np.isnan(spCBF)
        final_mask = mask_non_zero_CBF & mask_non_nan_CBF & mask_non_nan_spCBF
        Pearson_Correlation,_=pearsonr(spCBF[final_mask].flatten(),self.sCBF[final_mask].flatten())
        if Pearson_Correlation<0: return 0
        else: return Pearson_Correlation
        
    def Spatial_Variability(self):
        numberGM_voxels, numberWM_voxels, numberCSF_voxels = np.sum(self.GM_Mask.flatten()), np.sum(self.WM_Mask.flatten()), np.sum(self.CSF_Mask.flatten())
        varGM, varWM, varCSF = np.var(self.sCBF[self.GM_Mask==1].flatten()), np.var(self.sCBF[self.WM_Mask==1].flatten()), np.var(self.sCBF[self.CSF_Mask==1].flatten())
        pooledVariance = ((numberGM_voxels-1)*varGM + (numberWM_voxels-1)*varWM + (numberCSF_voxels-1)*varCSF) / (numberGM_voxels + numberWM_voxels + numberCSF_voxels -3)
        CV=pooledVariance/abs(np.mean(self.sCBF[self.GM_Mask==1].flatten()))
        return CV

    def Negative_GM_CBF(self):
        totalVoxels=np.sum(self.GM_Mask==1)
        negativeVoxels=np.sum(self.sCBF[self.GM_Mask==1]<0)
        Percentage_Negative_GM=(negativeVoxels/totalVoxels)
        return Percentage_Negative_GM

    def ASL_CBF_Heterogeneity(self):
        return np.std(self.CBF[self.GM_Mask==1])/np.mean(self.CBF[self.GM_Mask==1])

    def Image_Statistics(self, mask):
        mean=np.mean(self.CBF[mask==1])
        inverseStd= 1 / np.std(self.CBF[mask==1])
        five_Percentile=np.percentile(self.CBF[mask==1], 5)
        ninetyFive_Percentile= np.percentile(self.CBF[mask==1], 95)
        kurtosis_metric= kurtosis(self.CBF[mask==1])
        return mean,inverseStd,five_Percentile,ninetyFive_Percentile,kurtosis_metric

    def Shanon_Entropy(self):
        values, counts = np.unique(self.CBF, return_counts=True)
        probabilities = counts / counts.sum()
        Shanon_Entropy=entropy(probabilities, base=2)
        if Shanon_Entropy > 0:
            return 1 / Shanon_Entropy
        else:
            return 0

        
    def Spatial_Gradients(self):
        # Compute the Gradients
        grad_x, grad_y, grad_z = np.gradient(self.CBF, axis=(0, 1, 2))
        #Inverse the gradients 
        invGrad_x = 1 / np.where(grad_x != 0, grad_x, np.nan)
        invGrad_y = 1 / np.where(grad_y != 0, grad_y, np.nan)
        invGrad_z = 1 / np.where(grad_z != 0, grad_z, np.nan)
        return np.nanvar(invGrad_x),np.nanvar(invGrad_y),np.nanvar(invGrad_z)

    def computeFeatures(self,ID):
        # Call all feature extraction methods
        SNR_value = self.SNR()
        structural_similarity = self.Structural_Similarity()
        spatial_variability=self.Spatial_Variability()
        asl_cbf_heterogeneity = self.ASL_CBF_Heterogeneity()
        negative_GM = self.Negative_GM_CBF()
        image_statistics = self.Image_Statistics(self.GM_Mask)
        shanon_entropy = self.Shanon_Entropy()
        spatial_gradients = self.Spatial_Gradients()
        features = [ID, SNR_value, structural_similarity,spatial_variability, asl_cbf_heterogeneity, 
                    negative_GM, *image_statistics, shanon_entropy, *spatial_gradients]
        return features

    def normalizeFeatures(self, df):
        # Assuming 'ID' is not part of the features to normalize
        features_to_normalize = df.iloc[:, -3:]  # Select only the last 3 columns
        features_remaining = df.iloc[:, :-3]  # Select all columns except the last 3
    
        # Initialize MinMaxScaler
        scaler = MinMaxScaler()
        # Fit and transform the features to normalize
        features_normalized_array = scaler.fit_transform(features_to_normalize)
        
        # Convert back to DataFrame
        features_normalized = pd.DataFrame(features_normalized_array, columns=features_to_normalize.columns)
        
        # Update the original dataframe with the normalized values for the last 3 columns
        df.update(features_normalized)
        
        return df
        
    def execute(self, listIDs):
        allFeatures=[]
        for ID in listIDs:
            newImagePath=os.path.join(self.path, ID)
            self.readImages(newImagePath)
            allFeatures.append(self.computeFeatures(ID))
        # Write to Excel
        df = pd.DataFrame(allFeatures, columns=["ID", "SNR", "Structural_Similarity", "Spatial_Variability", "ASL_CBF_Heterogeneity", 
                                                  "Negative_GM_CBF", "Mean", "Inverse_Std", 
                                                  "5th_Percentile", "95th_Percentile", "Kurtosis", 
                                                  "Shanon_Entropy", "Spatial_Gradient_X", 
                                                  "Spatial_Gradient_Y", "Spatial_Gradient_Z"])
        # Preprocess features
        df_normalized = self.normalizeFeatures(df)
        
        df_normalized.to_excel("/home/xurbano/QEI-ASL/src_8_features/computed_features.xlsx", index=False)
        return df_normalized

if __name__ == "__main__":
    imgPath = '/home/xurbano/QEI-ASL/data_final'
    QEI_features=Features_QEI(imgPath)
    params={
            'pathData':imgPath,
            'targetSize':(64,64,25,1),
            'batchSize':5,
            'currentFold':1
        }
    # Configuration of the experiment
    config=Configuration(**params)
    features=QEI_features.execute(config.returnIDS())
    print(features.head())
    
    