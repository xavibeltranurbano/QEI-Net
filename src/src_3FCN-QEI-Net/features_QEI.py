# -----------------------------------------------------------------------------
# Feature Generator for 3FCN-QEI-Net
# Author: Xavier Beltran Urbano
# Date Created: 21-02-2024
# -----------------------------------------------------------------------------


import nibabel as nib
import numpy as np
import os
from scipy.stats import pearsonr, kurtosis, entropy
from configuration import Configuration
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Features_QEI:
    def __init__(self, path):
        # Initialize the class with the path to the data
        self.path = path

    def readNiftiFile(self, path):
        # Read and return NIFTI file data
        return nib.load(path).get_fdata()

    def readImages(self, imagePath):
        # Read all required image files and create masks
        self.CBF = self.readNiftiFile(os.path.join(imagePath, "CBF_Map_smoothed.nii"))
        self.GM_prob = self.readNiftiFile(os.path.join(imagePath, "GM_prob.nii"))
        self.WM_prob = self.readNiftiFile(os.path.join(imagePath, "WM_prob.nii"))
        self.CSF_prob = self.readNiftiFile(os.path.join(imagePath, "CSF_prob.nii"))
        self.GM_Mask = np.where(self.GM_prob > 0.9, 1, 0)
        self.CSF_Mask = np.where(self.CSF_prob > 0.9, 1, 0)
        self.WM_Mask = np.where(self.WM_prob > 0.9, 1, 0)

    def Structural_Similarity(self):
        # Calculate and return structural similarity
        spCBF = self.GM_prob * 2.5 + self.WM_prob
        mask_non_zero_CBF = self.CBF != 0
        mask_non_nan_CBF = ~np.isnan(self.CBF)
        mask_non_nan_spCBF = ~np.isnan(spCBF)
        final_mask = mask_non_zero_CBF & mask_non_nan_CBF & mask_non_nan_spCBF
        Pearson_Correlation, _ = pearsonr(spCBF[final_mask].flatten(), self.CBF[final_mask].flatten())
        if Pearson_Correlation < 0:
            return 0
        else:
            return Pearson_Correlation

    def Spatial_Variability(self):
        # Calculate and return spatial variability
        numberGM_voxels = np.sum(self.GM_Mask.flatten())
        numberWM_voxels = np.sum(self.WM_Mask.flatten())
        numberCSF_voxels = np.sum(self.CSF_Mask.flatten())
        varGM = np.var(self.CBF[self.GM_Mask == 1].flatten())
        varWM = np.var(self.CBF[self.WM_Mask == 1].flatten())
        varCSF = np.var(self.CBF[self.CSF_Mask == 1].flatten())
        pooledVariance = ((numberGM_voxels - 1) * varGM + (numberWM_voxels - 1) * varWM + (
                    numberCSF_voxels - 1) * varCSF) / (numberGM_voxels + numberWM_voxels + numberCSF_voxels - 3)
        CV = pooledVariance / abs(np.mean(self.CBF[self.GM_Mask == 1].flatten()))
        return CV

    def Negative_GM_CBF(self):
        # Calculate and return percentage of negative GM CBF
        totalVoxels = np.sum(self.GM_Mask == 1)
        negativeVoxels = np.sum(self.CBF[self.GM_Mask == 1] < 0)
        Percentage_Negative_GM = (negativeVoxels / totalVoxels)
        return Percentage_Negative_GM

    def computeFeatures(self, ID):
        # Compute and return features for a given ID
        structural_similarity = self.Structural_Similarity()
        negative_GM = self.Negative_GM_CBF()
        spatial_variability = self.Spatial_Variability()
        features = [ID, structural_similarity, negative_GM, spatial_variability]
        return features

    def normalizeFeatures(self, df):
        # Normalize the features in the dataframe
        features = df.drop(columns=['ID'])
        features = features.apply(pd.to_numeric, errors='coerce')
        scaler = MinMaxScaler()
        features_normalized_array = scaler.fit_transform(features)
        features_normalized = pd.DataFrame(features_normalized_array, columns=features.columns)
        df_normalized = pd.concat([df[['ID']].reset_index(drop=True), features_normalized.reset_index(drop=True)],
                                  axis=1)
        return df_normalized

    def execute(self, listIDs):
        # Execute feature extraction for a list of IDs and save to Excel
        allFeatures = []
        for ID in listIDs:
            newImagePath = os.path.join(self.path, ID)
            self.readImages(newImagePath)
            allFeatures.append(self.computeFeatures(ID))
        # Write to Excel
        df = pd.DataFrame(allFeatures, columns=["ID", "Structural_Similarity", "Negative_GM_CBF",
                                                "Spatial_Variability"])

        df.to_excel("/home/xurbano/QEI-ASL/src_3_features/computed_features.xlsx", index=False)
        return df


if __name__ == "__main__":
    imgPath = '/home/xurbano/QEI-ASL/data_final'
    QEI_features = Features_QEI(imgPath)
    params = {
        'pathData': imgPath,
        'targetSize': (64, 64, 25, 1),
        'batchSize': 5,
        'currentFold': 1
    }
    # Configuration of the experiment
    config = Configuration(**params)
    features = QEI_features.execute(config.returnIDS())
    print(features.head())
