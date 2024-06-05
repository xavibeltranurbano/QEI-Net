# -----------------------------------------------------------------------------
# Preprocessing file
# Author: Xavier Beltran Urbano
# Date Created: 22-02-2024
# -----------------------------------------------------------------------------

import os
import nibabel as nib
import numpy as np
import pandas as pd
import scipy
from scipy.ndimage import rotate


class Preprocessing():
    def __init__(self, data_aug, norm_intensity, imageSize):
        # Initialize the preprocessing class with given parameters
        self.dataAug = data_aug
        self.normIntensity = norm_intensity
        self.height, self.width, self.totalSlices = imageSize[0], imageSize[1], imageSize[2]
        self.ratings = pd.read_excel("/data_final/Ratings.xlsx")

    def dataAugmentation(self, images):
        # Perform data augmentation on the images
        augmented_img = []
        rand_flip1 = np.random.randint(0, 2)
        rand_flip2 = np.random.randint(0, 2)
        rand_rotate = np.random.randint(0, 4) 
        angle = np.random.uniform(-5, 5)  
        for img in images: 
            if rand_flip1 == 1: img = np.flip(img, 0)
            if rand_flip2 == 1: img = np.flip(img, 1)
            if rand_rotate == 1: img = rotate(img, angle, reshape=False, mode='nearest')
            augmented_img.append(img)
        return np.asarray(augmented_img)

    def normalizeIntensity(self, img):
        # Define the range for normalization
        image_min, image_max = 0, 1
        img = np.clip(img, -10, 80)
        background_mask = (img == 0)
        img_non_background = img[~background_mask]
        min_val = np.min(img_non_background)
        max_val = np.max(img_non_background)
        normalized_non_background = ((img_non_background - min_val) / (max_val - min_val)) * (
                image_max - image_min) + image_min
        epsilon = 1e-5
        normalized_non_background = np.where(normalized_non_background == 0, epsilon, normalized_non_background)
        normalized_img = np.zeros_like(img)
        normalized_img[~background_mask] = normalized_non_background
        return normalized_img

    def readAnnotation(self, ID):
        matched_row = self.ratings[self.ratings.iloc[:, 0] == ID]
        rating = (matched_row.iloc[-1, -1] - 1.0) / 3.0
        return rating

    def readNiftiFile(self, path):
        # Read and return NIFTI file data
        return nib.load(path).get_fdata()

    def process_case(self, ID, path):
        # Process a single case (image)
        img = self.readNiftiFile(os.path.join(path, ID, "CBF_Map_Reg.nii"))
        rating = self.readAnnotation(ID)
        # Normalize intensities
        if self.normIntensity:
            img = self.normalizeIntensity(img)
        if self.dataAug:
            img=self.dataAugmentation(img)
            
        return np.asarray(img), rating
