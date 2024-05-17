# -----------------------------------------------------------------------------
# Preprocessing  file
# Author: Xavier Beltran Urbano
# Date Created: 22-02-2024
# -----------------------------------------------------------------------------


import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import scipy
import cv2

class Preprocessing():
    def __init__(self,data_aug,norm_intensity,imageSize,rater):
        self.dataAug = data_aug
        self.normIntensity = norm_intensity
        self.height, self.width, self.totalSlices =imageSize[0],imageSize[1],imageSize[2]
        self.rater=rater
        self.ratings = pd.read_excel("/home/xurbano/QEI-ASL/data_final/Ratings.xlsx")

    def dataAugmentation(self, images):
        augmented_img=[]
        rand_flip1 =  np.random.randint(0, 2)
        rand_flip2 = np.random.randint(0, 2)
        for img in images:
            if rand_flip1 == 1: img = np.flip(img, 0)
            if rand_flip2 == 1: img = np.flip(img, 1)
            augmented_img.append(img)
        return augmented_img
    
    def maskBackground(self,img, mask_value=0):
        mask = img == mask_value
        return mask

    def normalizeIntensity(self,img):
        #backgroundMask=self.maskBackground(img)
        image_min, image_max = 0, 1
        normalizedImage = ((img - np.min(img)) / (np.max(img) - np.min(img))) * (image_max - image_min) + image_min
        #normalizedImage[backgroundMask] = 0
        return normalizedImage
    
    def resizeImage(self,img):  
        image_resized_spatially = scipy.ndimage.zoom(img, (self.height/img.shape[0], self.width/img.shape[1], 1), order=2) # Resizing spatial dimensions
        image_resized = scipy.ndimage.zoom(image_resized_spatially, (1, 1, self.totalSlices/img.shape[2]), order=2) # Adjusting z-dimension
        return image_resized

    def readAnnotation(self, ID):
        # Assuming the first column contains the IDs and each subsequent column contains ratings from different raters
        matched_row = self.ratings[self.ratings.iloc[:, 0] == ID]
        
        # Assuming the rater's name matches exactly with the column name in the ratings DataFrame
        rating = matched_row[self.rater].values[0]
        
        return int(rating)

    
    def saveImage(self, img, save_path):
        new_image = nib.Nifti1Image(img, np.eye(4))  # Create a NIfTI image, assuming no affine transformation
        nib.save(new_image, save_path)
        
    def readNiftiFile(self, path):
        return nib.load(path).get_fdata()
        
    def process_case(self,ID,path):
        img=self.readNiftiFile(os.path.join(path, ID,"CBF_Map_Reg.nii"))
        rating=self.readAnnotation(ID)
        # Normalize intensities
        if self.normIntensity:
            img=self.normalizeIntensity(img)
        # Data augmentation
        #if self.dataAug==True:
        #    img=self.dataAugmentation(img)
        # Save the processed and original images
        #self.saveImage(np.asarray(img), "/Users/xavibeltranurbano/Desktop/UPenn/QEI-PROEJCT/results/results.nii")  # Processed image
        #self.saveImage(self.readImage(path_img), "/Users/xavibeltranurbano/Desktop/UPenn/QEI-PROEJCT/results/original.nii")  # Original image

        return np.asarray(img), rating
    

if __name__ == "__main__":
    path='/home/xurbano/QEI-ASL/data'
    fileNames=[file for file in os.listdir(path) if not file.endswith('xlsx') and  not file.endswith('.DS_Store') and  not file.endswith('.xls')]
    prepro=Preprocessing(data_aug=True, norm_intensity=True, imageSize=(64,64,25), rater="Ali")
    img,rating=prepro.process_case(fileNames[0],path)
    print(f"Shape {img.shape}")
    print(f"Rating {rating}")
