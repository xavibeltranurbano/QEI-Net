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
from scipy.ndimage import rotate

class Preprocessing():
    def __init__(self,data_aug,norm_intensity,imageSize):
        self.dataAug = data_aug
        self.normIntensity = norm_intensity
        self.height, self.width, self.totalSlices =imageSize[0],imageSize[1],imageSize[2]
        self.ratings = pd.read_excel("/home/xurbano/QEI-ASL/data_final/Ratings.xlsx")

    def dataAugmentation(self, images):
        augmented_img = []
        rand_flip1 = np.random.randint(0, 2)
        rand_flip2 = np.random.randint(0, 2)
        #rand_rotate = np.random.randint(0, 4) 
        #angle = np.random.uniform(-5, 5)  
        for img in images: 
            if rand_flip1 == 1: img = np.flip(img, 0)
            if rand_flip2 == 1: img = np.flip(img, 1)
            #if rand_rotate == 1: img = rotate(img, angle, reshape=False, mode='nearest')
            augmented_img.append(img)
        return np.asarray(augmented_img)
        
    
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
        matched_row = self.ratings[self.ratings.iloc[:, 0] == ID]
        rating = (matched_row.iloc[-1, -1]-1.0)/3.0
        return rating
    
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

        #img = np.expand_dims(img, axis=-1)  
        #img_smoothed = np.expand_dims(img_smoothed, axis=-1)  
        #img_final=np.concatenate([img,img_smoothed], axis=-1)
        # Save the processed and original images
        #self.saveImage(np.asarray(img), "/Users/xavibeltranurbano/Desktop/UPenn/QEI-PROEJCT/results/results.nii")  # Processed image
        #self.saveImage(self.readImage(path_img), "/Users/xavibeltranurbano/Desktop/UPenn/QEI-PROEJCT/results/original.nii")  # Original image
        return np.asarray(img), rating
    

if __name__ == "__main__":
    path='/home/xurbano/QEI-ASL/data_final'
    fileNames=[file for file in os.listdir(path) if not file.endswith('xlsx') and  not file.endswith('.DS_Store') and  not file.endswith('.xls')]
    prepro=Preprocessing(data_aug=True, norm_intensity=True, imageSize=(64,64,30))
    img,rating=prepro.process_case(fileNames[0],path)
    plt.imshow(img[:,:,15,0])
    plt.savefig("Normal.png")
    plt.imshow(img[:,:,15,1])
    plt.savefig("mask.png")
    print(f"Shape {img.shape}")
    print(f"Rating {rating}")
