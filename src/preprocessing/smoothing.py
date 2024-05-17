import itk
import numpy as np
import os
from tqdm import tqdm  # Make sure tqdm is installed

class Smoothing:
    def __init__(self, fwhm=5):
        self.fwhm = fwhm

    def apply_gaussian_smoothing(self, input_image_path, output_image_path):
        sigma_mm = self.fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        image = itk.imread(input_image_path, itk.F)
        smoothed_image = itk.smoothing_recursive_gaussian_image_filter(image, sigma=sigma_mm)
        itk.imwrite(smoothed_image, output_image_path)
        print(f"Smoothed image saved to {output_image_path}")

if __name__ == "__main__":
    path = '/home/xurbano/QEI-ASL/data_v2'  # Adjust this to your input directory
    smoothing = Smoothing(fwhm=5)  # You can adjust the FWHM as needed
    fileNames = [file for file in os.listdir(path) if not file.endswith(('xlsx', '.DS_Store', 'xls', '.ipynb_checkpoints'))]
    
    # Use tqdm here to show progress
    for file_name in tqdm(fileNames, desc="Processing images"):
        input_path = os.path.join(path, file_name, "CBF_Map.nii")
        output_path = os.path.join(path, file_name, "CBF_Map_smoothed.nii")
        smoothing.apply_gaussian_smoothing(input_path, output_path)
