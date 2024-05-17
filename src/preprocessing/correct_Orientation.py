# -----------------------------------------------------------------------------
# Orientation Correction file
# Author: Xavier Beltran Urbano
# Date Created: 21-02-2024
# -----------------------------------------------------------------------------

import nibabel as nib
import numpy as np
import os


class Orientation_Corrector:
    def __init__(self, input_path, output_suffix='_CO.nii'):
        # Initialize the class with the input path and output suffix
        self.input_path = input_path
        self.output_suffix = output_suffix

    def readNiftiFile(self, file_path):
        # Load and return NIFTI file data
        return nib.load(file_path).get_fdata()

    def process_nifti_image(self, file_path, output_path):
        # Process the NIFTI image to correct orientation
        data = self.readNiftiFile(file_path)
        binary_data = (data != 0).astype(int)
        half_index = binary_data.shape[1] // 2  # Determine the dividing index for the two halves of the image
        lower_half_count = np.sum(
            binary_data[:, :half_index, :])  # Count voxels with value 1 in the lower half of the image
        upper_half_count = np.sum(
            binary_data[:, half_index:, :])  # Count voxels with value 1 in the upper half of the image
        if upper_half_count > lower_half_count:
            rotated_data = np.rot90(data, 2, axes=(
            0, 1))  # Rotate the image if the upper half count is greater than the lower half count
            rotated_nii = nib.Nifti1Image(rotated_data, affine=nib.load(file_path).affine)
        else:
            rotated_nii = nib.Nifti1Image(data, affine=nib.load(file_path).affine)
        nib.save(rotated_nii, output_path)  # Save the processed NIFTI image

    def process_dataset(self):
        # Process all NIFTI images in the dataset
        file_names = os.listdir(self.input_path)
        for folder_name in file_names:
            if not folder_name.endswith('.xlsx') and '.DS_Store' not in folder_name:
                folder_path = os.path.join(self.input_path, folder_name)
                for name in os.listdir(folder_path):
                    if '.DS_Store' not in name:
                        name_path = os.path.join(folder_path, name)
                        output_path = os.path.join(folder_path, f"{name.split('.nii')[0]}{self.output_suffix}")
                        self.process_nifti_image(name_path, output_path)


if __name__ == "__main__":
    input_path = "/home/xurbano/QEI-ASL/data"
    processor = Orientation_Corrector(input_path)
    processor.process_dataset()
