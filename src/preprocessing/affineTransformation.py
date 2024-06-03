# -----------------------------------------------------------------------------
# Registration file
# Author: Xavier Beltran Urbano
# Date Created: 21-02-2024
# -----------------------------------------------------------------------------


import nibabel as nib
import SimpleITK as sitk
import numpy as np
import os


class Transformation:
    def __init__(self, target_size=(64, 64, 32)):
        # Initialize the registration class with the target size
        self.target_size = target_size

    def load_nifti_image(self, image_path):
        # Load a NIfTI image and return the data and affine
        nifti_img = nib.load(image_path)
        return nifti_img.get_fdata(), nifti_img.affine

    def save_nifti_image(self, data, output_path, affine):
        # Save a NIfTI image with the given data and affine
        new_img = nib.Nifti1Image(data, affine)
        nib.save(new_img, output_path)

    def resample_image(self, image, new_size, new_spacing):
        # Resample an image to the new size and spacing
        resampler = sitk.ResampleImageFilter()
        orig_size = np.array(image.GetSize(), dtype=np.float64)
        orig_spacing = image.GetSpacing()
        resampler.SetSize(new_size.tolist())
        resampler.SetOutputSpacing(new_spacing)
        original_center = np.array(image.GetOrigin()) + (orig_spacing * orig_size) / 2.0
        new_center = (np.array(new_spacing) * np.array(new_size)) / 2.0
        new_origin = original_center - new_center
        resampler.SetOutputOrigin(new_origin.tolist())
        resampler.SetTransform(sitk.AffineTransform(3))
        resampler.SetInterpolator(sitk.sitkLinear)
        return resampler.Execute(image)

    def run(self, image_path, output_path):
        # Run the registration process
        original_data, affine = self.load_nifti_image(image_path)
        image_sitk = sitk.GetImageFromArray(np.transpose(original_data))
        original_size = np.array(image_sitk.GetSize(), dtype=np.float64)
        original_spacing = np.array(image_sitk.GetSpacing())
        new_spacing = original_size / self.target_size * original_spacing
        resampled_image = self.resample_image(image_sitk, np.array(self.target_size), new_spacing)
        resampled_data = sitk.GetArrayFromImage(resampled_image)
        resampled_data = np.transpose(resampled_data, (2, 1, 0))  # XYZ for nibabel
        self.save_nifti_image(resampled_data, output_path, affine)


if __name__ == "__main__":
    path='/Datasets/QEI-Dataset'
    fileNames=[file for file in os.listdir(path) if not file.endswith('xlsx') and  not file.endswith('.DS_Store') and  not file.endswith('.xls') and  not file.endswith('.ipynb_checkpoints')]
    affine = Transformation(target_size=(64, 64, 32))
    for folderName in fileNames:
        for name in os.listdir(os.path.join(path,folderName)):
            if not name.endswith('xlsx') and  not name.endswith('.DS_Store'):
                newPath=os.path.join(path, folderName, name)
                outputPath=os.path.join(path, folderName, f"{name.split('.nii')[0]}_Reg.nii")
                affine.run(newPath, outputPath)
