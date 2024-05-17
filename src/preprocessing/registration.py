import nibabel as nib
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os

class Registration:
    def __init__(self, target_size=(64, 64, 25)):
        self.target_size = target_size
    
    def load_nifti_image(self,image_path):
        nifti_img = nib.load(image_path)
        return nifti_img.get_fdata(), nifti_img.affine
    
    def save_nifti_image(self, data, output_path, affine):
        new_img = nib.Nifti1Image(data, affine)
        nib.save(new_img, output_path)
    
    def resample_image(self, image, new_size, new_spacing):
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
    
    def run(self,image_path,output_path):
        original_data, affine=self.load_nifti_image(image_path)
        image_sitk = sitk.GetImageFromArray(np.transpose(original_data))
        original_size = np.array(image_sitk.GetSize(), dtype=np.float64)
        original_spacing = np.array(image_sitk.GetSpacing())
        new_spacing = original_size / self.target_size * original_spacing
        resampled_image = self.resample_image(image_sitk, np.array(self.target_size), new_spacing)
        resampled_data = sitk.GetArrayFromImage(resampled_image)
        resampled_data = np.transpose(resampled_data, (2, 1, 0))  # XYZ for nibabel
        self.save_nifti_image(resampled_data,output_path, affine)

if __name__ == "__main__":
    path='/home/xurbano/QEI-ASL/data_final'
    fileNames=[file for file in os.listdir(path) if not file.endswith('xlsx') and  not file.endswith('.DS_Store') and  not file.endswith('.xls') and  not file.endswith('.ipynb_checkpoints')]
    reg = Registration(target_size=(64, 64, 32))
    for name in fileNames:
        newPath=os.path.join(path, name, "CBF_Map.nii")
        outputPath=os.path.join(path, name, "CBF_Map_reshaped.nii")
        reg.run(newPath, outputPath)
   