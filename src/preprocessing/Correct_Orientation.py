import nibabel as nib
import numpy as np
import os

def process_nifti_image(file_path,output_path):
    # Load the NIFTI image
    nii = nib.load(file_path)
    data = nii.get_fdata()
    
    # Binarize the image with a threshold of 0.9
    binary_data  = (data != 0).astype(int)

    # Determine the dividing index for the two halves of the image
    half_index = binary_data.shape[1] // 2
    
    # Count voxels with value 1 in each half of the image
    lower_half_count = np.sum(binary_data[:, :half_index, :])
    upper_half_count = np.sum(binary_data[:, half_index:, :])
    
    if upper_half_count > lower_half_count:
        rotated_data = np.rot90(data, 2, axes=(0, 1))
        rotated_nii = nib.Nifti1Image(rotated_data, affine=nii.affine)
    else: rotated_nii = nib.Nifti1Image(data, affine=nii.affine)
    nib.save(rotated_nii, output_path)  

pathDataset="/home/xurbano/QEI-ASL/data_v2"
fileNames= os.listdir(pathDataset)
for nameFolder in fileNames:
    if not nameFolder.endswith('.xlsx') and  '.DS_Store' not in nameFolder:
        folderPath=os.path.join(pathDataset, nameFolder)
        for name in os.listdir(folderPath):
            if '.DS_Store' not in name:
                namePath=os.path.join(folderPath, name)
                rotated_image = process_nifti_image(namePath,  folderPath+f"/{name.split('.nii')[0]}_CO.nii")
        
