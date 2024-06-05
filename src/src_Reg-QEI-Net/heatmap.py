# -----------------------------------------------------------------------------
# Heat Map Generator for 3D data
# Author: Xavier Beltran Urbano
# Date Created: 21-05-2024
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Heat Map Generator for 3D data
# Author: Xavier Beltran Urbano
# Date Created: 21-02-2024
# -----------------------------------------------------------------------------


import os
import numpy as np
import nibabel as nib
from network import QEI_Net
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3D
from scipy.ndimage import zoom
import matplotlib.pyplot as plt


class GradCamGenerator:
    def __init__(self, model_fold_paths, img_path, output_folder):
        self.model_fold_paths = model_fold_paths
        self.img_path = img_path
        self.output_folder = output_folder
        self.network = QEI_Net(imgSize=(64, 64, 32, 1))
        self.model = self.network.get_model()
        self.conv3d_layers = [layer.name for layer in self.model.layers if isinstance(layer, Conv3D)]
        if len(self.conv3d_layers) < 8:
            raise ValueError("Not enough Conv3D layers in the model")
        self.target_layer_name = self.conv3d_layers[5]

    def compute_grad_cam(self, img_array):
        # Compute the Grad-CAM heatmap for the given image array.
        conv_layer = self.model.get_layer(self.target_layer_name)
        grad_model = Model(inputs=self.model.inputs, outputs=[conv_layer.output, self.model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]
        output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]
        guided_grads = tf.cast(output > 0, "float32") * tf.cast(grads > 0, "float32") * grads
        weights = tf.reduce_mean(guided_grads, axis=(0, 1, 2))
        cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1)
        heatmap = tf.nn.relu(cam)
        heatmap = heatmap / tf.reduce_max(heatmap)
        return heatmap.numpy()

    def save_nifti(self, data, file_path):
        # Save the data as a NIfTI file at the specified file path.
        img = nib.Nifti1Image(data, np.eye(4))
        nib.save(img, file_path)

    def normalize_intensity(self, img):
        # Normalize the intensity of the image.
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

    def generate_heatmap(self):
        # Generate and display the averaged heatmap from all model folds.
        img = nib.load(self.img_path).get_fdata()
        img = self.normalize_intensity(img)
        original_mask = img > 0
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)
        accumulated_heatmap = np.zeros_like(img[0, :, :, :, 0])

        for fold_path in self.model_fold_paths:
            self.model.load_weights(fold_path, skip_mismatch=True)
            heatmap = self.compute_grad_cam(img)
            zoom_factors = np.array(original_mask.shape) / np.array(heatmap.shape)
            resized_heatmap = zoom(heatmap, zoom_factors, order=1)
            final_heatmap = resized_heatmap * original_mask
            accumulated_heatmap += final_heatmap

        averaged_heatmap = accumulated_heatmap / len(self.model_fold_paths)
        os.makedirs(self.output_folder, exist_ok=True)
        name = os.path.basename(os.path.dirname(self.img_path))
        output_file_path = os.path.join(self.output_folder, name + '.nii')
        self.save_nifti(averaged_heatmap, output_file_path)
        print(f"Heatmap saved as NIfTI file: {output_file_path}")


if __name__ == "__main__":
    model_fold_paths = [
        "/results/QEI-Net_3/1/Best_Model.keras",
        "L/results/QEI-Net_3/2/Best_Model.keras",
        "/results/QEI-Net_3/3/Best_Model.keras",
        "/results/QEI-Net_3/4/Best_Model.keras",
        "/QEI-Net_3/5/Best_Model.keras"
    ]
    img_path = "/data_final/ASL_92/CBF_Map_CO_Reg.nii"
    output_folder = "/results/QEI-Net_3/heatmaps"

    grad_cam_generator = GradCamGenerator(model_fold_paths, img_path, output_folder)
    grad_cam_generator.generate_heatmap()
