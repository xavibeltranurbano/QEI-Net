import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from network import QEI_Net
from tensorflow.keras.models import load_model
import nibabel as nib
from tensorflow.keras import backend as K

class GradCam:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.grad_model = Model(inputs=self.model.inputs,
                        outputs=[self.model.get_layer(self.layer_name).output, self.model.output])


    def compute_heatmap(self, image, eps=1e-8):
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image)
            loss = predictions  # Focus on the most influential output neuron
    
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))  # This should now work
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def save_heatmap(self, heatmap, path):
        nifti_img = nib.Nifti1Image(heatmap, affine=np.eye(4))
        # Save the NIfTI image to the specified path
        nib.save(nifti_img, path)


def normalizeIntensity(img):
        #backgroundMask=self.maskBackground(img)
        image_min, image_max = 0, 1
        normalizedImage = ((img - np.min(img)) / (np.max(img) - np.min(img))) * (image_max - image_min) + image_min
        #normalizedImage[backgroundMask] = 0
        return normalizedImage
    
if __name__ == "__main__":
     
    # Example model and input (Replace these with your actual model and input)
    network = QEI_Net(imgSize=(64,64,30,1))
    model = network.get_model()
    model.load_weights(f"/home/xurbano/QEI-ASL/results/QEI-NET_CNN/1/Best_Model.keras")
    layer_name = 'conv3d_1'
    
    grad_cam = GradCam(model, layer_name)

    path="/home/xurbano/QEI-ASL/data_v2/ASL_1/CBF_Map_reshaped.nii"
    img = nib.load(path).get_fdata()
    img=normalizeIntensity(img)
    img = np.expand_dims(img, axis=0)  
    img = np.expand_dims(img, axis=-1)  

    heatmap = grad_cam.compute_heatmap(img)
    grad_cam.save_heatmap(heatmap, 'ASL_1_heatmap.nii')  # Save the middle slice of the heatmap
    print("Heatmap middle slice saved to 'heatmap_middle_slice.png'.")
