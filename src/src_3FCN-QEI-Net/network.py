# -----------------------------------------------------------------------------
# Network for the 3FCN-QEI-Net
# Author: Xavier Beltran Urbano
# Date Created: 21-02-2024
# -----------------------------------------------------------------------------


from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

class FCDN_QEI:
    def __init__(self, nFeatures):
        # Initialize the class with the number of features
        self.nFeatures = nFeatures

    def get_model(self):
        # Build and return the Fully Connected Deep Network model
        FACTOR = 4
        inputs = Input(shape=self.nFeatures)
        x = Dense(16 * FACTOR, activation='relu')(inputs)
        x = Dense(64 * FACTOR, activation='relu')(x)
        x = Dense(128 * FACTOR, activation='relu')(x)
        x = Dense(64 * FACTOR, activation='relu')(x)
        x = Dense(16 * FACTOR, activation='relu')(x)
        x = Dense(4 * FACTOR, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=output)
        return model
