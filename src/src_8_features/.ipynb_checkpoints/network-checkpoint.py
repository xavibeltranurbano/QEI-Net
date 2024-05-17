from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Activation, BatchNormalization, Dense, Flatten,Dropout,Lambda
from tensorflow.keras.models import Model
import tensorflow as tf
import tensorflow.keras.backend as K

class FCDN_QEI:
    def __init__(self, nFeatures):
        self.nFeatures = nFeatures

    """def get_model(self):
        inputs = Input(shape=self.nFeatures)
        normalized_input = BatchNormalization()(inputs)
        
        # First dropout layer
        dropout1 = Dropout(0.3)(normalized_input)
        fully_connected_layer1 = Dense(16, activation='relu')(dropout1)
        
        # Additional dropout layers
        dropout2 = Dropout(0.3)(fully_connected_layer1)
        fully_connected_layer2 = Dense(16, activation='relu')(dropout2)
        
        dropout3 = Dropout(0.3)(fully_connected_layer2)
        fully_connected_layer3 = Dense(64, activation='relu')(dropout3)
        
        dropout4 = Dropout(0.3)(fully_connected_layer3)
        fully_connected_layer4 = Dense(64, activation='relu')(dropout4)
        
        dropout5 = Dropout(0.3)(fully_connected_layer4)
        fully_connected_layer5 = Dense(128, activation='relu')(dropout5)
        
        dropout6 = Dropout(0.3)(fully_connected_layer5)
        fully_connected_layer6 = Dense(128, activation='relu')(dropout6)
        
        dropout7 = Dropout(0.3)(fully_connected_layer6)
        fully_connected_layer7 = Dense(256, activation='relu')(dropout7)
        
        dropout8 = Dropout(0.3)(fully_connected_layer7)
        fully_connected_layer8 = Dense(256, activation='relu')(dropout8)
        
        dropout9 = Dropout(0.3)(fully_connected_layer8)
        fully_connected_layer9 = Dense(128, activation='relu')(dropout9)
        
        dropout10 = Dropout(0.3)(fully_connected_layer9)
        fully_connected_layer10 = Dense(128, activation='relu')(dropout10)
        
        dropout11 = Dropout(0.3)(fully_connected_layer10)
        fully_connected_layer11 = Dense(64, activation='relu')(dropout11)
        
        dropout12 = Dropout(0.3)(fully_connected_layer11)
        fully_connected_layer12 = Dense(64, activation='relu')(dropout12)
        
        dropout13 = Dropout(0.3)(fully_connected_layer12)
        fully_connected_layer13 = Dense(16, activation='relu')(dropout13)
        
        dropout14 = Dropout(0.3)(fully_connected_layer13)
        fully_connected_layer14 = Dense(16, activation='relu')(dropout14)
        
        dropout15 = Dropout(0.3)(fully_connected_layer14)
        fully_connected_layer15 = Dense(4, activation='relu')(dropout15)
        
        dropout16 = Dropout(0.3)(fully_connected_layer15)
        fully_connected_layer16 = Dense(4, activation='relu')(dropout16)
        
        output = Dense(1, activation='linear')(fully_connected_layer16)  # Linear activation for regression
        model = Model(inputs=inputs, outputs=output)
        return model"""

    def get_model(self):
        inputs = Input(shape=self.nFeatures)
        normalized_input = BatchNormalization()(inputs)
        FACTOR=4
        # First dense layer with 16 units
        x = Dense(16*FACTOR, activation='relu')(inputs)
        
        # Second dense layer with 64 units
        x = Dense(64*FACTOR, activation='relu')(x)
        
        # Third dense layer with 128 units
        x = Dense(128*FACTOR, activation='relu')(x)

        x = Dense(256*FACTOR, activation='relu')(x)
        x = Dense(512*FACTOR, activation='relu')(x)
        x = Dense(256*FACTOR, activation='relu')(x)
        x = Dense(128*FACTOR, activation='relu')(x)
        # Fourth dense layer back to 64 units
        x = Dense(64*FACTOR, activation='relu')(x)
        
        # Fifth dense layer reducing to 16 units
        x = Dense(16*FACTOR, activation='relu')(x)
        
        # Sixth dense layer reducing further to 4 units
        x = Dense(4*FACTOR, activation='relu')(x)
        
        # Final output layer with 1 unit, linear activation for regression
        output = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=output)
        return model
    
    """def get_model(self):
        print(self.nFeatures)
        inputs = Input(shape=self.nFeatures)
        normalized_input = BatchNormalization()(inputs)
        dropout = Dropout(0.3)(normalized_input)
        dense1 = Dense(16, activation='relu')(dropout)
        dense2 = Dense(16, activation='relu')(dense1)
        dense3 = Dense(64, activation='relu')(dense2)
        dense4 = Dense(64, activation='relu')(dense3)
        dense5 = Dense(128, activation='relu')(dense4)
        dense6 = Dense(128, activation='relu')(dense5)
        dense7 = Dense(256, activation='relu')(dense6)
        dense8 = Dense(256, activation='relu')(dense7)
        dense9 = Dense(128, activation='relu')(dense8)
        dense10 = Dense(128, activation='relu')(dense9)
        dense11 = Dense(64, activation='relu')(dense10)
        dense12 = Dense(64, activation='relu')(dense11)
        dense13 = Dense(16, activation='relu')(dense12)
        dense14 = Dense(16, activation='relu')(dense13)
        dense15 = Dense(4, activation='relu')(dense14)
        dense16 = Dense(4, activation='relu')(dense15)
        
         # Creating 5 separate outputs from the final layer
        output1 = Dense(1, activation='sigmoid')(dense16)
        output2 = Dense(1, activation='sigmoid')(dense16)
        output3 = Dense(1, activation='sigmoid')(dense16)
        output4 = Dense(1, activation='sigmoid')(dense16)
        output5 = Dense(1, activation='sigmoid')(dense16)
        
         # Averaging the outputs
        outputs = [output1, output2, output3, output4, output5]
        combined_output = Lambda(lambda x: K.mean(K.stack(x, axis=1), axis=1, keepdims=True))(outputs)
        
        model = Model(inputs=inputs, outputs=combined_output)
        return model"""
    
