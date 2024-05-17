from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Activation, BatchNormalization, Dense, Flatten,Dropout,Concatenate, Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

class QEI_Net:
    def __init__(self, imgSize):
        self.imgSize = imgSize

    def conv_block(self, size, x):
        conv = Conv3D(size, (3, 3, 3), padding='same', kernel_initializer='glorot_uniform')(x)
        conv = Activation('relu')(conv)
        conv = Conv3D(size, (3, 3, 3), padding='same', kernel_initializer='glorot_uniform')(conv)
        conv = Activation('relu')(conv)
        return conv

    def conv_block_residual_connections(self, size, x):
        conv = Conv3D(size, (3, 3, 3), padding='same', kernel_initializer='glorot_uniform')(x)
        #conv= BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        conv = Conv3D(size, (3, 3, 3), padding='same', kernel_initializer='glorot_uniform')(conv)
        #conv= BatchNormalization()(conv)
        x = Conv3D(size, (1, 1, 1), padding='same', kernel_initializer='glorot_uniform')(x)
        output = Add()([x, conv])
        output = Activation('relu')(output)
        return output
        

    def get_model(self):
        inputs = Input(self.imgSize)
        FACTOR=4
        POOL_SIZE=(2, 2, 2)
        STRIDES=(2, 2, 2)
        
        #conv0 = Conv3D(16*FACTOR, (3, 3, 3), activation=None, padding='same', kernel_initializer='glorot_uniform')(inputs)
        #conv0 = BatchNormalization()(conv0)
        #conv0 = Activation('relu')(conv0)
        conv1 = self.conv_block_residual_connections(16*FACTOR, inputs)  # Increased from 8 to 16
        pool1 = MaxPooling3D(pool_size=POOL_SIZE,strides=STRIDES)(conv1)
    
        conv2 = self.conv_block_residual_connections(32*FACTOR, pool1)  # Increased from 16 to 32
        pool2 = MaxPooling3D(pool_size=POOL_SIZE,strides=STRIDES)(conv2)
    
        conv3 = self.conv_block_residual_connections(64*FACTOR, pool2)  # Increased from 32 to 64
        pool3 = MaxPooling3D(pool_size=POOL_SIZE,strides=STRIDES)(conv3)
        
        conv4 = self.conv_block_residual_connections(128*FACTOR, pool3)  # Added new block
        pool4 = MaxPooling3D(pool_size=POOL_SIZE,strides=STRIDES)(conv4)

        conv5 = self.conv_block_residual_connections(256*FACTOR, pool3)
    
        # Flatten and Fully Connected Layer for Regression
        output = Flatten()(conv5)  # Changed from conv3 to conv4
        #output= Dropout(0.2)(output)
        #output = Dense(64, activation='relu')(output)  # Increased from 128 to 256
        #output = Dense(16, activation='relu')(output)  # Increased from 128 to 256
        output = Dense(4, activation='softmax')(output)  # Keep linear activation
        model = Model(inputs=inputs, outputs=output)
        return model

        """def get_model(self):  # Corrected to include 
        inputs = Input(shape=self.imgSize)  # Use 'shape=' and access imgSize with 'self'
        
        # Assuming 'resnet18' is the correct identifier for a 3D ResNet model you wish to use
        # This line might need adjustment based on the actual library and available models
        ResNet3D, _ = Classifiers.get('inceptionresnetv2')
        base_model = ResNet3D(input_shape=self.imgSize, include_top=False, weights=None, input_tensor=inputs)
        
        # Adding layers on top of the base model
        flatten = Flatten()(base_model.output)
        dropout = Dropout(0.3)(flatten)
        output = Dense(1, activation='linear')(dropout)  # Assuming a regression task
        
        model = Model(inputs=inputs, outputs=output)
        return model"""
