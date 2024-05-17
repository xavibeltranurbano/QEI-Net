from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Activation, BatchNormalization, Dense, Flatten,Dropout,Concatenate, Add,AveragePooling3D, Lambda, GlobalMaxPooling3D, GlobalAveragePooling3D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
#from classification_models_3D.tfkeras import Classifiers

class QEI_Net:
    def __init__(self, imgSize):
        self.imgSize = imgSize

    """def conv_block(self, size, x):
        conv = Conv3D(size, (3, 3, 3), padding='same', kernel_initializer='glorot_uniform')(x)
        conv = Activation('relu')(conv)
        conv = Conv3D(size, (3, 3, 3), padding='same', kernel_initializer='glorot_uniform')(conv)
        conv = Activation('relu')(conv)
        return conv"""

    """def conv_block_residual_connections(self, size, x):
        conv = Conv3D(size, (3, 3, 3), padding='same', kernel_initializer='glorot_uniform')(x)
        #conv= BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        conv = Conv3D(size, (3, 3, 3), padding='same', kernel_initializer='glorot_uniform')(conv)
        #conv= BatchNormalization()(conv)
        x = Conv3D(size, (1, 1, 1), padding='same', kernel_initializer='glorot_uniform')(x)
        output = Add()([x, conv])
        output = Activation('relu')(output)
        return output"""
        

    """def get_model(self):
        inputs = Input(self.imgSize)
        FACTOR=2
        POOL_SIZE=(2, 2, 2)
        STRIDES=(2, 2, 2)
        
        conv1 = self.conv_block_residual_connections(16*FACTOR, inputs)  # Increased from 8 to 16
        pool1 = MaxPooling3D(pool_size=POOL_SIZE,strides=STRIDES)(conv1)
    
        conv2 = self.conv_block_residual_connections(32*FACTOR, pool1)  # Increased from 16 to 32
        pool2 = MaxPooling3D(pool_size=POOL_SIZE,strides=STRIDES)(conv2)
    
        conv3 = self.conv_block_residual_connections(64*FACTOR, pool2)  # Increased from 32 to 64
        pool3 = MaxPooling3D(pool_size=POOL_SIZE,strides=STRIDES)(conv3)
        
        conv4 = self.conv_block_residual_connections(128*FACTOR, pool3)  # Added new block
        pool4 = MaxPooling3D()(conv4)
    
        # Flatten and Fully Connected Layer for Regression
        output = Flatten()(pool4)  # Changed from conv3 to conv4
        output= Dropout(0.3)(output)
        output = Dense(128, activation='relu')(output)  # Increased from 128 to 256
        output = Dense(32, activation='relu')(output)  # Increased from 128 to 256
        output = Dense(1, activation='sigmoid')(output)  # Keep linear activation
        model = Model(inputs=inputs, outputs=output)
        return model"""

    """def get_model_2(self):
        inputs = Input(shape=(64, 64, 32, 2))  # Assuming last dimension is channels, with 2 channels.
        FACTOR = 2
        POOL_SIZE = (2, 2, 2)
        STRIDES = (2, 2, 2)
        
        # Manually selecting each channel
        channel_0 = Lambda(lambda x: x[:,:,:,:,0:1])(inputs)  # Selecting first channel
        channel_1 = Lambda(lambda x: x[:,:,:,:,1:2])(inputs)  # Selecting second channel
    
        # Processing the first channel
        conv1_0 = self.conv_block_residual_connections(16*FACTOR, channel_0)
        pool1_0 = MaxPooling3D(pool_size=POOL_SIZE, strides=STRIDES)(conv1_0)
        conv2_0 = self.conv_block_residual_connections(32*FACTOR, pool1_0)
        pool2_0 = MaxPooling3D(pool_size=POOL_SIZE, strides=STRIDES)(conv2_0)
        conv3_0 = self.conv_block_residual_connections(64*FACTOR, pool2_0)
        pool3_0 = MaxPooling3D(pool_size=POOL_SIZE, strides=STRIDES)(conv3_0)
        conv4_0 = self.conv_block_residual_connections(128*FACTOR, pool3_0)
        
        # Processing the second channel
        conv1_1 = self.conv_block_residual_connections(16*FACTOR, channel_1)
        pool1_1 = MaxPooling3D(pool_size=POOL_SIZE, strides=STRIDES)(conv1_1)
        conv2_1 = self.conv_block_residual_connections(32*FACTOR, pool1_1)
        pool2_1 = MaxPooling3D(pool_size=POOL_SIZE, strides=STRIDES)(conv2_1)
        conv3_1 = self.conv_block_residual_connections(64*FACTOR, pool2_1)
        pool3_1 = MaxPooling3D(pool_size=POOL_SIZE, strides=STRIDES)(conv3_1)
        conv4_1 = self.conv_block_residual_connections(128*FACTOR, pool3_1)
    
        # Concatenate the features from both channels before the dense layers
        concatenated_features = Concatenate(axis=-1)([Flatten()(conv4_0), Flatten()(conv4_1)])
        
        # Flatten and Fully Connected Layer for Regression
        output = Dense(256, activation='relu')(concatenated_features)
        output = Dense(64, activation='relu')(output)
        output = Dense(1, activation='sigmoid')(output)  # Keep linear activation for regression
        model = Model(inputs=inputs, outputs=output)
        return model

    def conv_block_dilated(self, size, x):
        conv = Conv3D(size, (3, 3, 3), padding='same',dilation_rate=(6, 6, 6), kernel_initializer='glorot_uniform')(x)
        conv = Activation('relu')(conv)
        conv = Conv3D(size, (3, 3, 3), padding='same',dilation_rate=(6, 6, 6), kernel_initializer='glorot_uniform')(conv)
        conv = Activation('relu')(conv)
        return conv

    def conv_block_dilated_residual(self, size, x):
        conv = Conv3D(size, (3, 3, 3), padding='same',dilation_rate=(4, 4, 4), kernel_initializer='glorot_uniform')(x)
        #conv= BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        conv = Conv3D(size, (3, 3, 3), padding='same',dilation_rate=(4, 4, 4), kernel_initializer='glorot_uniform')(conv)
        #sconv= BatchNormalization()(conv)
        x = Conv3D(size, (1, 1, 1), padding='same', kernel_initializer='glorot_uniform')(x)
        output = Add()([x, conv])
        output = Activation('relu')(output)
        return output

    def get_model_dilated(self):
        inputs = Input(self.imgSize)
        FACTOR=2
        # Normal CONVOLUTIONS
        conv1 = self.conv_block_residual_connections(16*FACTOR, inputs)
        pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
        conv2 = self.conv_block_residual_connections(32*FACTOR, pool1)
        pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
        conv3 = self.conv_block_residual_connections(64*FACTOR, pool2)
        pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
        conv4 = self.conv_block_residual_connections(128*FACTOR, pool3)
        pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
    
        # DILATED CONVOLUTIONS
        conv1_dil = self.conv_block_dilated_residual(16*FACTOR, inputs)
        pool1_dil = MaxPooling3D(pool_size=(2, 2, 2))(conv1_dil)
        conv2_dil = self.conv_block_dilated_residual(32*FACTOR, pool1_dil)
        pool2_dil = MaxPooling3D(pool_size=(2, 2, 2))(conv2_dil)
        conv3_dil = self.conv_block_dilated_residual(64*FACTOR, pool2_dil)
        pool3_dil = MaxPooling3D(pool_size=(2, 2, 2))(conv3_dil)
        conv4_dil = self.conv_block_dilated_residual(128*FACTOR, pool3_dil)
        pool4_dil = MaxPooling3D(pool_size=(2, 2, 2))(conv4_dil)
    
        # Concatenation of the final convolutional outputs
        concatenated = Concatenate()([pool4, pool4_dil])
        
        # Flatten and Fully Connected Layer for Regression
        output = Flatten()(concatenated)
        output= Dropout(0.2)(concatenated)
        output = Dense(128, activation='relu')(output)  # Increased from 128 to 256
        output = Dense(32, activation='relu')(output)  # Increased from 128 to 256
        output = Dense(1, activation='sigmoid')(output)  # Keep linear activation
        model = Model(inputs=inputs, outputs=output)
        return model"""


    """def get_model_pretrained(self):  # Corrected to include 
        inputs = Input(shape=self.imgSize)  # Use 'shape=' and access imgSize with 'self'
        
        # Assuming 'resnet18' is the correct identifier for a 3D ResNet model you wish to use
        # This line might need adjustment based on the actual library and available models
        ResNet3D, _ = Classifiers.get('resnet18')
        base_model = ResNet3D(input_shape=self.imgSize, include_top=False, weights=None, input_tensor=inputs)
        
        # Adding layers on top of the base model
        flatten = Flatten()(base_model.output)
        #dropout = Dropout(0.3)(flatten)
        output = Dense(1, activation='sigmoid')(flatten)  # Assuming a regression task
        
        model = Model(inputs=inputs, outputs=output)
        return model"""

    def conv_block(self, size, x):
        conv = Conv3D(size, (3, 3, 3), padding='same', kernel_initializer='he_normal')(x)
        conv = Activation('relu')(conv)
        conv = Conv3D(size, (3, 3, 3), padding='same', kernel_initializer='he_normal')(conv)
        conv = Activation('relu')(conv)
        return conv

    def conv_block_residual_connections(self, size, x):
        conv = Conv3D(size, (3, 3, 3), padding='same', kernel_initializer='glorot_normal')(x)
        conv = Activation('relu')(conv)
        conv = Conv3D(size, (3, 3, 3), padding='same', kernel_initializer='glorot_normal')(conv)
        x = Conv3D(size, (1, 1, 1), padding='same', kernel_initializer='glorot_normal')(x) 
        output = Add()([x, conv])
        output = Activation('relu')(output)
        return output

    def get_model(self):
        inputs = Input(self.imgSize)
        FACTOR=4
        POOL_SIZE=(2, 2, 2)
        STRIDES=(2, 2, 2)
        
        conv1 = self.conv_block_residual_connections(16*FACTOR, inputs)  # Increased from 8 to 16
        pool1 = MaxPooling3D(pool_size=POOL_SIZE,strides=STRIDES)(conv1)
    
        conv2 = self.conv_block_residual_connections(32*FACTOR, pool1)  # Increased from 16 to 32
        pool2 = MaxPooling3D(pool_size=POOL_SIZE,strides=STRIDES)(conv2)
    
        conv3 = self.conv_block_residual_connections(64*FACTOR, pool2)  # Increased from 32 to 64
        pool3 = MaxPooling3D(pool_size=POOL_SIZE,strides=STRIDES)(conv3)
        
        conv4 = self.conv_block_residual_connections(128*FACTOR, pool3)  # Added new block
        pool4 = MaxPooling3D(pool_size=POOL_SIZE,strides=STRIDES)(conv4)
        
        # Flatten and Fully Connected Layer for Regression
        output = Flatten()(conv4)  # Changed from conv3 to conv4
        #output= Dropout(0.3)(output)
        output = Dense(128, activation='relu')(output)  # Increased from 128 to 256
        output = Dense(64, activation='relu')(output)  # Increased from 128 to 256
        output = Dense(1, activation='sigmoid')(output)  # Keep linear activation
        model = Model(inputs=inputs, outputs=output)
        return model

    """# WORK!
    def conv_block(self, size, x):
        conv = Conv3D(size, (3, 3, 3), padding='same', kernel_initializer='he_normal')(x)
        conv = Activation('relu')(conv)
        conv = Conv3D(size, (3, 3, 3), padding='same', kernel_initializer='he_normal')(conv)
        conv = Activation('relu')(conv)
        return conv

    def conv_block_residual_connections(self, size, x):
        conv = Conv3D(size, (3, 3, 3), padding='same', kernel_initializer='glorot_normal')(x)
        conv = Activation('relu')(conv)
        conv = Conv3D(size, (3, 3, 3), padding='same', kernel_initializer='glorot_normal')(conv)
        x = Conv3D(size, (1, 1, 1), padding='same', kernel_initializer='glorot_normal')(x)
        output = Add()([x, conv])
        output = Activation('relu')(output)
        return output

    def get_model(self):
        inputs = Input(self.imgSize)
        FACTOR=2
        POOL_SIZE=(2, 2, 2)
        STRIDES=(2, 2, 2)
        
        conv1 = self.conv_block_residual_connections(16*FACTOR, inputs)  # Increased from 8 to 16
        pool1 = MaxPooling3D(pool_size=POOL_SIZE,strides=STRIDES)(conv1)
    
        conv2 = self.conv_block_residual_connections(32*FACTOR, pool1)  # Increased from 16 to 32
        pool2 = MaxPooling3D(pool_size=POOL_SIZE,strides=STRIDES)(conv2)
    
        conv3 = self.conv_block_residual_connections(64*FACTOR, pool2)  # Increased from 32 to 64
        pool3 = MaxPooling3D(pool_size=POOL_SIZE,strides=STRIDES)(conv3)
        
        conv4 = self.conv_block_residual_connections(128*FACTOR, pool3)  # Added new block
        pool4 = MaxPooling3D(pool_size=POOL_SIZE,strides=STRIDES)(conv4)
    
        # Flatten and Fully Connected Layer for Regression
        output = Flatten()(conv4)  # Changed from conv3 to conv4
        #output= Dropout(0.3)(output)
        output = Dense(128, activation='relu')(output)  # Increased from 128 to 256
        output = Dense(32, activation='relu')(output)  # Increased from 128 to 256
        output = Dense(1, activation='sigmoid')(output)  # Keep linear activation
        model = Model(inputs=inputs, outputs=output)
        return model
        """
