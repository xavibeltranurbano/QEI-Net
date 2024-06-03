from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Activation, Dense, Flatten, Add, Dropout, BatchNormalization,GlobalAveragePooling3D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

class QEI_Net:
    def __init__(self, imgSize):
        # Initialize the class with the image size
        self.imgSize = imgSize

    def conv_block_residual_connections(self, size, x):
        # Define a convolutional block with residual connections
        conv = Conv3D(size, (3, 3, 3), padding='same', kernel_initializer='glorot_normal')(x)
        conv = Activation('relu')(conv)
        conv = Conv3D(size, (3, 3, 3), padding='same', kernel_initializer='glorot_normal')(conv)
        x = Conv3D(size, (1, 1, 1), padding='same', kernel_initializer='glorot_normal')(x)
        output = Add()([x, conv])
        output = Activation('relu')(output)
        return output

    def get_model(self):
        # Build and return the model
        inputs = Input(self.imgSize)
        FACTOR = 2
        POOL_SIZE = (2, 2, 2)
        STRIDES = (2, 2, 2)

        conv1 = self.conv_block_residual_connections(16 * FACTOR, inputs)
        pool1 = MaxPooling3D(pool_size=POOL_SIZE, strides=STRIDES)(conv1)

        conv2 = self.conv_block_residual_connections(32 * FACTOR, pool1)
        pool2 = MaxPooling3D(pool_size=POOL_SIZE, strides=STRIDES)(conv2)

        conv3 = self.conv_block_residual_connections(64 * FACTOR, pool2)
        pool3 = MaxPooling3D(pool_size=POOL_SIZE, strides=STRIDES)(conv3)

        conv4 = self.conv_block_residual_connections(128 * FACTOR, pool3)

        output = Flatten()(conv4)
        output = Dense(128, activation='relu')(output)
        output = Dropout(0.2)(output)
        output = Dense(32, activation='relu')(output)
        output = Dropout(0.2)(output)
        output = Dense(1, activation='sigmoid')(output)
        model = Model(inputs=inputs, outputs=output)
        return model
