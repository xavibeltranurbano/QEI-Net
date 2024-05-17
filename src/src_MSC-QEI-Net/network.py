from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Activation, Dense, Flatten, Add
from tensorflow.keras.models import Model


class MSC_QEI_Net:
    def __init__(self, imgSize):
        # Initialize the class with the image size
        self.imgSize = imgSize

    def conv_block(self, size, x):
        # Define a convolutional block
        conv = Conv3D(size, (3, 3, 3), padding='same', kernel_initializer='glorot_uniform')(x)
        conv = Activation('relu')(conv)
        conv = Conv3D(size, (3, 3, 3), padding='same', kernel_initializer='glorot_uniform')(conv)
        conv = Activation('relu')(conv)
        return conv

    def conv_block_residual_connections(self, size, x):
        # Define a convolutional block with residual connections
        conv = Conv3D(size, (3, 3, 3), padding='same', kernel_initializer='glorot_uniform')(x)
        conv = Activation('relu')(conv)
        conv = Conv3D(size, (3, 3, 3), padding='same', kernel_initializer='glorot_uniform')(conv)
        x = Conv3D(size, (1, 1, 1), padding='same', kernel_initializer='glorot_uniform')(x)
        output = Add()([x, conv])
        output = Activation('relu')(output)
        return output

    def get_model(self):
        # Build and return the model
        inputs = Input(self.imgSize)
        FACTOR = 4
        POOL_SIZE = (2, 2, 2)
        STRIDES = (2, 2, 2)

        conv1 = self.conv_block_residual_connections(16 * FACTOR, inputs)
        pool1 = MaxPooling3D(pool_size=POOL_SIZE, strides=STRIDES)(conv1)

        conv2 = self.conv_block_residual_connections(32 * FACTOR, pool1)
        pool2 = MaxPooling3D(pool_size=POOL_SIZE, strides=STRIDES)(conv2)

        conv3 = self.conv_block_residual_connections(64 * FACTOR, pool2)
        pool3 = MaxPooling3D(pool_size=POOL_SIZE, strides=STRIDES)(conv3)

        conv4 = self.conv_block_residual_connections(128 * FACTOR, pool3)
        pool4 = MaxPooling3D(pool_size=POOL_SIZE, strides=STRIDES)(conv4)

        conv5 = self.conv_block_residual_connections(256 * FACTOR, pool4)

        output = Flatten()(conv5)
        output = Dense(4, activation='softmax')(output)
        model = Model(inputs=inputs, outputs=output)
        return model
