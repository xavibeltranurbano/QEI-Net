# -----------------------------------------------------------------------------
# Data Generator for 3D data
# Author: Xavier Beltran Urbano 
# Date Created: 21-02-2024
# -----------------------------------------------------------------------------

from keras.utils import Sequence
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import math


class DataGenerator(Sequence):

    def __init__(self, list_IDs, image_directory, batch_size=1, minibatch_size=32, target_size=(64, 64, 25),
                 data_augmentation=True, shuffle=True):
        self.image_directory = image_directory
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.data_augmentation = data_augmentation
        self.indexes = np.arange(len(self.list_IDs))
        self.features = pd.read_excel("/src_7FCN-QEI-Net/computed_features.xlsx")
        self.ratings = pd.read_excel("/data_final/Ratings.xlsx")
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.list_IDs) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.list_IDs))
        batch_indexes = self.indexes[start:end]
        list_IDs_temp = [self.list_IDs[k] for k in batch_indexes]
        X, y = self.__data_generation(list_IDs_temp)
        # Assuming index corresponds to a batch in all_batches_x and all_batches_y
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def readAnnotation(self, ID):
        'Return the annotation of a specific ID'
        matched_row = self.ratings[self.ratings.iloc[:, 0] == ID]
        rating = (matched_row.iloc[-1, -1] - 1.0) / 3.0
        return rating

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialize as empty tensors
        batch_x = tf.constant([], dtype=tf.float32)
        batch_y = tf.constant([], dtype=tf.float32)

        for i, ID in enumerate(list_IDs_temp):
            x = self.features.loc[self.features['ID'] == ID]
            x = x.iloc[:, 1:]  # Drop the ID column
            y = self.readAnnotation(ID)
            # Convert x to a tensor, ensure dtype is float32 for consistency
            x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
            y_tensor = tf.convert_to_tensor([y], dtype=tf.float32)  # Wrap y in a list to make it a 1D tensor

            # Reshape x_tensor to ensure it has the correct shape (batch_size, num_features)
            x_tensor = tf.reshape(x_tensor, (1, -1))  # Reshape to (1, num_features)

            if i == 0:
                batch_x = x_tensor
                batch_y = y_tensor
            else:
                batch_x = tf.concat([batch_x, x_tensor], 0)
                batch_y = tf.concat([batch_y, y_tensor], 0)

        return batch_x, batch_y


