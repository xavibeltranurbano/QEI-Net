# -----------------------------------------------------------------------------
# Data Generator for 3D data
# Author: Xavier Beltran Urbano 
# Date Created: 21-02-2024
# -----------------------------------------------------------------------------

from keras.utils import Sequence
import numpy as np
import tensorflow as tf
from ASL_Preprocessing import Preprocessing
import math


class DataGenerator(Sequence):

    def __init__(self, list_IDs, image_directory, rater, batch_size=1, minibatch_size=32, target_size=(64, 64, 25),
                 data_augmentation=True, shuffle=True,**kwargs):
        super().__init__(**kwargs)
        self.image_directory = image_directory
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.data_augmentation = data_augmentation
        self.indexes = np.arange(len(self.list_IDs))
        self.preprocessing = Preprocessing(data_aug=data_augmentation, norm_intensity=True, imageSize=target_size, rater=rater)
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
        return X,y
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        for i, ID in enumerate(list_IDs_temp):
            x, y = self.preprocessing.process_case(ID, self.image_directory)
            #print(f"Rating {y}")
            # Reshape each slice to add a channel dimension
            x = np.expand_dims(x, axis=0)  
            y = np.expand_dims(y, axis=0)
            y = tf.constant(y) - 1
            y=tf.one_hot(y,depth=4)
            # Concatenate data in batch.
            if i == 0:
                batch_x = x
                batch_y = y
            else:
                batch_x = tf.concat([batch_x, x], 0)
                batch_y = tf.concat([batch_y, y], 0)
        
        return batch_x, batch_y