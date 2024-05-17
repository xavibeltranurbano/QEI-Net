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

    def __init__(self, list_IDs, image_directory, batch_size=1, minibatch_size=32, target_size=(64, 64, 25),
                 data_augmentation=True, shuffle=True):
        self.image_directory = image_directory
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.data_augmentation = data_augmentation
        self.indexes = np.arange(len(self.list_IDs))
        self.preprocessing = Preprocessing(data_aug=data_augmentation, norm_intensity=True, imageSize=target_size)
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
            # Reshape each slice to add a channel dimension
            x = np.expand_dims(x, axis=0)  
            y = np.expand_dims(y, axis=0)  
            # Concatenate data in batch.
            if i == 0:
                batch_x = x
                batch_y = y
            else:
                batch_x = tf.concat([batch_x, x], 0)
                batch_y = tf.concat([batch_y, y], 0)
        return batch_x, batch_y


# from keras.utils import Sequence
# import numpy as np
# import tensorflow as tf
# from ASL_Preprocessing import Preprocessing
# import math
# from multiprocessing import Pool

# class DataGenerator(Sequence):
#     def __init__(self, list_IDs, image_directory, batch_size=1, minibatch_size=32, target_size=(64, 64, 25),
#                  data_augmentation=True, shuffle=True, num_workers=4):
#         self.image_directory = image_directory
#         self.list_IDs = list_IDs
#         self.batch_size = batch_size
#         self.target_size = target_size
#         self.shuffle = shuffle
#         self.data_augmentation = data_augmentation
#         self.indexes = np.arange(len(self.list_IDs))
#         self.preprocessing = Preprocessing(data_aug=data_augmentation, norm_intensity=True, imageSize=target_size)
#         self.pool = Pool(num_workers)  # Pool of worker processes
#         self.on_epoch_end()

#     def __len__(self):
#         return math.ceil(len(self.list_IDs) / self.batch_size)

#     def __getitem__(self, index):
#         start = index * self.batch_size
#         end = min((index + 1) * self.batch_size, len(self.list_IDs))
#         batch_indexes = self.indexes[start:end]
#         list_IDs_temp = [self.list_IDs[k] for k in batch_indexes]
#         X, y = self.__data_generation(list_IDs_temp)
#         return X, y

#     def on_epoch_end(self):
#         if self.shuffle:
#             np.random.shuffle(self.indexes)

#     def __data_generation(self, list_IDs_temp):
#         # Prepare data inputs for parallel processing
#         inputs = [(ID, self.image_directory) for ID in list_IDs_temp]
#         results = self.pool.starmap(self.preprocessing.process_case, inputs)
        
#         # Initialize arrays for the entire batch
#         batch_x = np.empty((len(list_IDs_temp), *self.target_size))
#         batch_y = np.empty((len(list_IDs_temp), 1))

#         for i, (x, y) in enumerate(results):
#             batch_x[i, ...] = np.expand_dims(x, axis=-1)
#             batch_y[i, ...] = np.expand_dims(y, axis=-1)

#         return batch_x, batch_y

#     def __del__(self):
#         self.pool.close()  # Properly close the pool when the generator is destroyed


# def __data_generation(self, list_IDs_temp):
#         'Generates data containing batch_size samples'
#         for i, ID in enumerate(list_IDs_temp):
#             x, y = self.preprocessing.process_case(ID, self.image_directory)
#             # Reshape each slice to add a channel dimension
#             x = np.expand_dims(x, axis=0)  
#             y = np.expand_dims(y, axis=0)  
#             # Concatenate data in batch.
#             if i == 0:
#                 batch_x = x
#                 batch_y = y
#             else:
#                 batch_x = tf.concat([batch_x, x], 0)
#                 batch_y = tf.concat([batch_y, y], 0)
        
#         return batch_x, batch_y
