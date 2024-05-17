from keras.utils import Sequence
import numpy as np
import tensorflow as tf
import pandas as pd
import os
from configuration import Configuration
    
# Define the path to your dataset and set experiment parameters
path = '/home/xurbano/QEI-ASL/data'
params = {
    'pathData': path,
    'targetSize': (64, 64, 25),  # Adjust based on your actual data
    'batchSize': 5,
    'currentFold': 1  # Assuming you're testing with the first fold
}

# Initialize the Configuration with specified parameters
config = Configuration(**params)

# Use the Configuration to create DataGenerators for training and validation
train_generator, validation_generator = config.createAllDataGenerators()

# Optionally, fetch a batch from the train_generator and validation_generator to test
X_train, y_train = train_generator.__getitem__(0)  # Fetch the first batch from the training generator
X_val, y_val = validation_generator.__getitem__(0)  # Fetch the first batch from the validation generator

# Print shapes of the fetched batches to verify
print(f"Training batch X shape: {X_train.shape}, y shape: {y_train.shape}")
print(f"Validation batch X shape: {X_val.shape}, y shape: {y_val.shape}")

print(X_train)
print(y_train)
