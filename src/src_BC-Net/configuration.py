# -----------------------------------------------------------------------------
# Configuration file
# Author: Xavier Beltran Urbano
# Date Created: 21-02-2024
# -----------------------------------------------------------------------------

import os
from dataGenerator import DataGenerator
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random


class Configuration():
    def __init__(self, pathData, targetSize, batchSize, currentFold):
        self.pathData = pathData  # Path to the data directory
        self.targetSize = targetSize  # Target size of the images
        self.batchSize = batchSize  # Batch size for the data generator
        self.allIDS = [filename for filename in os.listdir(self.pathData)
                       if filename != ".DS_Store" and not filename.endswith(".xls") and not filename.endswith(".xlsx")]
        self.currentFold = currentFold  # Current fold number for cross-validation
        self.createFolds()  # Create training and validation folds

    def readAnnotation(self):
        # Read the annotations from the Ratings.xlsx file
        ratings_df = pd.read_excel(os.path.join(self.pathData, 'Binarized_Ratings.xlsx'))
        print(len(ratings_df))
        return ratings_df
    
    def categorizeRatings(self, ratings_dict):
        # Categorize ratings into binary values
        categories = {'0': [], '1': []}
        for name, rating in zip(ratings_dict['Name'], ratings_dict['Binarized']):
            if rating == 0:
                categories['0'].append(name)
            elif rating == 1:
                categories['1'].append(name)
        return categories


    def valDataKFolds(self, categories):
        # Split the data into 5 folds for cross-validation
        kfold = {i: [] for i in range(5)}
        for category in categories:
            vecNames = categories[category]
            random.seed(48)
            random.shuffle(vecNames)
            nFold = len(vecNames) // 5
            for i in range(5):
                startIndex = i * nFold
                endIndex = startIndex + nFold
                if i == 4:
                    endIndex = len(vecNames)
                kfold[i].extend(vecNames[startIndex:endIndex])
                startIndex = endIndex
        return kfold

    def createFolds(self):
        # Create training and validation folds
        allRatings = self.readAnnotation()
        categories = self.categorizeRatings(allRatings)
        self.valFolds = self.valDataKFolds(categories)
        self.trainFolds = {i: [] for i in range(5)}
        for i in range(5):
            trainNames = [name for name in allRatings['Name'] if name not in self.valFolds[i]]
            self.trainFolds[i] = trainNames

    def createDataGenerator(self, listIDS, dataAugmentation,shuffle):
        # Create a data generator
        data_generator = DataGenerator(
            image_directory=self.pathData,
            list_IDs=listIDS,
            batch_size=self.batchSize,
            target_size=self.targetSize,
            data_augmentation=dataAugmentation,
            shuffle=shuffle)
        return data_generator

    def createAllDataGenerators(self):
        # Create data generators for the current fold
        trainingIDS,validationIDS=self.trainFolds[self.currentFold-1],self.valFolds[self.currentFold-1]
        train_generator = self.createDataGenerator(trainingIDS, dataAugmentation=True, shuffle=True)
        validation_generator = self.createDataGenerator(validationIDS, dataAugmentation=False, shuffle=False)
        return train_generator, validation_generator

    def returnVal_IDS(self):
        # Return the validation IDs for the current fold
        return self.valFolds[self.currentFold - 1]

    def returnIDS(self):
        # Return all IDs for the current fold
        return self.trainFolds[self.currentFold - 1] + self.valFolds[self.currentFold - 1]

