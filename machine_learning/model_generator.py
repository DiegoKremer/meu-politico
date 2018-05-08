import numpy as np
import pandas as pd
import text_processor as tp
import keras as krs
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation


dataset = pd.read_excel('Dataset.xlsx')

class ModelDataProcessing ():
    """
        Executes a series of tasks to prepare and output the proper data 
        to be consumed by the training function.
        
        Parameters:
            dataset: DataFrame
                Requires a Pandas DataFrame as data input.
            label: String
                The label is the name of the column which containts 
                the target values to be predicted.
        
    """
    def __init__(self, dataset, label):
        self.dataset = dataset
        self.label = label
    
    def clutter_data_order(self):
        """
            Randomize the order of the data in the dataset in case the data
            comes sorted by any argument.
        """
        self.dataset = self.dataset.reindex(
                np.random.permutation(self.dataset.index))
        return self.dataset
    
    def text_processing(self):
        """
            Transforms the column 'TEXTO' column of the dataset into a 
            list of processed words.
        """
        text_processor = tp.TextProcessor(self.dataset,'TEXTO','portuguese')
        text_processor.full_process()
        return self.dataset
    
    def __split_dataframe(self):
        """
            Split the dataset into three other datasets.
                Training Dataset: 
                    Data which will be used to train the model.
                Validation Dataset: 
                    Data which will be used to validate results.
                Test Dataset:
                    Data that will be used to run a last test on 
                    the model with the highest accuracy achieved.
            Returns:
                Three objects of type DataFrame
        """
        
        train_sample_ratio = 70
        train_sample_volume = int((self.dataset.count().values[0] * train_sample_ratio) / 100)
        training_data = self.dataset.head(train_sample_volume)
        full_test_data = self.dataset.tail(int(self.dataset.count().values[0] - train_sample_volume))
        test_sample_volume = int((full_test_data.count().values[0] * 70) / 100)
        test_data = full_test_data.head(test_sample_volume)
        validation_data = full_test_data.tail(int(full_test_data.count().values[0] - test_sample_volume))
        return training_data, test_data, validation_data
    
    
    def assign_training_data(self):
        """
            Creates and assigns to a variable the TRAINING set. 
        """
        split_frame = ModelDataProcessing(self.dataset, self.label).__split_dataframe()[0]
        return split_frame
    
    def assign_test_data(self):
        """
            Creates and assigns to a variable the TEST set.
        """
        split_frame = ModelDataProcessing(self.dataset, self.label).__split_dataframe()[1]
        return split_frame

    def assign_validation_data(self):
        """
            Creates and assigns to a variable the VALIDATION set. 
        """
        split_frame = ModelDataProcessing(self.dataset, self.label).__split_dataframe()[2]
        return split_frame
    

    def extract_label(self):
        """
            Extract the labels of the dataset to be used separately.
        """
        label_frame = self.dataset[self.label]
        return label_frame
    
    def extract_features(self):
        """
            Extract the features of the dataset to be used separately.
        """
        feature_frame = self.dataset.drop(self.label, axis=1)
        return feature_frame
    
    def num_categorizer(self, column):
        """
            Transforms a categorical column of type String to type Integer.
            
            Parameters:
                column: The name of the column of categorical values in the
                dataset that needs to be converted to integer type.     
        """
        self.dataset[str(column)] = pd.Categorical(self.dataset[column])
        self.dataset[str(column)] = self.dataset[str(column)].cat.codes
        return self.dataset
    
    def create_one_hot_matrix():
        return
    
    
model_processor = ModelDataProcessing(dataset,'AREAS_TEMATICAS_APRESENTACAO')
training_v = model_processor.assign_training_data()
test_v = model_processor.assign_test_data()
validation_v = model_processor.assign_validation_data()

features = model_processor.extract_features()
label = model_processor.extract_label()
label = model_processor.num_categorizer('AREAS_TEMATICAS_APRESENTACAO')