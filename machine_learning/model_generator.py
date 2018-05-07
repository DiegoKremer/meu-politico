import numpy as np
import pandas as pd


class ModelPreprocessing ():
    
    
    def __init___ (self, dataset, label):
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
    
    def split_dataframe(self):
        """
            Split the dataset into three other datasets.
                Training Dataset: 
                    Data which will be used to train the model
                Validation Dataset: 
                    Data which will be used to validate results.
                Test Dataset:
                    Data that will be used to run a last test on 
                    the model with the highest accuracy achieved. 
        """
        train_sample_ratio = 70
        train_sample_volume = (self.dataset.count() * train_sample_ratio) / 100
        training_data = self.dataset.head(train_sample_volume)
        full_test_data = self.dataset.tail(self.dataset.count() - train_sample_volume)
        test_sample_volume = (full_test_data.count() * 70) / 100
        test_data = full_test_data.head(test_sample_volume)
        validation_data = full_test_data.tail(full_test_data.count() - test_sample_volume)
        return training_data, test_data, validation_data
    
    
    def process_label(self):
        label_frame = pd.DataFrame()
        label_frame.assign(self.dataset.at[self.label])
        return label_frame
    
    def set_features():
        return