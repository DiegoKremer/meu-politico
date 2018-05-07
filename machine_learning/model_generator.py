import numpy as np
import nltk as nltk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf


class ModelPreprocessing ():
    
    def __init___ (self, dataset, label):
        self.dataset = dataset
        self.label = label
    
    
    def split_dataframe():
        self.dataset
        training_data = self.dataset.iloc[]
        test_data = 
        validation_data = 
        return training_data, test_data, validation_data
    
    
    def process_label():
        label_frame = pd.DataFrame()
        label_frame.assign(self.dataset.at[self.label])
        return label_frame
    
    def set_features():
        return