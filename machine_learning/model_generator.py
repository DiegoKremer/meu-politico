import numpy as np
import pandas as pd
import text_processor as tp
import keras as krs
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from keras.optimizers import SGD


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
        print('Booting up Model Data Processor')
        self.dataset = dataset
        self.label = label
    
    def clutter_data_order(self):
        """
            Randomize the order of the data in the dataset in case the data
            comes sorted by any argument.
        """
        print('Cluttering the data...')
        self.dataset = self.dataset.reindex(
                np.random.permutation(self.dataset.index))
        return self.dataset
    
    def text_processing(self):
        """
            Transforms the column 'TEXTO' column of the dataset into a 
            list of processed words.
        """
        text_processor = tp.TextProcessor(self.dataset,'TEXTO','portuguese')
        self.dataset = text_processor.full_process()
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
        print('Splitting data frames...')
        train_sample_ratio = 70
        train_sample_volume = int((self.dataset.count().values[0] * train_sample_ratio) / 100)
        training_data = self.dataset.head(train_sample_volume)
        full_test_data = self.dataset.tail(int(self.dataset.count().values[0] - train_sample_volume))
        test_sample_volume = int((full_test_data.count().values[0] * 70) / 100)
        test_data = full_test_data.head(test_sample_volume)
        validation_data = full_test_data.tail(int(full_test_data.count().values[0] - test_sample_volume))
        return training_data, test_data, validation_data
    
    
    def assign_x_train(self):
        """
            Creates and assigns to a variable the X TRAINING (Features) set. 
        """
        split_frame = ModelDataProcessing(self.dataset, self.label).__split_dataframe()[0]
        x_train = ModelDataProcessing(split_frame, self.label).extract_features('TOKENIZED')
        return x_train
    
    def assign_y_train(self):
        """
            Creates and assigns to a variable the Y TRAINING (Labels) set. 
        """
        split_frame = ModelDataProcessing(self.dataset, self.label).__split_dataframe()[0]
        y_train = ModelDataProcessing(split_frame, self.label).extract_label()
        return y_train
    
    
    def assign_x_test(self):
        """
            Creates and assigns to a variable the X TEST (Features) set.
        """
        split_frame = ModelDataProcessing(self.dataset, self.label).__split_dataframe()[1]
        x_test = ModelDataProcessing(split_frame, self.label).extract_features('TOKENIZED')
        return x_test
    
    
    def assign_y_test(self):
        """
            Creates and assigns to a variable the Y TEST (Labels) set.
        """
        split_frame = ModelDataProcessing(self.dataset, self.label).__split_dataframe()[1]
        y_train = ModelDataProcessing(split_frame, self.label).extract_label()
        return y_train

#   *Checking if this will be deprecated*
#    def assign_validation_data(self):
#        """
#            Creates and assigns to a variable the VALIDATION set. 
#        """
#        split_frame = ModelDataProcessing(self.dataset, self.label).__split_dataframe()[2]
#        return split_frame
    

    def extract_label(self):
        """
            Extract the labels of the dataset to be used separately.
        """
        print('Extracting labels...')
        label_frame = self.dataset[self.label]
        return label_frame
    
    def extract_features(self, feature):
        """
            Extract the features of the dataset to be used separately.
        """
        print('Extracting features...')
        feature_frame = self.dataset[feature]
#        feature_frame = self.dataset.drop(self.label, axis=1)
        return feature_frame
    
    def num_categorizer(self, column):
        """
            Transforms a categorical column of type String to type Integer.
            
            Parameters:
                column: The name of the column of categorical values in the
                dataset that needs to be converted to integer type.     
        """
        print('Transforming text to numeric categories...')
        self.dataset[str(column)] = pd.Categorical(self.dataset[column])
        self.dataset[str(column)] = self.dataset[str(column)].cat.codes
        return self.dataset
    
    
    def num_classes(self):
        """
            Get the total number of classes present of a categorical label.
        """
        print('Getting total number of classes...')
        num_classes = self.dataset[self.label].unique().max() + 1
        return num_classes
    
    def data_to_matrix(self, data):
        """
            Transforms the input data into a binary matrix
        """
        tokenizer = Tokenizer(num_words=1000)
        data = tokenizer.sequences_to_matrix(data, mode='binary')
        return data
    
    def process_pipeline():
        return
    
    
# TESTING
#Booting up class
model_processor = ModelDataProcessing(dataset,'AREAS_TEMATICAS_APRESENTACAO')

# Clutter method test
cluttered_dataset = model_processor.clutter_data_order()

# Text process method test
processed_text = model_processor.text_processing()

#Dataset creation tests
x_train = model_processor.assign_x_train()
y_train = model_processor.assign_y_train()

x_test = model_processor.assign_x_test()
y_test = model_processor.assign_y_test()

#test_v = model_processor.assign_test_data()
#validation_v = model_processor.assign_validation_data()
#
#features = model_processor.extract_features()
#label = model_processor.extract_label()
#label = model_processor.num_categorizer('AREAS_TEMATICAS_APRESENTACAO')
#label_totalclasses = model_processor.num_classes()




class ModelTrainer():
    """
        Creates the model and train it.
        
        Parameters:
            training_features: 
                An array of the features to be used in prediction.
            training_labels:
                An array with the labels to be used in prediction.
        
        Returns:
            A trained model.
    """    
    def __init__(self, 
                 training_features, 
                 training_labels,
                 test_features,
                 test_labels):
        self.training_features = training_features
        self.training_labels = training_labels
        self.test_features = test_features
        self.test_labels = test_labels
        
    
    def input_fn():
        return
    
    def build_model_layers(self):
        self.model = Sequential()
        self.model.add(Dense(512, input_shape=(1024,), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2, activation='softmax'))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        return self.model
        
    def fit_model(self):
        self.model.fit(self.training_features, self.training_labels,
                  batch_size=64,
                  epochs=10,
                  verbose=1,
                  validation_split=0.1,
                  shuffle=True)
        return self.model
    
    def evaluate_model(self):
        score = self.model.evaluate(self.test_features, self.test_labels, batch_size=128)
        return score
    