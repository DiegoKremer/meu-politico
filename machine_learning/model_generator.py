import numpy as np
import pandas as pd
import text_processor as tp
import keras as krs
import keras.preprocessing.text as kpt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder


dataset = pd.read_excel('Dataset_processed.xlsx')

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
        train_sample_ratio = 80
        train_sample_volume = int((self.dataset.count().values[0] * train_sample_ratio) / 100)
        training_data = self.dataset.head(train_sample_volume)
        test_data = self.dataset.tail(int(self.dataset.count().values[0] - train_sample_volume))
#        test_sample_volume = int((full_test_data.count().values[0] * 70) / 100)
#        test_data = full_test_data.head(test_sample_volume)
#        validation_data = full_test_data.tail(int(full_test_data.count().values[0] - test_sample_volume))
        return training_data, test_data
    
    
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
    
    def category_to_matrix(self, data):
        num_of_classes = self.num_classes(data)
        data = krs.utils.to_categorical(data, num_of_classes)
        return
    
    
    def num_classes(self):
        """
            Get the total number of classes present of a categorical label.
        """
        num_classes = self.dataset[self.label].unique().max() + 1
        return num_classes
    
    
    def data_to_matrix(self, data):
        """
            Transforms the text input data into a binary matrix
        """
        tokenizer = Tokenizer(num_words=6000)
        tokenizer.fit_on_texts(data)
        dictionary = tokenizer.word_index
        def convert_text_to_index_array(text):
            return [dictionary[word] for word in kpt.text_to_word_sequence(text)]
        allWordIndices = []
        for text in data:
            wordIndices = convert_text_to_index_array(text)
            allWordIndices.append(wordIndices)
        allWordIndices = np.asarray(allWordIndices)
        data = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
        return data
    
    def cat_encode_and_vectorize(self, data):
        """
            Transforms a string categorical column into a encoded label and 
            then output it as a one hot encoded matrix.
        """
        data = data.values.astype(str)
        encoder = LabelEncoder()
        encoder.fit(data)
        encoded_data = encoder.transform(data)
        one_hot_encoded = krs.utils.to_categorical(encoded_data)
        return one_hot_encoded
    
    def process_pipeline():
        return


    
# TESTING
#Booting up class
model_processor = ModelDataProcessing(dataset,'TEMAS')

# Clutter method test
cluttered_dataset = model_processor.clutter_data_order()

# Text process method test
#processed_text = model_processor.text_processing()
#processed_text.to_excel('Processed_File.xlsx', sheet_name = 'Dataset')


test_add_feat = model_processor.cat_encode_and_vectorize(dataset['SEXO'])
print(test_add_feat)

#num_categorizer = model_processor.num_categorizer('TEMAS')
#num_classes = model_processor.num_classes()
#
##Dataset creation tests
#x_train = model_processor.assign_x_train()
#y_train = model_processor.assign_y_train()
#
#x_test = model_processor.assign_x_test()
#y_test = model_processor.assign_y_test()
#
##data_to_matrix tests
#x_train = model_processor.data_to_matrix(x_train)
#x_test = model_processor.data_to_matrix(x_test)

#categorical 


#y_train = krs.utils.to_categorical(y_train, num_classes)
#y_test = krs.utils.to_categorical(y_test, num_classes)

#from imblearn.over_sampling import randomoversampler
#ros = randomoversampler(random_state=0)
#x_t_resampled, y_t_resampled = ros.fit_sample(x_train, y_train)


#test_v = model_processor.assign_test_data()
#validation_v = model_processor.assign_validation_data()
#
#features = model_processor.extract_features()
#label = model_processor.extract_label()
#label = model_processor.num_categorizer('TEMAS')
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
        print('Booting up Model Trainer')
        self.training_features = training_features
        self.training_labels = training_labels
        self.test_features = test_features
        self.test_labels = test_labels
        
    
    def input_fn():
        return
    
    def build_model_layers(self):
        print('Building Model Layers...')
        self.model = Sequential()
        print('Adding Dense layer...')
        self.model.add(Dense(512, input_shape=(6000,), activation='relu'))
        self.model.add(Dropout(0.5))
        print('Adding Dense layer...')
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        print('Adding Dense layer...')
        self.model.add(Dense(32, activation='softmax'))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        print('Compiling Model...')
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return self.model
        
    def fit_model(self):
        self.model.fit(self.training_features, self.training_labels,
                  batch_size=32,
                  epochs=5,
                  verbose=1,
                  validation_split=0.1,
                  shuffle=True)
        return self.model
    
    def evaluate_model(self):
        score = self.model.evaluate(self.test_features, self.test_labels, batch_size=32)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        return score


#trainer = ModelTrainer(x_train, y_train, x_test, y_test)
#trainer.build_model_layers()
#trainer.fit_model()
#trainer.evaluate_model()

##TEST WITHOUT CLASSES
#
#print('Building Model Layers...')
#model = Sequential()
#print('Adding Dense layer...')
#model.add(Dense(64, input_shape=(6000,), activation='relu'))
#model.add(Dropout(0.25))
#print('Adding Dense layer...')
#model.add(Dense(64, input_shape=(6000,), activation='relu'))
#model.add(Dropout(0.25))
#print('Adding Dense layer...')
#model.add(Dense(num_classes, activation='softmax'))
#sgd = SGD(lr=0.7, decay=1e-6, momentum=0.9, nesterov=True)
#print('Compiling Model...')
#model.compile(loss='categorical_crossentropy',
#              optimizer=sgd,
#              metrics=['accuracy'])
#
#
#model.fit(x_train, y_train,
#          batch_size=64,
#          epochs=30,
#          verbose=1,
#          validation_split=0.1,
#          shuffle=True)
#
#score = model.evaluate(x_test, 
#                       y_test, 
#                       batch_size=512)
#print('Test score:', score[0])
#print('Test accuracy:', score[1])

#y_pred = model.predict(x_test)
#print(y_pred)
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)