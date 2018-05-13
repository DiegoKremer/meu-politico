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


def clutter_data_order(dataset):
    """
        Randomize the order of the data in the dataset in case the data
        comes sorted by any argument.
    """
    print('Cluttering the data...')
    dataset = dataset.reindex(
            np.random.permutation(dataset.index))
    return dataset

def text_processing(data):
    """
        Transforms the column 'TEXTO' column of the dataset into a 
        list of processed words.
    """
    text_processor = tp.TextProcessor(data,'TEXTO','portuguese')
    data = text_processor.full_process()
    return data

def split_dataframe(dataset):
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
    train_sample_volume = int((dataset.count().values[0] * train_sample_ratio) / 100)
    training_data = dataset.head(train_sample_volume)
    test_data = dataset.tail(int(dataset.count().values[0] - train_sample_volume))
    return training_data, test_data


def extract_label(dataset, label):
    """
        Extract the labels of the dataset to be used separately.
    """
    print('Extracting labels...')
    label_frame = dataset[label]
    dataset.drop(label, axis=1)
    return label_frame

def extract_feature(dataset, feature):
    """
        Extract the features of the dataset to be used separately.
    """
    print('Extracting features...')
    feature_frame = dataset[feature]
    return feature_frame

def num_categorizer(dataset, column):
    """
        Transforms a categorical column of type String to type Integer.
        
        Parameters:
            column: The name of the column of categorical values in the
            dataset that needs to be converted to integer type.     
    """
    print('Transforming text to numeric categories...')
    dataset[str(column)] = pd.Categorical(dataset[column])
    dataset[str(column)] = dataset[str(column)].cat.codes
    return dataset

def category_to_matrix(data):
    num_of_classes = num_classes(data)
    data = krs.utils.to_categorical(data, num_of_classes)
    return


def num_classes(dataset, label):
    """
        Get the total number of classes present of a categorical label.
    """
    num_classes = dataset[label].unique().max() + 1
    return num_classes


def text_to_matrix(data):
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

def cat_encode_and_vectorize(data):
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


#Pipeline the processing of the dataset
    
#Clutter the dataset order to avoid issues with sorted data
dataset = clutter_data_order(dataset)

#Process text in Portuguese language
dataset = text_processing(dataset)

#Split the dataset in train and test
train, test = split_dataframe(dataset)

#Extract label 
y_train = extract_label(train, 'TEMAS')
y_test = extract_label(test, 'TEMAS')

#Encode and Vectorize Y datasets
y_train = cat_encode_and_vectorize(y_train)
y_test = cat_encode_and_vectorize(y_test)


#Extract each feature from the TRAIN dataset, encode it and vectorize 
#to a binary array
f_sexo = cat_encode_and_vectorize(extract_feature(train, 'SEXO'))
f_politico = cat_encode_and_vectorize(extract_feature(train, 'POLITICO'))
f_partido = cat_encode_and_vectorize(extract_feature(train, 'PARTIDO'))
f_uf = cat_encode_and_vectorize(extract_feature(train, 'UF'))
f_regiao = cat_encode_and_vectorize(extract_feature(train, 'REGIAO'))
f_tokenized = text_to_matrix(extract_feature(train, 'TOKENIZED'))
            
#Stacking the processed arrays back into a single file
x_train = np.hstack((f_sexo, 
                     f_politico, 
                     f_partido, 
                     f_uf, 
                     f_regiao,
                     f_tokenized))

#Extract each feature from the TEST dataset, encode it and vectorize 
#to a binary array
f_sexo = cat_encode_and_vectorize(extract_feature(test, 'SEXO'))
f_politico = cat_encode_and_vectorize(extract_feature(test, 'POLITICO'))
f_partido = cat_encode_and_vectorize(extract_feature(test, 'PARTIDO'))
f_uf = cat_encode_and_vectorize(extract_feature(test, 'UF'))
f_regiao = cat_encode_and_vectorize(extract_feature(test, 'REGIAO'))
f_tokenized = text_to_matrix(extract_feature(test, 'TOKENIZED'))

#Stacking the processed arrays back into a single file
x_test = np.hstack((f_sexo, 
                    f_politico, 
                    f_partido, 
                    f_uf, 
                    f_regiao,
                    f_tokenized))

#Clear unnecessary variable values from memory for faster processing
dataset = None
train = None
test = None
f_sexo = None
f_politico = None
f_partido = None
f_uf = None
f_regiao = None
f_tokenized = None

#Check array shape
array_shape = x_train.shape[1]


#Builds the model
print('Building Model Layers...')
model = Sequential()
print('Adding Dense layer...')
model.add(Dense(128, input_shape=(array_shape,), activation='relu'))
model.add(Dropout(0.50))
print('Adding Dense layer...')
model.add(Dense(64, input_shape=(array_shape,), activation='relu'))
model.add(Dropout(0.50))
print('Adding Dense layer...')
model.add(Dense(36, activation='softmax'))
sgd = SGD(lr=0.7, decay=1e-6, momentum=0.9, nesterov=True)
print('Compiling Model...')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#Fits and train the model
model.fit(x_train, y_train,
          batch_size=64,
          epochs=40,
          verbose=1,
          validation_split=0.2,
          shuffle=True)

score = model.evaluate(x_test, 
                       y_test, 
                       batch_size=256)
print('Test score:', score[0])
print('Test accuracy:', score[1])