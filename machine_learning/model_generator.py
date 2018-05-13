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


#def assign_x_train(self):
#    """
#        Creates and assigns to a variable the X TRAINING (Features) set. 
#    """
#    split_frame = ModelDataProcessing(self.dataset, self.label).split_dataframe()[0]
#    x_train = ModelDataProcessing(split_frame, self.label).extract_features('TOKENIZED')
#    return x_train
#
#def assign_y_train(self):
#    """
#        Creates and assigns to a variable the Y TRAINING (Labels) set. 
#    """
#    split_frame = ModelDataProcessing(self.dataset, self.label).split_dataframe()[0]
#    y_train = ModelDataProcessing(split_frame, self.label).extract_label()
#    return y_train
#
#
#def assign_x_test(self):
#    """
#        Creates and assigns to a variable the X TEST (Features) set.
#    """
#    split_frame = ModelDataProcessing(self.dataset, self.label).split_dataframe()[1]
#    x_test = ModelDataProcessing(split_frame, self.label).extract_features('TOKENIZED')
#    return x_test
#
#
#def assign_y_test(self):
#    """
#        Creates and assigns to a variable the Y TEST (Labels) set.
#    """
#    split_frame = ModelDataProcessing(self.dataset, self.label).split_dataframe()[1]
#    y_train = ModelDataProcessing(split_frame, self.label).extract_label()
#    return y_train


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
    tokenizer = Tokenizer(num_words=7000)
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


#Build X datasets
#Break the dataset into single features for specific processing and then
#rebuild it with the matrix values
features = ['SEXO', 'POLITICO', 'PARTIDO', 'UF', 'REGIAO', 'TOKENIZED']
datasets = [train, test]
proc_feat_frame = []
c = 0
for dataset in datasets:
    c += 1
    for feature in features:
        feat_frame = extract_feature(dataset, feature)
        if feature == 'TOKENIZED':
            feat_frame = text_to_matrix(feat_frame)
        else:
            feat_frame = cat_encode_and_vectorize(feat_frame)
        if proc_feat_frame == []:
            proc_feat_frame = feat_frame
        else:
            proc_feat_frame = np.hstack((proc_feat_frame, feat_frame))
    if c == 1: x_train = proc_feat_frame
    else: x_test = proc_feat_frame



#Builds the model
print('Building Model Layers...')
model = Sequential()
print('Adding Dense layer...')
model.add(Dense(128, input_shape=(8483,), activation='relu'))
model.add(Dropout(0.50))
print('Adding Dense layer...')
model.add(Dense(64, input_shape=(8483,), activation='sigmoid'))
model.add(Dropout(0.50))
print('Adding Dense layer...')
model.add(Dense(36, activation='softmax'))
sgd = SGD(lr=0.7, decay=1e-6, momentum=0.9, nesterov=True)
print('Compiling Model...')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

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

#y_pred = model.predict(x_test)
#print(y_pred)
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)