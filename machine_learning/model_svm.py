# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 13:45:15 2018

@author: Diego Kremer
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

#Imports Dataset
print('Importing Dataset')
dataset = pd.read_excel('Dataset_pre_processed.xlsx')

print('Setting as matrix')
numpy_array = dataset.as_matrix()
X = numpy_array[:,0]
Y = numpy_array[:,1]

print('Splitting')
X_train, X_test, Y_train, Y_test = train_test_split(
X, Y, test_size=0.3, random_state=42)

print('Count Vectorizer')
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
X_train_counts.shape

print('TFID Transformer')
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

print('Classifier')
text_clf_svm = Pipeline([('vect', CountVectorizer()), 
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='squared_hinge', 
                                                   penalty='l2',
                                                   alpha=1e-3, 
                                                   max_iter=5, 
                                                   random_state=42,
                                                   learning_rate='constant',
                                                   eta0=0.01))])

print('Fitting to Classifier')
text_clf_svm = text_clf_svm.fit(X_train, Y_train)

print('Predicting')
predicted = text_clf_svm.predict(X_test)
print(np.mean(predicted == Y_test))
print(predicted)

print('GridSearch')
parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 
                                    'tfidf__use_idf': (True, False), 
                                    'clf__alpha': (1e-2, 1e-3)}

gs_clf_svm = GridSearchCV(text_clf_svm, parameters, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(X_train, Y_train)
print(gs_clf_svm.best_score_)
print(gs_clf_svm.best_params_)
