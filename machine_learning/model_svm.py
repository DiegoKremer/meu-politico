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
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

#Importa o Dataset
print('Importing Dataset...')
dataset = pd.read_excel('Dataset_pre_processed.xlsx')

#Transforma os dados de treino e label em matrizes 
print('Setting as matrix...')
numpy_array = dataset.as_matrix()
X = numpy_array[:,0]
Y = numpy_array[:,1]

#Separa os dadps em treino e teste
print('Splitting...')
X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                    Y, 
                                                    test_size=0.25, 
                                                    random_state=32)

#Transforma as palavras dos textos em vetores
print('Count Vectorizer...')
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
X_train_counts.shape

#Normaliza os pesos da matriz gerados pela frequência de termos
print('TFID Transformer...')
tfidf_transf = TfidfTransformer()
X_train_tfidf = tfidf_transf.fit_transform(X_train_counts)
X_train_tfidf.shape

#Cria o classificador em um pipeline
print('Generating classifier...')

#Support Vector Machine
classifier = Pipeline([('cvec', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('svm', SGDClassifier(loss='squared_hinge', 
                                               penalty='l2',
                                               alpha=1e-3, 
                                               max_iter=150, 
                                               random_state=32,
                                               learning_rate='constant',
                                               eta0=0.001))])

##Naive Bayes
#classifier = Pipeline([('cvec', CountVectorizer()),
#                         ('tfidf', TfidfTransformer()),
#                         ('nb', MultinomialNB())])


##One Versus Rest: Exige muito poder computacional e tempo para executar
#classifier = Pipeline([('cvec', CountVectorizer()),
#                     ('tfidf', TfidfTransformer()),
#                     ('onevsrest', OneVsRestClassifier(SVC(kernel='linear'), n_jobs=-1))])
     

#Executa o treino
print('Fitting to Classifier...')
classifier = classifier.fit(X_train, Y_train)

#Efetua a predição nos dados de testes e calcula média de predição
print('Predicting...')
predicted = classifier.predict(X_test)
print(np.mean(predicted == Y_test))

#GridSearchCV: Faz uma busca exaustiva pela melhor combinação de parametros
#Observação: Demanda alto poder de processamento e tempo
#Padrão comentado para não executar
#print('GridSearch')
#parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 
#                                    'tfidf__use_idf': (True, False), 
#                                    'clf__alpha': (1e-2, 1e-3)}
#
#gs_clf_svm = GridSearchCV(text_clf_svm, parameters, n_jobs=-1)
#gs_clf_svm = gs_clf_svm.fit(X_train, Y_train)
#print(gs_clf_svm.best_score_)
#print(gs_clf_svm.best_params_)

#Salva no disco o modelo em um arquivo pkl
#print('Saving model...')
#joblib.dump(text_clf_svm, 'svm_model.pkl') 