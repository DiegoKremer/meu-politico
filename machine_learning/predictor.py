# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 20:17:15 2018

@author: TCC_ADS
"""

import pandas as pd
import text_processor as tp
from sklearn.externals import joblib

#Importa arquivo com novas entradas
entries = pd.read_excel('projetos.xlsx')

#Processa o texto
text_processor = tp.TextProcessor(entries,'TEXTO','portuguese')
proc_entries = text_processor.full_process()
proc_entries = proc_entries['TOKENIZED']

#Carrega modelo
print('Loading model...')
text_clf_trained = joblib.load('svm_model.pkl')

#Exibe configuração do modelo carregado
print(text_clf_trained)

#Testa modelo recarregado
predicted = text_clf_trained.predict(proc_entries)
print('\nInitial data:')
print(entries)
print('\nProcessed data:')
print(proc_entries)
print('\nPrediction values:')
print(predicted)