import nltk as nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.corpus import stopwords

#Importa o dataset de textos como um novo dataframe
df_prop = pd.read_excel('Dataset.xlsx')




#Faz a tokenização da coluna TEXTO criando subsequentemente uma nova
#coluna no dataset com os textos tokenizados. 
def TokenizeText (dataset):
    dataset = dataset.assign(TOKENIZADO=pd.Series().astype(list))
    for idx, text in dataset.iterrows():
        if not text['TEXTO']:
            print('O texto a ser tokenizado é do tipo None. Resetando para string vazia.')
            text['TEXTO'] = ''
        dataset.at[dataset.index[idx], 'TOKENIZADO'] = nltk.word_tokenize(text['TEXTO'])
    return dataset
    
#Remove as stopwords dos textos tokenizados
def RemoveStopwords (dataset, language):
    stop_words = set(stopwords.words(language))
    for idx, row in dataset.iterrows():
        for word in row['TOKENIZADO']:
            if word in stop_words:
                dataset.at[dataset.index[idx], 'TOKENIZADO'] = [w for w in row['TOKENIZADO'] if not w in stop_words]        
    return dataset


#Remove itens de pontuação do texto
def RemovePunctuation (dataset):
    for idx, row in dataset.iterrows():
        dataset.at[dataset.index[idx], 'TOKENIZADO'] = [word for word in row['TOKENIZADO'] if word.isalpha()]
    return dataset

#Converte todas as palavras para minusculas para que não haja diferenciação 
#entre duas palavras iguais
def SetAllLowerCase (dataset):
    for idx, row in dataset.iterrows():
        dataset.at[dataset.index[idx], 'TOKENIZADO'] = [w.lower() for w in row['TOKENIZADO']]
    return dataset

#Remove sufixos das palavras para encontrar uma forma raíz da palavra e 
#eliminar variações de uma mesma palavra
def Stemming (dataset):
    stemmer = nltk.stem.RSLPStemmer()
    for idx, row in dataset.iterrows():
        dataset.at[dataset.index[idx], 'TOKENIZADO'] = [stemmer.stem(word) for word in row['TOKENIZADO']]
    return dataset
    

#Passa o dataset pelas funções de limpeza do arquivo
df_prop_tokenized = TokenizeText(df_prop)
df_prop_no_sw = RemoveStopwords(df_prop_tokenized, 'portuguese')
df_prop_no_pun = RemovePunctuation (df_prop_no_sw)
df_prop_lower = SetAllLowerCase(df_prop_no_pun)
df_prop_stem = Stemming(df_prop_lower)

