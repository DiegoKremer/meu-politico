import nltk as nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.corpus import stopwords

#Importa o dataset de textos como um novo dataframe
df_prop = pd.read_excel('Dataset.xlsx')


class TextPreprocessing ():
    """ 
        
    """
    #Faz a tokenização da coluna TEXTO criando subsequentemente uma nova
    #coluna no dataset com os textos tokenizados. 
    def __init__(self, dataset, column):
        self.dataset = dataset
        self.column = column
        self.tokenized = 'TOKENIZED'
    
    def TokenizeText (self): 
        self.dataset = self.dataset.assign(TOKENIZED=pd.Series().astype(list))
        for idx, text in self.dataset.iterrows():
            if not text[self.column]:
                print('O texto a ser tokenizado é do tipo None. Resetando para string vazia.')
                text[self.column] = ''
            self.dataset.at[self.dataset.index[idx], self.tokenized] = nltk.word_tokenize(text[self.column])
        return self.dataset
    
    #Remove as stopwords dos textos tokenizados
    def RemoveStopwords (self, language):
        stop_words = set(stopwords.words(language))
        for idx, row in self.dataset.iterrows():
            for word in row[self.tokenized]:
                if word in stop_words:
                    self.dataset.at[self.dataset.index[idx], self.tokenized] = [w for w in row[self.tokenized] if not w in stop_words]        
        return self.dataset


    #Remove itens de pontuação do texto
    def RemoveNonAlphabetical (self):
        for idx, row in self.dataset.iterrows():
            self.dataset.at[self.dataset.index[idx], self.tokenized] = [word for word in row[self.tokenized] if word.isalpha()]
        return self.dataset

    #Converte todas as palavras para minusculas para que não haja diferenciação 
    #entre duas palavras iguais
    def SetAllLowerCase (self):
        for idx, row in self.dataset.iterrows():
            self.dataset.at[self.dataset.index[idx], self.tokenized] = [w.lower() for w in row[self.tokenized]]
        return self.dataset

    #Remove sufixos das palavras para encontrar uma forma raíz da palavra e 
    #eliminar variações de uma mesma palavra
    def Stemming (self):
        stemmer = nltk.stem.RSLPStemmer()
        for idx, row in self.dataset.iterrows():
            self.dataset.at[self.dataset.index[idx], self.tokenized] = [stemmer.stem(word) for word in row[self.tokenized]]
        return self.dataset
    
    def getDataset(self):
        return self.dataset
    
    def FullProcess(self, language):
        processor = TextPreprocessing(self.dataset, self.column)
        processed_dataset = processor.TokenizeText()
        processed_dataset = processor.RemoveStopwords(language)
        processed_dataset = processor.RemoveNonAlphabetical()
        processed_dataset = processor.SetAllLowerCase
        processed_dataset = processor.Stemming()
        return processed_dataset
        



processador = TextPreprocessing(df_prop, 'TEXTO')
processed_data = processador.FullProcess('portuguese')



