import nltk as nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.corpus import stopwords


df_prop = pd.read_excel('Dataset.xlsx')


class TextPreprocessing ():
    """ 
        Performs preprocessing functionalities on a given column of a dataframe.
        
        Parameters:
            dataset : DataFrame
                Requires a Pandas DataFrame as data input.
            column : String
                Is the name of the column which will serve as the data source 
                for the text processing.
            language : String
                Defines the language of source text for processing.
                **ONLY SUPPORTS PORTUGUESE**
                
        Returns:
            DataFrame
                DataFrame with a new column with the processed text
        
    """

    def __init__(self, dataset, column, language):
        self.dataset = dataset
        self.column = column
        self.language = language
        self.tokenized = 'TOKENIZED'
    
    def TokenizeText (self): 
        """
            Tokenizes a text. To tokenize is to break a text into an list of 
            words.
        """
        self.dataset = self.dataset.assign(TOKENIZED=pd.Series().astype(list))
        for idx, text in self.dataset.iterrows():
            if not text[self.column]:
                print('O texto a ser tokenizado é do tipo None. Resetando para string vazia.')
                text[self.column] = ''
            self.dataset.at[self.dataset.index[idx], self.tokenized] = nltk.word_tokenize(text[self.column])
        return self.dataset
    
    #Remove as stopwords dos textos tokenizados
    def RemoveStopwords (self):
        """
            Removes all stopwords (words that have no significance)
            from the text for a given language.
        """
        stop_words = set(stopwords.words(self.language))
        for idx, row in self.dataset.iterrows():
            for word in row[self.tokenized]:
                if word in stop_words:
                    self.dataset.at[self.dataset.index[idx], self.tokenized] = [w for w in row[self.tokenized] if not w in stop_words]        
        return self.dataset


    #Remove itens de pontuação do texto
    def RemoveNonAlphabetical (self):
        """
            Removes all words from the list that are not alphabetical. That 
            includes numbers, punctuations and empty spaces. 
        """
        for idx, row in self.dataset.iterrows():
            self.dataset.at[self.dataset.index[idx], self.tokenized] = [word for word in row[self.tokenized] if word.isalpha()]
        return self.dataset

    #Converte todas as palavras para minusculas para que não haja diferenciação 
    #entre duas palavras iguais
    def SetAllLowerCase (self):
        """
            Set all characters of the text to lower case.
        """
        for idx, row in self.dataset.iterrows():
            self.dataset.at[self.dataset.index[idx], self.tokenized] = [w.lower() for w in row[self.tokenized]]
        return self.dataset

    #Remove sufixos das palavras para encontrar uma forma raíz da palavra e 
    #eliminar variações de uma mesma palavra
    def Stemming (self):
        """
            Stemming removes the suffix of words to avoid duplicated words of
            same meaning.
        """
        stemmer = nltk.stem.RSLPStemmer()
        for idx, row in self.dataset.iterrows():
            self.dataset.at[self.dataset.index[idx], self.tokenized] = [stemmer.stem(word) for word in row[self.tokenized]]
        return self.dataset
    
    def getDataset(self):
        """
            Get the current dataset.
        """
        return self.dataset
    
    def FullProcess(self):
        """
            Executes the following class functions on a text:
                TokenizeText
                RemoveStopwords
                RemoveNonAlphabetical
                SetAllLowerCase
                Stemming
        """
        processor = TextPreprocessing(self.dataset, self.column, self.language)
        processed_dataset = processor.TokenizeText()
        processed_dataset = processor.RemoveStopwords()
        processed_dataset = processor.RemoveNonAlphabetical()
        processed_dataset = processor.SetAllLowerCase
        processed_dataset = processor.Stemming()
        return processed_dataset
        



processador = TextPreprocessing(df_prop, 'TEXTO', 'portuguese')
processed_data = processador.FullProcess()



