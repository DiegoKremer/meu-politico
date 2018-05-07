import nltk as nltk
import pandas as pd
from nltk.corpus import stopwords


# Import dataset
df_prop = pd.read_excel('Dataset.xlsx')


class TextProcessor ():
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
    
    def tokenize_text (self): 
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
    

    def remove_stopwords (self):
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


    def remove_nonalphabetical (self):
        """
            Removes all words from the list that are not alphabetical. That 
            includes numbers, punctuations and empty spaces. 
        """
        for idx, row in self.dataset.iterrows():
            self.dataset.at[self.dataset.index[idx], self.tokenized] = [word for word in row[self.tokenized] if word.isalpha()]
        return self.dataset


    def set_all_lowercase (self):
        """
            Set all characters of the text to lower case.
        """
        for idx, row in self.dataset.iterrows():
            self.dataset.at[self.dataset.index[idx], self.tokenized] = [w.lower() for w in row[self.tokenized]]
        return self.dataset


    def stemming (self):
        """
            Stemming removes the suffix of words to avoid duplicated words of
            same meaning.
        """
        stemmer = nltk.stem.RSLPStemmer()
        for idx, row in self.dataset.iterrows():
            self.dataset.at[self.dataset.index[idx], self.tokenized] = [stemmer.stem(word) for word in row[self.tokenized]]
        return self.dataset
    
    def get_dataset(self):
        """
            Get the current dataset.
        """
        return self.dataset
    
    def full_process(self):
        """
            Executes the following class functions on a text:
                tokenize_text
                remove_stopwords
                remove_nonalphabetical
                set_all_lowercase
                stemming
        """
        processor = TextProcessor(self.dataset, self.column, self.language)
        processed_dataset = processor.tokenize_text()
        processed_dataset = processor.remove_stopwords()
        processed_dataset = processor.remove_nonalphabetical()
        processed_dataset = processor.set_all_lowercase
        processed_dataset = processor.stemming()
        return processed_dataset
        



processador = TextProcessor(df_prop, 'TEXTO', 'portuguese')
processed_data = processador.full_process()



