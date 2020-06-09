import re
import string
import pandas as pd
import numpy as np
import distance
from bs4 import BeautifulSoup
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import nltk
from nltk.corpus import wordnet as wn, stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
import unicodedata
from tqdm.notebook import tqdm
from pdb import set_trace
from fuzzywuzzy import  fuzz





class Text_FillNa(BaseEstimator, TransformerMixin):

    def __init__(self,fill_value = ''):
        self.fill_value = fill_value

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.fillna(self.fill_value)

class Text_Summerizer(BaseEstimator, TransformerMixin):
    
    def fit(self,X,y=None):
        return self

    def transform(self,X):
        dic = {
            'count_letters' : lambda x: len(str(x)),
            'count_word' : lambda x: len(str(x).split()),
            'count_unique_word' : lambda x: len(set(str(x).split())),
            'count_sent' :  lambda x: len(nltk.sent_tokenize(str(x))),
            'count_punctuations': lambda x: len([c for c in str(x) if c in string.punctuation]),
            'mean_word_len' : lambda x: np.mean([len(w) for w in str(x).split()]),
            #'count_stopwords' : lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords])
        }        
        
        mat = np.zeros((len(X),len(dic)))
        
        
        
        for ind,col in enumerate(dic):
            mat[:,ind] =X.apply(dic[col]).values

        return pd.DataFrame(mat, columns = list(dic.keys()))


class Text_SimStats(BaseEstimator, TransformerMixin):
    
    def fit(self,X,y=None):
        return self

    def transform(self, X):
        cols = X.columns
        X['freq_qid1'] = X.groupby('qid1')['qid1'].transform('count') 
        X['freq_qid2'] = X.groupby('qid2')['qid2'].transform('count')
        X['q1len'] = X['question1'].str.len() 
        X['q2len'] = X['question2'].str.len()
        X['q1_n_words'] = X['question1'].apply(lambda row: len(row.split(" ")))
        X['q2_n_words'] = X['question2'].apply(lambda row: len(row.split(" ")))

        def normalized_word_Common(row):
            w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
            w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
            return 1.0 * len(w1 & w2)
        X['word_Common'] = X.apply(normalized_word_Common, axis=1)

        def normalized_word_Total(row):
            w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
            w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
            return 1.0 * (len(w1) + len(w2))
        X['word_Total'] = X.apply(normalized_word_Total, axis=1)

        def normalized_word_share(row):
            w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
            w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
            return 1.0 * len(w1 & w2)/(len(w1) + len(w2))
        X['word_share'] = X.apply(normalized_word_share, axis=1)

        X['freq_q1+q2'] = X['freq_qid1']+X['freq_qid2']
        X['freq_q1-q2'] = abs(X['freq_qid1']-X['freq_qid2'])
        X_return = X.drop(cols, axis=1)
        return X_return.values


class Quora_TextPreprocess(BaseEstimator, TransformerMixin):

    def __init__(self,stopwords = None):
        self.stopwords = stopwords
    
    def preprocess(self,x):
        x = str(x).lower()
        x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
                            .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                            .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                            .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                            .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
                            .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                            .replace("€", " euro ").replace("'ll", " will")
        x = re.sub(r"([0-9]+)000000", r"\1m", x)
        x = re.sub(r"([0-9]+)000", r"\1k", x)
        
        
        porter = PorterStemmer()
        pattern = re.compile('\W')
        
        if type(x) == type(''):
            x = re.sub(pattern, ' ', x)
        
        
        if type(x) == type(''):
            x = porter.stem(x)
            example1 = BeautifulSoup(x)
            x = example1.get_text()
                
        
        return x

    def fit(self,X,y=None):
        return self

    def transform(self, X):
        for col in X.columns:
            X[col] = X[col].apply(self.preprocess)
        return X

class DF_NLP_extract(BaseEstimator, TransformerMixin):


    def get_token_features(self,q1, q2):
        token_features = [0.0]*10
    
        # Converting the Sentence into Tokens: 
        q1_tokens = q1.split()
        q2_tokens = q2.split()

        if len(q1_tokens) == 0 or len(q2_tokens) == 0:
            return token_features

        # Get the non-stopwords in Questions
        q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
        q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
        
        #Get the stopwords in Questions
        q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
        q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
        
        # Get the common non-stopwords from Question pair
        common_word_count = len(q1_words.intersection(q2_words))
        
        # Get the common stopwords from Question pair
        common_stop_count = len(q1_stops.intersection(q2_stops))
        
        # Get the common Tokens from Question pair
        common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
        
        SAFE_DIV = 0.0001
        token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
        token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
        token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
        token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
        token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        
        # Last word of both question is same or not
        token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
        
        # First word of both question is same or not
        token_features[7] = int(q1_tokens[0] == q2_tokens[0])
        
        token_features[8] = abs(len(q1_tokens) - len(q2_tokens))
        
        #Average Token Length of both Questions
        token_features[9] = (len(q1_tokens) + len(q2_tokens))/2
        return token_features


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Merging Features with dataset
        cols = X.columns

        token_features = X.apply(lambda x: self.get_token_features(x[cols[0]], x[cols[1]]), axis=1)
        
        X["cwc_min"]       = list(map(lambda x: x[0], token_features))
        X["cwc_max"]       = list(map(lambda x: x[1], token_features))
        X["csc_min"]       = list(map(lambda x: x[2], token_features))
        X["csc_max"]       = list(map(lambda x: x[3], token_features))
        X["ctc_min"]       = list(map(lambda x: x[4], token_features))
        X["ctc_max"]       = list(map(lambda x: x[5], token_features))
        X["last_word_eq"]  = list(map(lambda x: x[6], token_features))
        X["first_word_eq"] = list(map(lambda x: x[7], token_features))
        X["abs_len_diff"]  = list(map(lambda x: x[8], token_features))
        X["mean_len"]      = list(map(lambda x: x[9], token_features))
        X_return = X.drop(cols, axis=1)
        return X_return.values


class DF_fuzzySimExtract(BaseEstimator, TransformerMixin):


    def fit(self, X,y=None):
        return self

    def get_longest_substr_ratio(self,a, b):
        strs = list(distance.lcsubstrings(a, b))
        if len(strs) == 0:
            return 0
        else:
            return len(strs[0]) / (min(len(a), len(b)) + 1)

    def transform(self, X):
        cols = X.columns

        X["token_set_ratio"]       = X.apply(lambda x: fuzz.token_set_ratio(x[cols[0]], x[cols[1]]), axis=1)
        # The token sort approach involves tokenizing the string in question, sorting the tokens alphabetically, and 
        # then joining them back into a string We then compare the transformed strings with a simple ratio().
        X["token_sort_ratio"]      = X.apply(lambda x: fuzz.token_sort_ratio(x[cols[0]], x[cols[1]]), axis=1)
        X["fuzz_ratio"]            = X.apply(lambda x: fuzz.QRatio(x[cols[0]], x[cols[1]]), axis=1)
        X["fuzz_partial_ratio"]    = X.apply(lambda x: fuzz.partial_ratio(x[cols[0]], x[cols[1]]), axis=1)
        X["longest_substr_ratio"]  = X.apply(lambda x: self.get_longest_substr_ratio(x[cols[0]], x[cols[1]]), axis=1)
        X_return = X.drop(cols, axis=1)
        return X_return.values



class TFIDF_WV_transformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, TFIDF_dict = None,nlp = spacy.load('en_core_web_sm')):
        self.TFIDF_dict = TFIDF_dict
        self.nlp = nlp

    def fit(self,X,y=None):
        return self

    def vectorize(self,X):
        vecs = []        
        for sent in X:
            doc = self.nlp(sent)
            mean_vec = np.zeros([len(doc), len(doc[0].vector)])

            

            for word in doc:
                # word2vec
                vec = word.vector
                # fetch df score
                try:
                    idf = self.TFIDF_dict[str(word)]
                except:
                    idf = 0
                # compute final vec
                mean_vec += vec * idf
            mean_vec = mean_vec.mean(axis=0)
            vecs.append(mean_vec)
        return np.stack(vecs, axis=0)

    def transform(self,df):
        allArrays = []
        for col in df.columns:
            myArray = self.vectorize(df[col])
            allArrays.append(myArray)
        return np.concatenate(allArrays, axis=1)
