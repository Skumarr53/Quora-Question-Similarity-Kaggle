import re
import string
from string import punctuation
import functools
from copy import deepcopy
import pandas as pd
from collections import Counter
import numpy as np
import distance
from bs4 import BeautifulSoup
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from nltk.tokenize import word_tokenize
from spacy.lang.en.stop_words import STOP_WORDS
import nltk
from nltk.corpus import wordnet as wn, stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer,SnowballStemmer

from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis

import unicodedata
from tqdm import tqdm
from pdb import set_trace
from fuzzywuzzy import  fuzz

#tqdm_notebook().pandas()



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
        return X_return


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

    def transform(self, data):
        # Merging Features with dataset

        X = pd.DataFrame()
        token_features = data.apply(lambda x: self.get_token_features(x['question1'], x['question2']), axis=1)
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
        
        return X


class DF_fuzzySimExtract(BaseEstimator, TransformerMixin):


    def fit(self, X,y=None):
        return self

    def get_longest_substr_ratio(self,a, b):
        strs = list(distance.lcsubstrings(a, b))
        if len(strs) == 0:
            return 0
        else:
            return len(strs[0]) / (min(len(a), len(b)) + 1)

    def transform(self, data):
        X = pd.DataFrame()
        

        X["token_set_ratio"]       = data.apply(lambda x: fuzz.token_set_ratio(x['question1'], x['question2']), axis=1)
        # The token sort approach involves tokenizing the string in question, sorting the tokens alphabetically, and 
        # then joining them back into a string We then compare the transformed strings with a simple ratio().
        X["token_sort_ratio"]      = data.apply(lambda x: fuzz.token_sort_ratio(x['question1'], x['question2']), axis=1)
        X["fuzz_ratio"]            = data.apply(lambda x: fuzz.QRatio(x['question1'], x['question2']), axis=1)
        X["fuzz_partial_ratio"]    = data.apply(lambda x: fuzz.partial_ratio(x['question1'], x['question2']), axis=1)
        X["longest_substr_ratio"]  = data.apply(lambda x: self.get_longest_substr_ratio(x['question1'], x['question2']), axis=1)
        return X



class TFIDF_WV_transformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, TFIDF_dict = None,nlp = spacy.load('en_core_web_sm')):
        self.TFIDF_dict = TFIDF_dict
        self.nlp = nlp

    def fit(self,X,y=None):
        return self

    def vectorize(self,X):
        vecs = []        
        for sent in tqdm(X):
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

    def transform(self,X):
        return self.vectorize(X)


class Quora_TextCleaner(BaseEstimator, TransformerMixin):

    def fit(self,X,y=None):
        return self


    def clean(self, text):

        
        SPECIAL_TOKENS = {
        'quoted': 'quoted_item',
        'non-ascii': 'non_ascii_word',
        'undefined': 'something'
        }



        
        if pd.isnull(text):
            return ''

        #    stops = set(stopwords.words("english"))
        # Clean the text, with the option to stem words.
        
        # Empty question
        stops = set(stopwords.words("english"))
        
        if type(text) != str or text=='':
            return ''

        # Clean the text
        text = re.sub("\'s", " ", text) # we have cases like "Sam is" or "Sam's" (i.e. his) these two cases aren't separable, I choose to compromise are kill "'s" directly
        text = re.sub(" whats ", " what is ", text, flags=re.IGNORECASE)
        text = re.sub("\'ve", " have ", text)
        text = re.sub("can't", "can not", text)
        text = re.sub("n't", " not ", text)
        text = re.sub("i'm", "i am", text, flags=re.IGNORECASE)
        text = re.sub("\'re", " are ", text)
        text = re.sub("\'d", " would ", text)
        text = re.sub("\'ll", " will ", text)
        text = re.sub("e\.g\.", " eg ", text, flags=re.IGNORECASE)
        text = re.sub("b\.g\.", " bg ", text, flags=re.IGNORECASE)
        text = re.sub("(\d+)(kK)", " \g<1>000 ", text)
        text = re.sub("e-mail", " email ", text, flags=re.IGNORECASE)
        text = re.sub("(the[\s]+|The[\s]+)?U\.S\.A\.", " America ", text, flags=re.IGNORECASE)
        text = re.sub("(the[\s]+|The[\s]+)?United State(s)?", " America ", text, flags=re.IGNORECASE)
        text = re.sub("\(s\)", " ", text, flags=re.IGNORECASE)
        text = re.sub("[c-fC-F]\:\/", " disk ", text)
        
        # remove comma between numbers, i.e. 15,000 -> 15000
        
        text = re.sub('(?<=[0-9])\,(?=[0-9])', "", text)
        
        #     # all numbers should separate from words, this is too aggressive
            
        #     def pad_number(pattern):
        #         matched_string = pattern.group(0)
        #         return pad_str(matched_string)
        #     text = re.sub('[0-9]+', pad_number, text)
        
        # add padding to punctuations and special chars, we still need them later
        
        text = re.sub('\$', " dollar ", text)
        text = re.sub('\%', " percent ", text)
        text = re.sub('\&', " and ", text)
        
        #    def pad_pattern(pattern):
        #        matched_string = pattern.group(0)
        #       return pad_str(matched_string)
        #    text = re.sub('[\!\?\@\^\+\*\/\,\~\|\`\=\:\;\.\#\\\]', pad_pattern, text) 
            
        text = re.sub('[^\x00-\x7F]+', ' '+SPECIAL_TOKENS['non-ascii']+' ', text) # replace non-ascii word with special word
        
        # indian dollar
        
        text = re.sub("(?<=[0-9])rs ", " rs ", text, flags=re.IGNORECASE)
        text = re.sub(" rs(?=[0-9])", " rs ", text, flags=re.IGNORECASE)
        
        # clean text rules get from : https://www.kaggle.com/currie32/the-importance-of-cleaning-text
        text = re.sub(r" (the[\s]+|The[\s]+)?US(A)? ", " America ", text)
        text = re.sub(r" UK ", " England ", text, flags=re.IGNORECASE)
        text = re.sub(r" india ", " India ", text)
        text = re.sub(r" switzerland ", " Switzerland ", text)
        text = re.sub(r" china ", " China ", text)
        text = re.sub(r" chinese ", " Chinese ", text) 
        text = re.sub(r" imrovement ", " improvement ", text, flags=re.IGNORECASE)
        text = re.sub(r" intially ", " initially ", text, flags=re.IGNORECASE)
        text = re.sub(r" quora ", " Quora ", text, flags=re.IGNORECASE)
        text = re.sub(r" dms ", " direct messages ", text, flags=re.IGNORECASE)  
        text = re.sub(r" demonitization ", " demonetization ", text, flags=re.IGNORECASE) 
        text = re.sub(r" actived ", " active ", text, flags=re.IGNORECASE)
        text = re.sub(r" kms ", " kilometers ", text, flags=re.IGNORECASE)
        text = re.sub(r" cs ", " computer science ", text, flags=re.IGNORECASE) 
        text = re.sub(r" upvote", " up vote", text, flags=re.IGNORECASE)
        text = re.sub(r" iPhone ", " phone ", text, flags=re.IGNORECASE)
        text = re.sub(r" \0rs ", " rs ", text, flags=re.IGNORECASE)
        text = re.sub(r" calender ", " calendar ", text, flags=re.IGNORECASE)
        text = re.sub(r" ios ", " operating system ", text, flags=re.IGNORECASE)
        text = re.sub(r" gps ", " GPS ", text, flags=re.IGNORECASE)
        text = re.sub(r" gst ", " GST ", text, flags=re.IGNORECASE)
        text = re.sub(r" programing ", " programming ", text, flags=re.IGNORECASE)
        text = re.sub(r" bestfriend ", " best friend ", text, flags=re.IGNORECASE)
        text = re.sub(r" dna ", " DNA ", text, flags=re.IGNORECASE)
        text = re.sub(r" III ", " 3 ", text)
        text = re.sub(r" banglore ", " Banglore ", text, flags=re.IGNORECASE)
        text = re.sub(r" J K ", " JK ", text, flags=re.IGNORECASE)
        text = re.sub(r" J\.K\. ", " JK ", text, flags=re.IGNORECASE)
        
        # replace the float numbers with a random number, it will be parsed as number afterward, and also been replaced with word "number"
        
        text = re.sub('[0-9]+\.[0-9]+', " 87 ", text)
    
        # Remove punctuation from text
        text = ''.join([c for c in text if c not in punctuation]).lower()

        #text = text.split()
        #text = [w for w in text if not w in stops]
        #text = ' '.join(text)

        return text

    def transform(self, X):
        for col in X.columns:
            X[col] = X[col].progress_apply(self.clean)
        return X



class NLP_TextStats(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    # jaccard coeff
    def jaccard(self, row):
        wic = set(row['question1']).intersection(set(row['question2']))
        uw = set(row['question1']).union(row['question2'])
        if len(uw) == 0:
            uw = [1]
        return (len(wic) / len(uw))

    def total_unique_words(self, row):
        return len(set(row['question1']).union(row['question2']))

    def total_unq_words_stop(self, row, stops):
        return len([x for x in set(row['question1']).union(row['question2']) if x not in stops])

    def wc_diff(self,row):
        return abs(len(row['question1']) - len(row['question2']))

    # ratio of word count
    def wc_ratio(self,row):
        l1 = len(row['question1'])*1.0 
        l2 = len(row['question2'])
        if l2 == 0:
            return np.nan
        if l1 / l2:
            return l2 / l1
        else:
            return l1 / l2
    
    def wc_diff_unique(self,row):
        return abs(len(set(row['question1'])) - len(set(row['question2'])))

    def wc_diff_unique_stop(self, row, stops=None):
        return abs(len([x for x in set(row['question1']) if x not in stops]) - len([x for x in set(row['question2']) if x not in stops]))

    # word count for unique words
    def wc_ratio_unique_stop(self,row, stops=None):
        l1 = len([x for x in set(row['question1']) if x not in stops])*1.0 
        l2 = len([x for x in set(row['question2']) if x not in stops])
        if l2 == 0:
            return np.nan
        if l1 / l2:
            return l2 / l1
        else:
            return l1 / l2

    def wc_ratio_unique(self,row):
        l1 = len(set(row['question1'])) * 1.0
        l2 = len(set(row['question2']))
        if l2 == 0:
            return np.nan
        if l1 / l2:
            return l2 / l1
        else:
            return l1 / l2

    

    # stats at character level
    def char_diff(self,row):
        return abs(len(''.join(row['question1'])) - len(''.join(row['question2'])))

    def char_ratio(self, row):
        l1 = len(''.join(row['question1'])) 
        l2 = len(''.join(row['question2']))
        if l2 == 0:
            return np.nan
        if l1 / l2:
            return l2 / l1
        else:
            return l1 / l2

    def char_diff_unique_stop(self, row, stops=None):
        return abs(len(''.join([x for x in set(row['question1']) if x not in stops])) - len(''.join([x for x in set(row['question2']) if x not in stops])))


    def tfidf_word_match_share_stops(self, row, stops=None, weights=None):
        q1words = {}
        q2words = {}
        for word in row['question1']:
            if word not in stops:
                q1words[word] = 1
        for word in row['question2']:
            if word not in stops:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            # The computer-generated chaff includes a few questions that are nothing but stopwords
            return 0

        shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
        total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
        
        R = np.sum(shared_weights) / np.sum(total_weights)
        return R
    
    def tfidf_word_match_share(self, row, weights=None):
        q1words = {}
        q2words = {}
        for word in row['question1']:
            q1words[word] = 1
        for word in row['question2']:
            q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            # The computer-generated chaff includes a few questions that are nothing but stopwords
            return 0
        
        shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
        total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
        
        R = np.sum(shared_weights) / np.sum(total_weights)
        return R

    

    def transform(self, data_ori):

        stops = set(stopwords.words("english"))

        def get_weight(count, eps=10000, min_count=2):
            if count < min_count:
                return 0
            else:
                return 1 / (count + eps)

        data = data_ori.copy()
        data['question1'] = data['question1'].map(lambda x: str(x).lower().split())
        data['question2'] = data['question2'].map(lambda x: str(x).lower().split())

        train_qs = pd.Series(data['question1'].tolist() + data['question2'].tolist())

        words = [x for y in train_qs for x in y]
        counts = Counter(words)
        weights = {word: get_weight(count) for word, count in counts.items()}

        X = pd.DataFrame()

        
        #f = functools.partial(self.word_match_share, stops=stops)
        #X['word_match'] = data.apply(f, axis=1, raw=False) #1

        f = functools.partial(self.tfidf_word_match_share, weights=weights)
        X['tfidf_wm'] = data.apply(f, axis=1, raw=False) #2

        f = functools.partial(self.tfidf_word_match_share_stops, stops=stops, weights=weights)
        X['tfidf_wm_stops'] = data.apply(f, axis=1, raw=False) #3

        X['jaccard'] = data.apply(self.jaccard, axis=1, raw=False) #4
        X['wc_diff'] = data.apply(self.wc_diff, axis=1, raw=False) #5
        X['wc_ratio'] = data.apply(self.wc_ratio, axis=1, raw=False) #6
        X['wc_diff_unique'] = data.apply(self.wc_diff_unique, axis=1, raw=False) #7
        X['wc_ratio_unique'] = data.apply(self.wc_ratio_unique, axis=1, raw=False) #8

        f = functools.partial(self.wc_diff_unique_stop, stops=stops)    
        X['wc_diff_unq_stop'] = data.apply(f, axis=1, raw=False) #9
        f = functools.partial(self.wc_ratio_unique_stop, stops=stops)    
        X['wc_ratio_unique_stop'] = data.apply(f, axis=1, raw=False) #10

        X['char_diff'] = data.apply(self.char_diff, axis=1, raw=False) #12

        f = functools.partial(self.char_diff_unique_stop, stops=stops) 
        X['char_diff_unq_stop'] = data.apply(f, axis=1, raw=False) #13

        X['total_unique_words'] = data.apply(self.total_unique_words, axis=1, raw=False)  #15

        f = functools.partial(self.total_unq_words_stop, stops=stops)
        X['total_unq_words_stop'] = data.apply(f, axis=1, raw=False)  #16
        
        X['char_ratio'] = data.apply(self.char_ratio, axis=1, raw=False) #17
        
        return X



class NLP_DistMetrics(BaseEstimator, TransformerMixin):

    def __init__(self,gensim_model = None, verbose = False):
        self.gensim_model = gensim_model
        self.verbose = verbose

    def fit(self, X, y=None):
        return self

    def sent2vec(self,s):
        words = str(s).lower()
        words = word_tokenize(words)
        stop_words = stopwords.words('english')
        words = [w for w in words if not w in stop_words]
        words = [w for w in words if w.isalpha()]
        M = []
        for w in words:
            try:
                M.append(self.gensim_model[w])
            except:
                continue
        M = np.array(M)
        v = M.sum(axis=0)
        return v / np.sqrt((v ** 2).sum())

    def wmd(self,s1, s2, model):
        s1 = str(s1).lower().split()
        s2 = str(s2).lower().split()
        stop_words = stopwords.words('english')
        s1 = [w for w in s1 if w not in stop_words]
        s2 = [w for w in s2 if w not in stop_words]
        return model.wmdistance(s1, s2)

    def transform(self, X):
        data = pd.DataFrame()
        
        question1_vectors = np.zeros((X.shape[0], 300))
        error_count = 0


        if self.verbose: print('processing started')
        for i, q in enumerate(X.question1.values):
            question1_vectors[i, :] = self.sent2vec(q)

        question2_vectors  = np.zeros((X.shape[0], 300))
        for i, q in enumerate(X.question2.values):
            question2_vectors[i, :] = self.sent2vec(q)

        wm_model = self.gensim_model

        data['wmd'] = X.progress_apply(lambda x: self.wmd(x['question1'], x['question2'],wm_model), axis=1)
        del wm_model

        norm_wm_model = self.gensim_model
        norm_wm_model.init_sims(replace=True)
        data['norm_wmd'] = X.progress_apply(lambda x: self.wmd(x['question1'], x['question2'],norm_wm_model), axis=1)

        del self.gensim_model, norm_wm_model

        if self.verbose: print('processing cosine_distance')
        data['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                np.nan_to_num(question2_vectors))]
        if self.verbose: print('processing cityblock_distance')
        data['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                np.nan_to_num(question2_vectors))]
        if self.verbose: print('processing jaccard_distance')
        data['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                np.nan_to_num(question2_vectors))]

        if self.verbose: print('processing canberra_distance')
        data['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                np.nan_to_num(question2_vectors))]

        if self.verbose: print('processing euclidean_distance')
        data['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                np.nan_to_num(question2_vectors))]

        if self.verbose: print('processing minkowski_distance')
        data['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                np.nan_to_num(question2_vectors))]

        if self.verbose: print('processing braycurtis_distance')
        data['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                np.nan_to_num(question2_vectors))]

        if self.verbose: print('processing skew')
        data['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
        data['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
        
        if self.verbose: print('processing kurtosis')
        data['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
        data['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]
        
        return data