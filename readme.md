# Quora Question Pairs - Kaggle

Source: https://www.kaggle.com/c/quora-question-pairs

## Problem Statement

* Identify which questions asked on Quora are duplicates of questions that have already been asked.
* This could be useful to instantly provide answers to questions that have already been answered.
* task is to predict whether a pair of questions given are duplicates or not.

### Real world/Business Objectives and Constraints¶
* The cost of a mis-classification can be very high.
* You would want a probability of a pair of questions to be duplicates so that you can choose any threshold of choice.
* No strict latency concerns.
* Interpretability is partially important.

## Data Description
- Data will be in a file Train.csv
- Train.csv contains 5 columns : qid1, qid2, question1, question2, is_duplicate
- Size of Train.csv - 60MB
- Number of rows in Train.csv = 404,290

#### columns
* id: Looks like a simple rowID
* qid{1, 2}: The unique ID of each question in the pair
* question{1, 2}: The actual textual contents of the questions.
* is_duplicate: The label that we are trying to predict - whether the two questions are duplicates of each other.

## Metric
Since higher misclassification leads to Customer dissatisfaction (lower precision rate) and consumer disengagement (lower recall rate)

We have to be highly sure about our predictions So probailities of examples belonging to a class seems to be sound metric for our problem. LogLoss looks at the probabilities themselves and not just the order of the predictions like AUC

* log-loss : LogarithmicLoss (metric we want to optimize for)
* Binary Confusion Matrix (more human intrepretable)


## Feature Extraction

### Basic Feature Extraction 

* freq_qid1 = Frequency of qid1's
* freq_qid2 = Frequency of qid2's
* q1len = Length of q1
* q2len = Length of q2
* q1_n_words = Number of words in Question 1
* q2_n_words = Number of words in Question 2
* word_Common = (Number of common unique words in Question 1 and Question 2)
* word_Total =(Total num of words in Question 1 + Total num of words in Question 2)
* word_share = (word_common)/(word_Total)
* freq_q1+freq_q2 = sum total of frequency of qid1 and qid2
* freq_q1-freq_q2 = absolute difference of frequency of qid1 and qid2

### Advanced NLP features (based descriptive stats)

* cwc_min : Ratio of common_word_count to min lenghth of word count of Q1 and Q2
* cwc_min = common_word_count / (min(len(q1_words), len(q2_words))



* cwc_max : Ratio of common_word_count to max lenghth of word count of Q1 and Q2
* cwc_max = common_word_count / (max(len(q1_words), len(q2_words))



* csc_min : Ratio of common_stop_count to min lenghth of stop count of Q1 and Q2
* csc_min = common_stop_count / (min(len(q1_stops), len(q2_stops))



* csc_max : Ratio of common_stop_count to max lenghth of stop count of Q1 and Q2
* csc_max = common_stop_count / (max(len(q1_stops), len(q2_stops))



* ctc_min : Ratio of common_token_count to min lenghth of token count of Q1 and Q2
* ctc_min = common_token_count / (min(len(q1_tokens), len(q2_tokens))



* ctc_max : Ratio of common_token_count to max lenghth of token count of Q1 and Q2
* ctc_max = common_token_count / (max(len(q1_tokens), len(q2_tokens))



* last_word_eq : Check if First word of both questions is equal or not
* last_word_eq = int(q1_tokens[-1] == q2_tokens[-1])



* first_word_eq : Check if First word of both questions is equal or not
* first_word_eq = int(q1_tokens[0] == q2_tokens[0])



* abs_len_diff : Abs. length difference
* abs_len_diff = abs(len(q1_tokens) - len(q2_tokens))



* mean_len : Average Token Length of both Questions
* mean_len = (len(q1_tokens) + len(q2_tokens))/2

### Advanced fuzzy feature (based on various similartiy scores)

* Package -  https://github.com/seatgeek/fuzzywuzzy#usage
* features description - http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/

**Feature names**:

- _fuzz_ratio_ 
<br>

- _fuzz_partial_ratio_
<br>

- _token_sort_ratio_
<br>

- _token_set_ratio_
<br>


## Best Model

Since our objective is to optimize for log loss (dependent on predicted probabilities). We need need be sure of probabalities coming out of our model. So ML algorithms are warpped in ```CalibratedClassifierCV``` which calibrates model probabilities.  

### Simple TFIDF 

```python
clf = Pipeline([
    ('fill_na', Pipeline([
        ('fill_null',Text_FillNa(fill_value=''))])),
    ('features',FeatureUnion([
        ('numerics_stats', Pipeline([('summerize',Text_SimStats())])),
        ('NLP_vectorize',Pipeline([
            ('extract',ColumnExtractor(['question1','question2'])),
            ('preprocess', Quora_TextPreprocess()),
            ('nlp_features',DF_NLP_extract())])),
        ('Fuzzy_vectorize',Pipeline([
            ('extract',ColumnExtractor(['question1','question2'])),
            ('preprocess', Quora_TextPreprocess()),
            ('fuzzy_features',DF_fuzzySimExtract())])),
        ('TFIDF_vectorizer',Pipeline([
            ('extract',ColumnExtractor(['question1','question2'])),
            ('combine',Converter()),
            ('TFIDF_wv_features',TfidfVectorizer(ngram_range=(1,2)))]))
        ]))])

```

### TFIDF weighted Word2Vec features 

```python
clf = Pipeline([
    ('fill_na', Pipeline([
        ('fill_null',Text_FillNa(fill_value=''))])),
    ('features',FeatureUnion([
        ('numerics_stats', Pipeline([('summerize',Text_SimStats())])),
        ('NLP_vectorize',Pipeline([
            ('extract',ColumnExtractor(['question1','question2'])),
            ('preprocess', Quora_TextPreprocess()),
            ('nlp_features',DF_NLP_extract())])),
        ('Fuzzy_vectorize',Pipeline([
            ('extract',ColumnExtractor(['question1','question2'])),
            ('preprocess', Quora_TextPreprocess()),
            ('fuzzy_features',DF_fuzzySimExtract())])),
        ('TFIDF_vectorizer',Pipeline([
            ('extract',ColumnExtractor(['question1','question2'])),
            ('TFIDF_wv_features',TFIDF_WV_transformer(TFIDF_dict = tfidf_dict))]))
        ]))])

```

### HyperParameters search space
``` py
param_grid = {
    'base_estimator__n_estimators': [100, 150, 250],
    'base_estimator__learning_rate': [0.10, 0.5],
    'base_estimator__min_child_weight': [1, 5, 10],
    'base_estimator__gamma': [0.5, 1, 1.5],
    'base_estimator__subsample': [0.6, 0.8],
    'base_estimator__colsample_bytree': [0.6, 0.8],
}
```

### XGB Validation results

![](BestModel.png)

### Calibration on Best Estimator

