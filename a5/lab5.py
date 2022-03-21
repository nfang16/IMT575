#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 14:16:15 2022

@author: nickfang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.random import seed
import tensorflow as tf
seed(1234)
tf.random.set_seed(1234)

header = ["score", "text"]
train = pd.read_csv("train.tsv", sep='\t', names=header)
test = pd.read_csv("test.tsv", sep='\t', names=header)

train.shape
train.head

#train df has 268 rows and 2 columns (score and test of the essays)
#test df has 1515 rows and 2 columns (score and test of the essays)

from sklearn.feature_extraction.text import CountVectorizer

max_feature = 1000
vectorizer = CountVectorizer(max_features=max_feature)
tfidf_train = vectorizer.fit_transform(train.text.to_list())
tfidf_test = vectorizer.transform(test.text.to_list())

vectorizer.get_feature_names_out()[:20]
len(vectorizer.get_feature_names_out())

#from collections import Counter
total_list = [(lambda x:np.count_nonzero(x == 0))(x) for x in tfidf_train.toarray()]
print("Average 0 occurences in %s is %d times"\
      %(max_feature, round(sum(total_list) / len(total_list),)))
    
tfidf_train.toarray()[:10]
tfidf_test.toarray()[:10]

# A lot of 0's in the training vector

# Removing stop words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

print(stopwords.words('english'))
stop_words = set(stopwords.words('english'))

import regex as re

def preproc(target_text_list, target_stops):
    texts = []
    
    for text in target_text_list:
        text = re.sub(r"@[\w]+", '', text)
        text = re.sub(r"[^\w\s]", '', text)
        text = re.sub(r"\b\w+\d+\b", '', text)
        text = text.lower().strip()
        text = text.split()
        
        text = ' '.join([x for x in text if x not in target_stops])
        texts.append(text)
        
    return texts

score_list_noStop = train.score.to_list()
text_list_noStop = preproc(train.text.to_list(), stop_words)
data_noStop = {'score':score_list_noStop, 'text':text_list_noStop}
proced_train_text = pd.DataFrame(data_noStop)
proced_train_text.head()

#Stem
from nltk.stem import PorterStemmer

def stem(target_text_list):
    ps = PorterStemmer()
    words = []
    
    for text in target_text_list:
        text = text.split()
        
        text = ' '.join([(lambda x: ps.stem(x))(x) for x in text])
        words.append(text)
        
    return words

#test
t_proced_train_text = stem(proced_train_text.text.to_list())

print(t_proced_train_text[0])

print(proced_train_text.text.to_list()[0]) 

data_stem = {'score':proced_train_text.score.to_list(), 'text':t_proced_train_text}
stem_train_text = pd.DataFrame(data_stem)
stem_train_text.head()

#lemmatize 
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

def lemma(target_text_list):
    wordnet_lemmatizer = WordNetLemmatizer()
    words = []
    
    for text in target_text_list:
        text = text.split()
        
        text = ' '.join([(lambda x: wordnet_lemmatizer.lemmatize(x))(x) for x in text])
        words.append(text)
        
    return words

lemma_train_text = lemma(proced_train_text.text.to_list())
print(lemma_train_text[0])

#Lemmatization is the process of converting a word to its base form. 
#The difference between stemming and lemmatization is, lemmatization 
#considers the context and converts the word to its meaningful base form,
#whereas stemming just removes the last few characters, often leading to 
#incorrect meanings and spelling errors.

#Tokenize unigram
from sklearn.feature_extraction.text import CountVectorizer
#ngram_range = (1,1)

#lemma_text = lemma_train_text
#lemma_score = score_list_noStop
lemma_data = {'score':score_list_noStop, 'text':lemma_train_text}
lemma_df = pd.DataFrame(lemma_data)
lemma_df.head()

corpus_lemma = lemma_df.text.to_list()
vectorizer_lemma = CountVectorizer()
lemma_tokenize = vectorizer_lemma.fit_transform(corpus_lemma)
lemma_tokenize.toarray()

len(vectorizer_lemma.get_feature_names_out())
vectorizer_lemma.get_feature_names_out()[:20]

print("There are ", len(vectorizer_lemma.get_feature_names_out()), "unique tokens in our vocabulary")
# ngram_range of (1,1) represents only unigrams. There are 13,308 unique tokens 
# within this library

# Processing test data
test_lemma = lemma(test.text.to_list())
test_lemma = preproc(test_lemma, stop_words)

#lemma_text_test = test_lemma
#lemma_score_test = test.score.to_list()
lemma_data_test = {'score':test.score.to_list(), 'text':test_lemma}
lemma_test_df = pd.DataFrame(lemma_data_test)
lemma_test_df.head()

tfidf_train_lemma = lemma_tokenize
tfidf_test_lemma = vectorizer_lemma.transform(test.text.to_list())
print(tfidf_train_lemma.shape)
print(tfidf_test_lemma.shape)

print("The number of feature in training is", tfidf_train_lemma.shape[1], 
      "and in test is" ,tfidf_test_lemma.shape[1])

#Part 2: Superverised Learning
#bi-grams
ngram_range = (2,2)

#Tfidf
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfVectorizer = TfidfVectorizer(stop_words='english')

tfidf_xtrain = tfidfVectorizer.fit_transform(lemma_df.text.to_list())
tfidf_xtest = tfidfVectorizer.transform(lemma_test_df.text.to_list())
ytrain = lemma_df.score.to_list()
ytest = lemma_test_df.score.to_list()

from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(tfidf_xtrain, ytrain)
model1 = mean_squared_error(ytest, ridge.predict(tfidf_xtest))

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import svm

lin_clf = svm.LinearSVC()
lin_clf.fit(tfidf_xtrain, ytrain)
lin_clf_predictions = lin_clf.predict(tfidf_xtest)
print(classification_report(ytest, lin_clf.predict(tfidf_xtest)))
model2 = mean_squared_error(ytest, lin_clf.predict(tfidf_xtest))

#https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/
#Tuning hyper parameters SVM
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['linear']}
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
grid.fit(tfidf_xtrain, ytrain)
grid_predictions = grid.predict(tfidf_xtest)
print(classification_report(ytest, grid.predict(tfidf_xtest)))
model3 = mean_squared_error(ytest, grid.predict(tfidf_xtest))
print(grid.best_params_)

#Ridge: 1.218, seems to be best fit 
#Linear SVC: 1.698
#Tuned SVC, not sure what I did there... but results did not improve upon untuned

names = ["Ridge", "SVM", "SVM-tuned"]
model_mse = [model1, model2, model3]
plt.bar(names, model_mse)

#Part 3: Deep Sentence Embedder 
import tensorflow_hub as hub
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)

x_train = lemma_df.text.to_list()
x_test = lemma_test_df.text.to_list()

model_x_train = model(x_train)
model_x_test = model(x_test)

ridge.fit(model_x_train, ytrain)
mean_squared_error(ytest, ridge.predict(model_x_test))

lin_clf.fit(model_x_train, ytrain)
#lin_clf.predict(tfidf_xtest)
mean_squared_error(ytest, lin_clf.predict(model_x_test))

#Using embeddings:
    #Ridge MSE: 0.871
    #SVM MSE: 1.414

#Both improvements 


















