# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import json
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

from nltk.corpus import stopwords
from nltk import word_tokenize
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

test = pd.read_json('../input/test.json', orient = 'columns')    
train = pd.read_json('../input/train.json', orient = 'columns')

print(train['text'])
stopWords = stopwords.words('russian')
print(stopWords)

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train["text"])
X_train_counts.shape

from sklearn.feature_extraction.text import TfidfTransformer


X_train_tfidf = TfidfTransformer().fit_transform(X_train_counts)
X_train_tfidf.shape

neg = len(train[train["sentiment"] == "negative"])
pos = len(train[train["sentiment"] == "positive"])
neu = len(train[train["sentiment"] == "neutral"])
print("Negative values", neg)
print("Positive values", pos)
print("Neutral values", neu)

from sklearn.model_selection import train_test_split
y_train = train['sentiment']
X_train, X_test, y_train_new, y_test = train_test_split(train['text'], y_train, test_size = 0.2)

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import numpy as np
import collections

svm_pipeline = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf-svm', SGDClassifier(loss='modified_huber', penalty='l2',alpha=1e-3, max_iter=5, random_state=42))])

svm_pipeline = svm_pipeline.fit(X_train, y_train_new)
predicted_svm = svm_pipeline.predict(X_test)
print(np.mean(predicted_svm == y_test))
#print(predicted_svm)
#print(collections.Counter(predicted_svm))

with open('../input/test.json') as f:
    testing_data = json.load(f)

predicted_svm = svm_pipeline.predict(test['text'])
new_dataframe = pd.DataFrame()
new_dataframe['id'] = [i['id'] for i in testing_data]
print(predicted_svm)
new_dataframe['sentiment'] = predicted_svm
new_dataframe.head()

new_dataframe.to_csv('svm_sklearn.csv', index = False)
