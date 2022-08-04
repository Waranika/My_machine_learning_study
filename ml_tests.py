# -*- coding: utf-8 -*-
"""ML-tests

Import these libraries : 
* Panda (file importation)
* scikit-learn (for the algorithm application)
* numpy (for result accuracy calculation)


"""

from __future__ import print_function

import pandas as pd

from sklearn.utils import Bunch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

import numpy as np

"""Get dataset

"""

"""
While I use the avalaible datasets from the sklearn library, using pandas it's possible to import read files and put them to Bunch format

Example:
twenty_train = pd.read_csv("twenty_train.csv", header = 0, sep=";")
twenty_test = pd.read_csv("twenty_test.csv", header = 0, sep =";")
#Turn the files into Bunchs with respect to each columns
twenty_train = Bunch(data =  twenty_train["data"].fillna(' ').to_list(), targets = twenty_train["targets"].fillna(' ').to_list())
twenty_test = Bunch(data =  twenty_test["data"].fillna(' ').to_list(), targets = twenty_test["targets"].fillna(' ').to_list())
"""
from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)

#We can print the categories that are going to get sorted
print(twenty_train.target_names)

"""*Naive Bayers* test"""

text_clf2 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
text_clf2 = text_clf2.fit(twenty_train.data, twenty_train.target)

predicted = text_clf2.predict(twenty_test.data)
print (predicted)
np.mean(predicted == twenty_test.target)

pd.DataFrame(predicted)

"""*Support Vector Machine* test"""

text_clf_svm = Pipeline ([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf-svm', SGDClassifier(loss='hinge',penalty = 'l2', alpha = 1e-3, random_state = 42)),])
_=text_clf_svm.fit(twenty_train.data, twenty_train.target)

predicted_svm = text_clf_svm.predict(twenty_test.data)
print(predicted_svm)
np.mean(predicted_svm == twenty_test.target)

pd.DataFrame(predicted)

"""

*Logistic regression* test
"""

text_clf_svm2 = Pipeline ([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf-svm', LogisticRegression(penalty= 'l2', random_state = 42, max_iter=1000)),])
_=text_clf_svm2.fit(twenty_train.data, twenty_train.target)

import numpy as np
#libel_test.data = [libel_test.data]
predicted_svm2 = text_clf_svm2.predict(twenty_test.data)
print(predicted_svm2)
predicted_svm2 = text_clf_svm2.predict_proba(twenty_test.data)
print(predicted_svm2)
print(np.max(predicted_svm2))
np.mean(predicted_svm2 == twenty_test.target)
