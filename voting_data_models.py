#!/usr/bin/env python3

import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
import sklearn.model_selection as ms
from sklearn import metrics
from sklearn.impute import SimpleImputer

voting_data = pd.read_csv('voting_data.csv',
                          header=0,
                          index_col=False)
print(voting_data.head())
voting_data.info()
print()

voting_data.replace('?', np.NaN, inplace=True)
voting_data.replace('y', 1, inplace=True)
voting_data.replace('n', 0, inplace=True)

labels = voting_data['Class Name']
features = voting_data.drop('Class Name', axis=1)

# Version 1: Drop rows with NaN
features_v1 = features.dropna(axis=0, how='any')
features_v1.info()

# Version 2: Replace NaN with third category
features_v2 = features.fillna(2)
print(features_v2.head())

# Version 3: Replace NaN with mode (most frequent)
imp = SimpleImputer(strategy='most_frequent')
features_v3 = imp.fit_transform(features)
