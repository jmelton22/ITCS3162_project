#!/usr/bin/env python3

import pandas as pd
import numpy as np
from my_metrics import *

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
import sklearn.model_selection as ms
from sklearn.impute import SimpleImputer


voting_data = pd.read_csv('voting_data.csv',
                          header=0,
                          index_col=False)
# print(voting_data.head())
# voting_data.info()

voting_data.replace('?', np.NaN, inplace=True)
voting_data.replace('y', 1, inplace=True)
voting_data.replace('n', 0, inplace=True)

voting_data.replace('republican', 1, inplace=True)
voting_data.replace('democrat', 0, inplace=True)

# Version 1: Drop rows with NaN
voting_data_v1 = voting_data.dropna(axis=0, how='any')
labels_v1 = voting_data_v1['Class Name']
features_v1 = voting_data_v1.drop('Class Name', axis=1)

# Version 2: Replace NaN with third category
voting_data_v2 = voting_data.fillna(2)
labels_v2 = voting_data_v2['Class Name']
features_v2 = voting_data_v2.drop('Class Name', axis=1)

# Version 3: Replace NaN with mode (most frequent)
labels_v3 = voting_data['Class Name']
features_v3 = voting_data.drop('Class Name', axis=1)

imp = SimpleImputer(strategy='most_frequent')
features_v3 = imp.fit_transform(features_v3)

for i, (features, labels) in enumerate([(features_v1, labels_v1), (features_v2, labels_v2), (features_v3, labels_v3)]):

    X_train, X_test, y_train, y_test = ms.train_test_split(features, labels,
                                                           test_size=0.2,
                                                           random_state=123)

    # Decision Tree model
    print('Version {}: Decision Tree'.format(i+1))
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)

    y_pred_tree = tree.predict(X_test)
    print_metrics(y_test, y_pred_tree)
    print()

    tree_cv_f1 = ms.cross_val_score(tree, features, labels, cv=5, scoring='f1')
    tree_cv_precision = ms.cross_val_score(tree, features, labels, cv=5, scoring='precision')
    tree_cv_recall = ms.cross_val_score(tree, features, labels, cv=5, scoring='recall')

    print_cv_scores(tree_cv_f1, tree_cv_precision, tree_cv_recall)
    print('-' * 25)

    # Naive Bayes model
    print('Version {}: Naive Bayes'.format(i+1))
    bnb = BernoulliNB()
    bnb.fit(X_train, y_train)

    y_pred_nb = bnb.predict(X_test)
    print_metrics(y_test, y_pred_nb)
    print()

    bnb_cv_f1 = ms.cross_val_score(tree, features, labels, cv=5, scoring='f1')
    bnb_cv_precision = ms.cross_val_score(tree, features, labels, cv=5, scoring='precision')
    bnb_cv_recall = ms.cross_val_score(tree, features, labels, cv=5, scoring='recall')

    print_cv_scores(bnb_cv_f1, bnb_cv_precision, bnb_cv_recall)
    print('-' * 50)
    print('-' * 50)
