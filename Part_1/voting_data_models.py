#!/usr/bin/env python3

import pandas as pd
from Part_1.my_metrics import *

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
import sklearn.model_selection as ms
from sklearn.impute import SimpleImputer

from graphviz import Source

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import os
os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin'

voting_data = pd.read_csv('voting_data.csv',
                          header=0,
                          index_col=False)
# print(voting_data.head())
# voting_data.info()

voting_data.replace({'y': 1, 'n': 0, '?': np.NaN,
                     'republican': 1, 'democrat': 0}, inplace=True)

print(voting_data.groupby('Class Name').size())
print(voting_data[voting_data.isnull().any(axis=1)].groupby('Class Name').size())
print()

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

features = [features_v1, features_v2, features_v3]
labels = [labels_v1, labels_v2, labels_v3]
scoring = ['f1', 'precision', 'recall', 'accuracy']

for i, (feat, lab) in enumerate(zip(features, labels)):

    X_train, X_test, y_train, y_test = ms.train_test_split(feat, lab,
                                                           test_size=0.2,
                                                           random_state=1776)

    # Decision Tree model
    print('Version {}: Decision Tree'.format(i+1))
    tree = DecisionTreeClassifier(criterion='gini',
                                  class_weight='balanced',
                                  random_state=1916)
    tree.fit(X_train, y_train)

    graph = Source(export_graphviz(tree, out_file=None, feature_names=voting_data.columns[1:]))
    graph.render('tree_v{}.gv'.format(i+1), view=True)

    y_pred_tree = tree.predict(X_test)
    print_metrics(y_test, y_pred_tree)
    print()

    tree_cv_scores = ms.cross_validate(tree, feat, lab,
                                       cv=5, scoring=scoring)
    print_cv_scores(tree_cv_scores)
    print('-' * 25)

    # Random Forest model
    print('Version {}: Random Forest'.format(i+1))
    forest = RandomForestClassifier(criterion='gini',
                                    bootstrap=True,
                                    class_weight='balanced',
                                    random_state=1916)
    forest.fit(X_train, y_train)

    y_pred_forest = forest.predict(X_test)
    print_metrics(y_test, y_pred_forest)
    print()

    forest_cv_scores = ms.cross_validate(forest, feat, lab,
                                         cv=5, scoring=scoring)
    print_cv_scores(forest_cv_scores)
    print('-' * 25)

    # Naive Bayes model
    print('Version {}: Bernoulli Naive Bayes'.format(i+1))
    bnb = BernoulliNB()
    bnb.fit(X_train, y_train)

    y_pred_bnb = bnb.predict(X_test)
    print_metrics(y_test, y_pred_bnb)
    print()

    bnb_cv_scores = ms.cross_validate(bnb, feat, lab,
                                      cv=5, scoring=scoring)
    print_cv_scores(bnb_cv_scores)
    print('-' * 50)
    print('-' * 50)
