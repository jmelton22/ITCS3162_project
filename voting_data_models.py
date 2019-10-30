#!/usr/bin/env python3

import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
import sklearn.model_selection as ms
from sklearn import metrics
from sklearn.impute import SimpleImputer


def print_metrics(labels, preds):
    """
        Prints confusion matrix and metrics scores for a binary classification
    """
    scores = metrics.precision_recall_fscore_support(labels, preds)
    conf = metrics.confusion_matrix(labels, preds)
    print(' ' * 4 + 'Confusion Matrix')
    print(' ' * 19 + 'Predict Republican    Predict Democrat')
    print('Actual Republican         {}                   {}'.format(conf[0, 0], conf[0, 1]))
    print('Actual Democrat            {}                  {}'.format(conf[1, 0], conf[1, 1]))
    print()
    print('Accuracy: {0:.3f}'.format(metrics.accuracy_score(labels, preds)))
    print()
    print(' ' * 4 + 'Classification Report')
    print(' ' * 11 + 'Republican    Democrat')
    print('Num cases    {}             {}'.format(scores[3][0], scores[3][1]))
    print('Precision    {0:.2f}         {0:.2f}'.format(scores[0][0], scores[0][1]))
    print('Recall       {0:.2f}         {0:.2f}'.format(scores[1][0], scores[1][1]))
    print('F1 Score     {0:.2f}         {0:.2f}'.format(scores[2][0], scores[2][1]))


def print_cv_scores(f1, precision, recall):
    print(' ' * 4 + 'Cross Validation Scores')
    print(' ' * 9 + 'F1     Precision    Recall')
    for i, (f, p, r) in enumerate(zip(f1, precision, recall)):
        print('Fold {}   {:.3f}    {:.3f}      {:.3f}'.format(i, f, p, r))
    print()
    print('Mean F1: {:.3f}'.format(f1.mean()))
    print('Mean Precision: {:.3f}'.format(precision.mean()))
    print('Mean Recall: {:.3f}'.format(recall.mean()))


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

    nb_cv_f1 = ms.cross_val_score(tree, features, labels, cv=5, scoring='f1')
    nb_cv_precision = ms.cross_val_score(tree, features, labels, cv=5, scoring='precision')
    nb_cv_recall = ms.cross_val_score(tree, features, labels, cv=5, scoring='recall')

    print_cv_scores(nb_cv_f1, nb_cv_precision, nb_cv_recall)
    print('-' * 50)
    print('-' * 50)
