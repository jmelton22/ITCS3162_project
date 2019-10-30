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
    print(' ' * 17 + 'Predict Republican    Predict Democrat')
    print('Actual Republican         {}                 {}'.format(conf[0, 0], conf[0, 1]))
    print('Actual Democrat            {}                 {}'.format(conf[1, 0], conf[1, 1]))
    print()
    print('Accuracy: {0:.3f}'.format(metrics.accuracy_score(labels, preds)))
    print()
    print(' ' * 4 + 'Classification Report')
    print(' ' * 11 + 'Republican    Democrat')
    print('Num cases    {}           {}'.format(scores[3][0], scores[3][1]))
    print('Precision    {0:.2f}       {0:.2f}'.format(scores[0][0], scores[0][1]))
    print('Recall       {0:.2f}       {0:.2f}'.format(scores[1][0], scores[1][1]))
    print('F1 Score     {0:.2f}       {0:.2f}'.format(scores[2][0], scores[2][1]))


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

X1_train, X1_test, y1_train, y1_test = ms.train_test_split(features_v1, labels_v1,
                                                           test_size=0.2,
                                                           random_state=123)

# Version 2: Replace NaN with third category
voting_data_v2 = voting_data.fillna(2)
labels_v2 = voting_data_v2['Class Name']
features_v2 = voting_data_v2.drop('Class Name', axis=1)

X2_train, X2_test, y2_train, y2_test = ms.train_test_split(features_v2, labels_v2,
                                                           test_size=0.2,
                                                           random_state=123)

# Version 3: Replace NaN with mode (most frequent)
labels_v3 = voting_data['Class Name']
features_v3 = voting_data.drop('Class Name', axis=1)

imp = SimpleImputer(strategy='most_frequent')
features_v3 = imp.fit_transform(features_v3)

X3_train, X3_test, y3_train, y3_test = ms.train_test_split(features_v3, labels_v3,
                                                           test_size=0.2,
                                                           random_state=123)

# Version 1: Decision Tree
print('Version 1: Decision Tree')
tree_v1 = DecisionTreeClassifier()
tree_v1.fit(X1_train, y1_train)

y1_pred = tree_v1.predict(X1_test)
print_metrics(y1_test, y1_pred)
print()

tree1_cv_f1 = ms.cross_val_score(tree_v1, features_v1, labels_v1, cv=5, scoring='f1')
tree1_cv_precision = ms.cross_val_score(tree_v1, features_v1, labels_v1, cv=5, scoring='precision')
tree1_cv_recall = ms.cross_val_score(tree_v1, features_v1, labels_v1, cv=5, scoring='recall')

print_cv_scores(tree1_cv_f1, tree1_cv_precision, tree1_cv_recall)
print('-' * 25)

# Version 1: Naive Bayes
print('Version 1: Naive Bayes')
bnb_v1 = BernoulliNB()
bnb_v1.fit(X1_train, y1_train)

y2_pred = bnb_v1.predict(X1_test)
print_metrics(y1_test, y1_pred)
print()

nb1_cv_f1 = ms.cross_val_score(tree_v1, features_v1, labels_v1, cv=5, scoring='f1')
nb1_cv_precision = ms.cross_val_score(tree_v1, features_v1, labels_v1, cv=5, scoring='precision')
nb1_cv_recall = ms.cross_val_score(tree_v1, features_v1, labels_v1, cv=5, scoring='recall')

print_cv_scores(nb1_cv_f1, nb1_cv_precision, nb1_cv_recall)
print('-' * 50)
