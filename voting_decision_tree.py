#!/usr/bin/env python3

import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
import sklearn.model_selection as ms
from sklearn import metrics
from sklearn.impute import SimpleImputer


def print_metrics(labels, preds):
    """
        Prints confusion matrix and metrics scores for a binary classification
    """
    scores = metrics.precision_recall_fscore_support(labels, preds)
    conf = metrics.confusion_matrix(labels, preds)
    print(' ' * 11 + 'Confusion Matrix')
    print(' ' * 17 + 'Predict positive    Predict negative')
    print('Actual positive         {}                 {}'.format(conf[0, 0], conf[0, 1]))
    print('Actual negative         {}                 {}'.format(conf[1, 0], conf[1, 1]))
    print()
    print('Accuracy: {0:.3f}'.format(metrics.accuracy_score(labels, preds)))
    print()
    print(' ' * 11 + 'Positive    Negative')
    print('Num cases    {}           {}'.format(scores[3][0], scores[3][1]))
    print('Precision    {0:.2f}       {0:.2f}'.format(scores[0][0], scores[0][1]))
    print('Recall       {0:.2f}       {0:.2f}'.format(scores[1][0], scores[1][1]))
    print('F1 Score     {0:.2f}       {0:.2f}'.format(scores[2][0], scores[2][1]))


voting_data = pd.read_csv('voting_data.csv',
                          header=0,
                          index_col=False,
                          dtype='category')
print(voting_data.head())
voting_data.info()

labels = voting_data['Class Name']
features = voting_data.drop('Class Name', axis=1)

features.replace('?', np.NaN, inplace=True)
features.replace('y', 1, inplace=True)
features.replace('n', 0, inplace=True)
