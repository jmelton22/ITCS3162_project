#!/usr/bin/env python3

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import sklearn.model_selection as ms
from sklearn import metrics


def print_metrics(labels, preds):
    """
        Prints confusion matrix and metrics scores for a binary classification
    """
    scores = metrics.precision_recall_fscore_support(labels, preds)
    conf = metrics.confusion_matrix(labels, preds)
    print(' ' * 11 + 'Confusion Matrix')
    print(' ' * 17 + 'Predict Malignant    Predict Benign')
    print('Actual Malignant         {}                 {}'.format(conf[0, 0], conf[0, 1]))
    print('Actual Benign            {}                 {}'.format(conf[1, 0], conf[1, 1]))
    print()
    print('Accuracy: {0:.3f}'.format(metrics.accuracy_score(labels, preds)))
    print()
    print(' ' * 11 + 'Malignant    Benign')
    print('Num cases    {}           {}'.format(scores[3][0], scores[3][1]))
    print('Precision    {0:.2f}       {0:.2f}'.format(scores[0][0], scores[0][1]))
    print('Recall       {0:.2f}       {0:.2f}'.format(scores[1][0], scores[1][1]))
    print('F1 Score     {0:.2f}       {0:.2f}'.format(scores[2][0], scores[2][1]))


cancer_data = pd.read_csv('breastcancer_data.csv',
                          header=0,
                          index_col=0)
cancer_data.drop(cancer_data.filter(regex="Unnamed"),
                 axis=1, inplace=True)
# print(cancer_data.head())
# cancer_data.info()

labels = cancer_data['diagnosis']
labels.replace('M', 1, inplace=True)
labels.replace('B', 0, inplace=True)

features = cancer_data.drop('diagnosis', axis=1)

X_train, X_test, y_train, y_test = ms.train_test_split(features, labels,
                                                       test_size=0.2,
                                                       random_state=123)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred_gnb = gnb.predict(X_test)

print('Naive Bayes Classifier')
print_metrics(y_test, y_pred_gnb)
print('-'*50)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

print('K-Nearest Neighbors Classifier')
print_metrics(y_test, y_pred_knn)
