#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import sklearn.model_selection as ms
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


def print_metrics(labels, preds):
    """
        Prints confusion matrix and metrics scores for a binary classification
    """
    scores = metrics.precision_recall_fscore_support(labels, preds)
    conf = metrics.confusion_matrix(labels, preds)
    print(' ' * 4 + 'Confusion Matrix')
    print(' ' * 17 + 'Predict Malignant    Predict Benign')
    print('Actual Malignant         {}                 {}'.format(conf[0, 0], conf[0, 1]))
    print('Actual Benign            {}                 {}'.format(conf[1, 0], conf[1, 1]))
    print()
    print('Accuracy: {0:.3f}'.format(metrics.accuracy_score(labels, preds)))
    print()
    print(' ' * 4 + 'Classification Report')
    print(' ' * 11 + 'Malignant    Benign')
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


def plot_cv_scores(score_dict):
    print(score_dict)

    n_neighbors = score_dict['param_n_neighbors']
    mean_scores = score_dict['mean_test_score']
    std_scores = score_dict['std_test_score']
    colors = ['orange', 'cornflowerblue', 'forestgreen', 'red', 'purple']

    fig = plt.figure(figsize=(10, 6))
    plt.bar(n_neighbors, mean_scores,
            yerr=std_scores, capsize=2)

    plt.title('Cross-Validation Scores for KNN Model')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Mean F1 Score')
    plt.ylim(bottom=0.6)
    plt.xticks([int(i) for i in n_neighbors])

    plt.show()


cancer_data = pd.read_csv('breastcancer_data.csv',
                          header=0,
                          index_col=0)
cancer_data.drop(cancer_data.filter(regex='Unnamed'),
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
print()

gnb_cv_f1 = ms.cross_val_score(gnb, features, labels, cv=5, scoring='f1')
gnb_cv_precision = ms.cross_val_score(gnb, features, labels, cv=5, scoring='precision')
gnb_cv_recall = ms.cross_val_score(gnb, features, labels, cv=5, scoring='recall')

print_cv_scores(gnb_cv_f1, gnb_cv_precision, gnb_cv_recall)
print('-' * 50)

knn = KNeighborsClassifier()

knn = ms.GridSearchCV(estimator=knn, param_grid={'n_neighbors': [1, 3, 5, 7, 9]},
                      cv=3,
                      scoring='f1',
                      refit=True)
knn.fit(X_train, y_train)

plot_cv_scores(knn.cv_results_)

y_pred_knn = knn.predict(X_test)

print('K-Nearest Neighbors Classifier')
print_metrics(y_test, y_pred_knn)
print()

knn_cv_f1 = ms.cross_val_score(knn, features, labels, cv=5, scoring='f1')
knn_cv_precision = ms.cross_val_score(knn, features, labels, cv=5, scoring='precision')
knn_cv_recall = ms.cross_val_score(knn, features, labels, cv=5, scoring='recall')

print_cv_scores(knn_cv_f1, knn_cv_precision, knn_cv_recall)
