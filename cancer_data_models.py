#!/usr/bin/env python3

import pandas as pd
from my_metrics import *

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import sklearn.model_selection as ms

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

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

knn = ms.GridSearchCV(estimator=KNeighborsClassifier(),
                      param_grid={'n_neighbors': [1, 3, 5, 7, 9]},
                      cv=3, scoring='f1',
                      refit=True)
knn.fit(X_train, y_train)

plot_grid_cv(knn.cv_results_)

y_pred_knn = knn.predict(X_test)

print('K-Nearest Neighbors Classifier')
print_metrics(y_test, y_pred_knn)
print()

knn_cv_f1 = ms.cross_val_score(knn, features, labels, cv=5, scoring='f1')
knn_cv_precision = ms.cross_val_score(knn, features, labels, cv=5, scoring='precision')
knn_cv_recall = ms.cross_val_score(knn, features, labels, cv=5, scoring='recall')

plot_cv_scores(knn_cv_f1, knn_cv_precision, knn_cv_recall)
print_cv_scores(knn_cv_f1, knn_cv_precision, knn_cv_recall)
plot_cv_scores_bar(knn_cv_f1, knn_cv_precision, knn_cv_recall)
