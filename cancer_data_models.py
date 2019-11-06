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
labels.replace({'M': 1, 'B': 0}, inplace=True)

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

scoring = ['f1', 'precision', 'recall', 'accuracy']

gnb_cv_scores = ms.cross_validate(gnb, features, labels,
                                  cv=5, scoring=scoring)
print_cv_scores(gnb_cv_scores)
print('-' * 50)

knn = ms.GridSearchCV(estimator=KNeighborsClassifier(),
                      param_grid={'n_neighbors': range(1, 10, 2)},
                      cv=5, scoring=scoring, refit='accuracy',
                      return_train_score=False)
knn.fit(X_train, y_train)

# for k, v in knn.cv_results_.items():
#     print('{}: {}'.format(k, ' '.join(str(x) for x in v)))
# print()

y_pred_knn = knn.predict(X_test)

print('K-Nearest Neighbors Classifier')
print_metrics(y_test, y_pred_knn)
print()

knn_cv_scores = ms.cross_validate(knn, features, labels,
                                  cv=5, scoring=scoring)
print_cv_scores(knn_cv_scores)

plot_cv_scores(knn.cv_results_)
plot_cv_scores_bar(knn.cv_results_)
