#!/usr/bin/env python3

import pandas as pd
import sklearn.model_selection as ms
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA

from Part_2.utils import *

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

pd.set_option('display.expand_frame_repr', False)


daily_data = pd.read_csv('Bike-Sharing-Dataset/day.csv',
                         header=0, index_col=0,
                         parse_dates=[1])
print(daily_data.head())
daily_data.info()

features = daily_data.drop(['dteday', 'casual', 'registered', 'cnt'], axis=1)
print(features.describe())

# pca = PCA(n_components=0.99,
#           svd_solver='full')
# features = pca.fit_transform(features)
# print(['{:.5f}'.format(x) for x in pca.explained_variance_ratio_])
# fig1 = plt.figure(figsize=(16, 8))
# plt.bar(x=range(len(pca.explained_variance_ratio_)), height=pca.explained_variance_ratio_)
# plt.show()

label_set = ['casual', 'registered', 'cnt']
for lab in label_set:
    print(lab)
    print()
    labels = daily_data[lab]

    X_train, X_test, y_train, y_test = ms.train_test_split(features, labels,
                                                           test_size=0.2,
                                                           random_state=123)

    # Linear Regression model
    print('Linear Regression model')
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    lin_preds = lin_reg.predict(X_test)
    print_metrics(y_test, lin_preds)

    hist_resids(y_test, lin_preds, lab, 'lin_reg')
    resid_qq(y_test, lin_preds, lab, 'lin_reg')
    resid_plot(y_test, lin_preds, lab, 'lin_reg')
    print('-' * 25)

    # KNN Regression model
    print('KNN Regression model')
    knn_reg = ms.GridSearchCV(KNeighborsRegressor(),
                              param_grid={'n_neighbors': range(1, 20, 2)},
                              cv=5, scoring='neg_mean_squared_error',
                              refit=True)
    knn_reg.fit(X_train, y_train)
    print('Best k:', knn_reg.best_params_)

    knn_preds = knn_reg.predict(X_test)
    print_metrics(y_test, knn_preds)

    hist_resids(y_test, lin_preds, lab, 'knn_reg')
    resid_qq(y_test, lin_preds, lab, 'knn_reg')
    resid_plot(y_test, lin_preds, lab, 'knn_reg')
    print('#' * 50)
