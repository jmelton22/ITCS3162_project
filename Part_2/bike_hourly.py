#!/usr/bin/env python3

import pandas as pd
import sklearn.model_selection as ms
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics

hourly_data = pd.read_csv('Bike-Sharing-Dataset/hour.csv',
                          header=0, index_col=0,
                          parse_dates=[1])
# print(hourly_data.head())
# hourly_data.info()

features = hourly_data.drop(['dteday', 'casual', 'registered', 'cnt'], axis=1)

for lab in ['casual', 'registered', 'cnt']:
    labels = hourly_data[lab]

    X_train, X_test, y_train, y_test = ms.train_test_split(features, labels,
                                                           test_size=0.2,
                                                           random_state=123)

    corrs = features.corrwith(labels, axis=0,
                              method='pearson')
    corrs.sort_values(ascending=False, inplace=True)
    # print(corrs)

    # Linear Regression model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    lin_preds = lin_reg.predict(X_test)
    print(metrics.mean_squared_error(y_test, lin_preds))
