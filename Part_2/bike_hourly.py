#!/usr/bin/env python3

import math
import pandas as pd
import numpy as np
import scipy.stats as ss

import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.model_selection as ms
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
from sklearn.decomposition import PCA


def hist_resids(y_test, y_score):
    # Compute vector of residuals
    resids = np.subtract(y_test, y_score)
    # Make residuals plot
    sns.distplot(resids)
    plt.title('Histogram of residuals')
    plt.xlabel('Residual value')
    plt.ylabel('Count')
    plt.show()


def resid_qq(y_test, y_score):
    # Compute vector of residuals
    resids = np.subtract(y_test, y_score)
    # Make residuals plot
    ss.probplot(resids, plot=plt)
    plt.title('Residuals vs. Predicted values')
    plt.xlabel('Predicted value')
    plt.ylabel('Residual')
    plt.show()


def resid_plot(y_test, y_score):
    # Compute vector of residuals
    resids = np.subtract(y_test, y_score)
    # Make residuals plot
    sns.regplot(y_score, resids, fit_reg=False)
    plt.title('Residuals vs. Predicted values')
    plt.xlabel('Predicted value')
    plt.ylabel('Residual')
    plt.show()


def print_metrics(y_true, y_predicted):
    print('Mean Square Error      =', str(metrics.mean_squared_error(y_true, y_predicted)))
    print('Root Mean Square Error =', str(math.sqrt(metrics.mean_squared_error(y_true, y_predicted))))
    print('Mean Absolute Error    =', str(metrics.mean_absolute_error(y_true, y_predicted)))
    print('Median Absolute Error  =', str(metrics.median_absolute_error(y_true, y_predicted)))
    print('R^2                    =', str(metrics.r2_score(y_true, y_predicted)))


hourly_data = pd.read_csv('Bike-Sharing-Dataset/hour.csv',
                          header=0, index_col=0,
                          parse_dates=[1])
# print(hourly_data.head())
# hourly_data.info()

features = hourly_data.drop(['dteday', 'casual', 'registered', 'cnt'], axis=1)

# pca = PCA(n_components=0.95,
#           svd_solver='full')
# features = pca.fit_transform(features)
# print(pca.components_)
# print(['{:.5f}'.format(x) for x in pca.explained_variance_])
# print(['{:.5f}'.format(x) for x in pca.explained_variance_ratio_])
# fig1 = plt.figure(figsize=(16, 8))
# plt.bar(x=range(len(pca.explained_variance_ratio_)), height=pca.explained_variance_ratio_)
# plt.show()

for lab in ['casual', 'registered', 'cnt']:
    print(lab)
    labels = hourly_data[lab]
    print('Mean: {:.3f}'.format(labels.mean()))
    print('Std: {:.3f}'.format(labels.std()))

    X_train, X_test, y_train, y_test = ms.train_test_split(features, labels,
                                                           test_size=0.2,
                                                           random_state=123)

    # corrs = features.corrwith(labels, axis=0,
    #                           method='pearson')
    # corrs.sort_values(ascending=False, inplace=True)
    # # print(corrs)
    #
    # fig, ax = plt.subplots(2, 2, figsize=(16, 8))
    # ind = 0
    # for row in [0, 1]:
    #     for col in [0, 1]:
    #         df_col = corrs.index[ind]
    #         ax[row, col].scatter(features[df_col], labels, s=3)
    #
    #         ax[row, col].set_xlabel(df_col, fontweight='bold')
    #         ax[row, col].set_ylabel(lab)
    #
    #         ax[row, col].text(0.05, 0.95, 'R = {:.4f}'.format(corrs[ind]),
    #                           transform=ax[row, col].transAxes)
    #         ind += 1
    #
    # plt.show()

    # Linear Regression model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    lin_preds = lin_reg.predict(X_test)

    print_metrics(y_test, lin_preds)

    hist_resids(y_test, lin_preds)
    resid_qq(y_test, lin_preds)
    resid_plot(y_test, lin_preds)
    print('-' * 50)
