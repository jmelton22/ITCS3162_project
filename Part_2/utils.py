#!/usr/bin/env python3

import math
import numpy as np
import scipy.stats as ss
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics


def hist_resids(y_test, y_preds, label, model):
    # Compute vector of residuals
    resids = np.subtract(y_test, y_preds)
    # Make residuals plot
    sns.distplot(resids)
    plt.title('{}: {}\nHistogram of residuals'.format(label, model))
    plt.xlabel('Residual value')
    plt.ylabel('Count')
    plt.show()


def resid_qq(y_test, y_preds, label, model):
    # Compute vector of residuals
    resids = np.subtract(y_test, y_preds)
    # Make residuals plot
    ss.probplot(resids, plot=plt)
    plt.title('{}: {}\nResiduals vs. Predicted values'.format(label, model))
    plt.xlabel('Predicted value')
    plt.ylabel('Residual')
    plt.show()


def resid_plot(y_test, y_preds, label, model):
    # Compute vector of residuals
    resids = np.subtract(y_test, y_preds)
    # Make residuals plot
    sns.regplot(y_preds, resids, fit_reg=False)
    plt.title('{}: {}\nResiduals vs. Predicted values'.format(label, model))
    plt.xlabel('Predicted value')
    plt.ylabel('Residual')
    plt.show()


def print_metrics(y_true, y_predicted):
    print('Mean Square Error      =', str(metrics.mean_squared_error(y_true, y_predicted)))
    print('Root Mean Square Error =', str(math.sqrt(metrics.mean_squared_error(y_true, y_predicted))))
    print('Mean Absolute Error    =', str(metrics.mean_absolute_error(y_true, y_predicted)))
    print('Median Absolute Error  =', str(metrics.median_absolute_error(y_true, y_predicted)))
    print('R^2                    =', str(metrics.r2_score(y_true, y_predicted)))
