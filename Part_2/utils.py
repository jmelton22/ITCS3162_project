#!/usr/bin/env python3

import math
import numpy as np
import scipy.stats as ss
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics


def hist_resids(y_true, y_preds, label, model):
    # Compute vector of residuals
    resids = np.subtract(y_true, y_preds)
    # Make residuals histogram
    sns.distplot(resids)
    plt.title('{}: {}\nHistogram of residuals'.format(label, model))
    plt.xlabel('Residual value')
    plt.ylabel('Count')
    plt.show()


def resid_qq(y_true, y_preds, label, model):
    # Compute vector of residuals
    resids = np.subtract(y_true, y_preds)
    # Make residuals quantile-quantile plot
    ss.probplot(resids, plot=plt)
    plt.title('{}: {}\nResiduals vs. Predicted values'.format(label, model))
    plt.xlabel('Predicted value')
    plt.ylabel('Residual')
    plt.show()


def resid_plot(y_true, y_preds, label, model):
    # Compute vector of residuals
    resids = np.subtract(y_true, y_preds)
    # Make residuals scatter plot
    sns.regplot(y_preds, resids, fit_reg=False)
    plt.title('{}: {}\nResiduals vs. Predicted values'.format(label, model))
    plt.xlabel('Predicted value')
    plt.ylabel('Residual')
    plt.show()


def print_metrics(y_true, y_preds):
    print('Mean Square Error      = {:.3f}'.format(metrics.mean_squared_error(y_true, y_preds)))
    print('Root Mean Square Error = {:.3f}'.format(math.sqrt(metrics.mean_squared_error(y_true, y_preds))))
    print('Mean Absolute Error    = {:.3f}'.format(metrics.mean_absolute_error(y_true, y_preds)))
    print('Median Absolute Error  = {:.3f}'.format(metrics.median_absolute_error(y_true, y_preds)))
    print('R^2                    = {:.3f}'.format(metrics.r2_score(y_true, y_preds)))
