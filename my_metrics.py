#!/usr/bin/env python3

from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np


def print_metrics(labels, preds):
    """
        Prints confusion matrix and metrics scores for a binary classification
    """
    scores = metrics.precision_recall_fscore_support(labels, preds)
    conf = metrics.confusion_matrix(labels, preds)
    print(' ' * 4 + 'Confusion Matrix')
    print(' ' * 17 + 'Predict Positive    Predict Negative')
    print('Actual Positive         {}                 {}'.format(conf[0, 0], conf[0, 1]))
    print('Actual Negative         {}                 {}'.format(conf[1, 0], conf[1, 1]))
    print()
    print('Accuracy: {0:.3f}'.format(metrics.accuracy_score(labels, preds)))
    print()
    print(' ' * 4 + 'Classification Report')
    print(' ' * 11 + 'Positive    Negative')
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


def plot_cv_scores(f1, precision, recall):
    folds = range(1, len(f1)+1)
    fig = plt.figure(figsize=(10, 6))

    plt.plot(folds, precision,
             color='cornflowerblue', label='Precision',
             alpha=0.8, linewidth=2)
    plt.plot(folds, f1,
             color='orange', label='F1 Score',
             alpha=0.8, linewidth=2)
    plt.plot(folds, recall,
             color='forestgreen', label='Recall',
             alpha=0.8, linewidth=2)

    plt.axhline(precision.mean(), color='cornflowerblue', label='Mean Precision', linestyle='dashed')
    plt.axhline(f1.mean(), color='orange', label='Mean F1 Score', linestyle='dashed')
    plt.axhline(recall.mean(), color='forestgreen', label='Mean Recall', linestyle='dashed')

    plt.xlabel('CV Fold')
    plt.title('Cross Validation Scores')
    plt.xticks(folds)

    plt.legend(title='Scoring Method', loc='best', ncol=2, frameon=True)

    plt.show()


def plot_cv_scores_bar(f1, precision, recall):
    folds = range(1, len(f1)+1)
    bar_width = 0.25

    r1 = np.arange(len(f1))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    plt.bar(r1, f1,
            color='orange', width=bar_width,
            edgecolor='white', label='F1 Score')
    plt.bar(r2, precision,
            color='cornflowerblue', width=bar_width,
            edgecolor='white', label='Precision')
    plt.bar(r3, recall,
            color='forestgreen', width=bar_width,
            edgecolor='white', label='Recall')

    plt.axhline(f1.mean(), color='orange', linestyle='dashed', label='Mean F1 Score')
    plt.axhline(precision.mean(), color='cornflowerblue', linestyle='dashed', label='Mean Precision')
    plt.axhline(recall.mean(), color='forestgreen', linestyle='dashed', label='Mean Recall')

    plt.ylim(top=1.3)
    plt.xlabel('Fold', fontweight='bold')
    plt.xticks([r + bar_width for r in range(len(f1))], folds)
    plt.legend(title='Scoring Method', loc='best', ncol=2, frameon=True)

    plt.show()


def plot_grid_cv(score_dict):
    n_neighbors = score_dict['param_n_neighbors']
    mean_scores = score_dict['mean_test_score']
    std_scores = score_dict['std_test_score']

    fig = plt.figure(figsize=(10, 6))
    plt.bar(n_neighbors, mean_scores,
            yerr=std_scores, capsize=2)

    plt.title('Grid Search CV')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Mean F1 Score')

    plt.ylim(bottom=0.6)
    plt.xticks([int(i) for i in n_neighbors])

    plt.show()
