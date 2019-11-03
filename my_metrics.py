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


def plot_cv_scores(n, f1, precision, recall):
    fig = plt.figure(figsize=(10, 6))

    plt.plot(n, precision,
             color='cornflowerblue', label='Precision',
             alpha=0.75, linewidth=2)
    plt.plot(n, f1,
             color='orange', label='F1 Score',
             alpha=0.75, linewidth=2)
    plt.plot(n, recall,
             color='forestgreen', label='Recall',
             alpha=0.75, linewidth=2)

    # plt.axhline(precision.mean(), color='cornflowerblue', label='Mean Precision', linestyle='dashed')
    # plt.axhline(f1.mean(), color='orange', label='Mean F1 Score', linestyle='dashed')
    # plt.axhline(recall.mean(), color='forestgreen', label='Mean Recall', linestyle='dashed')

    plt.xlabel('Num Neighbors')
    plt.title('Cross Validation Scores')
    plt.xticks(n)

    plt.legend(title='Scoring Method', loc='best', ncol=2, frameon=True)

    plt.show()


def plot_cv_scores_bar(n, f1, precision, recall):
    bar_width = 0.25

    r1 = np.arange(len(f1))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)

    plt.bar(r1, f1,
            color='orange', width=bar_width, alpha=0.75,
            edgecolor='white', label='F1 Score')
    plt.bar(r2, precision,
            color='cornflowerblue', width=bar_width, alpha=0.75,
            edgecolor='white', label='Precision')
    plt.bar(r3, recall,
            color='forestgreen', width=bar_width, alpha=0.75,
            edgecolor='white', label='Recall')

    # plt.axhline(f1.mean(), color='orange', linestyle='dashed', label='Mean F1 Score')
    # plt.axhline(precision.mean(), color='cornflowerblue', linestyle='dashed', label='Mean Precision')
    # plt.axhline(recall.mean(), color='forestgreen', linestyle='dashed', label='Mean Recall')

    plt.ylim(bottom=0.7)
    plt.xlabel('Num Neighbors', fontweight='bold')
    plt.title('Cross Validation Scores')
    plt.xticks([r + bar_width for r in range(len(f1))], n)

    ax.legend(title='Scoring Method', loc='lower left', bbox_to_anchor=(1, 0.5), ncol=1, frameon=True)

    plt.show()
