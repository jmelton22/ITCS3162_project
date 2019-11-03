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


def print_cv_scores(results):
    f1 = results['test_f1']
    precision = results['test_precision']
    recall = results['test_recall']
    accuracy = results['test_accuracy']

    print(' ' * 4 + 'Cross Validation Scores')
    print(' ' * 9 + 'F1     Precision    Recall    Accuracy')
    for i, (f, p, r, a) in enumerate(zip(f1, precision, recall, accuracy)):
        print('Fold {}   {:.3f}    {:.3f}      {:.3f}     {:.3f}'.format(i, f, p, r, a))
    print()
    print('Mean F1: {:.3f}'.format(f1.mean()))
    print('Mean Precision: {:.3f}'.format(precision.mean()))
    print('Mean Recall: {:.3f}'.format(recall.mean()))
    print('Mean Accuracy: {:.3f}'.format(accuracy.mean()))


def plot_cv_scores(results):
    n = [int(x) for x in results['param_n_neighbors']]

    f1 = results['mean_test_f1']
    precision = results['mean_test_precision']
    recall = results['mean_test_recall']
    accuracy = results['mean_test_accuracy']

    fig = plt.figure(figsize=(10, 6))
    plt.plot(n, precision,
             color='cornflowerblue', label='Precision',
             linewidth=2)
    plt.plot(n, f1,
             color='orange', label='F1 Score',
             linewidth=2)
    plt.plot(n, recall,
             color='forestgreen', label='Recall',
             linewidth=2)
    plt.plot(n, accuracy,
             color='purple', label='Accuracy',
             linewidth=2)

    # plt.ylim((0.8, 1.0))
    plt.xlabel('Num Neighbors', fontweight='bold')
    plt.title('Mean Cross Validation Scores')
    plt.xticks(n)

    plt.legend(title='Scoring Method', loc='best', ncol=2, frameon=True)

    plt.show()


def plot_cv_scores_bar(results):
    n = [int(x) for x in results['param_n_neighbors']]

    f1 = results['mean_test_f1']
    precision = results['mean_test_precision']
    recall = results['mean_test_recall']
    accuracy = results['mean_test_accuracy']

    fill = [0] * len(n)

    f1_std = [fill, results['std_test_f1']]
    precision_std = [fill, results['std_test_precision']]
    recall_std = [fill, results['std_test_recall']]
    accuracy_std = [fill, results['std_test_accuracy']]

    bar_width = 0.15
    r1 = np.arange(len(f1))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]

    fig = plt.figure(figsize=(10, 6))
    plt.bar(r1, f1, yerr=f1_std,
            color='orange', width=bar_width,
            edgecolor='white', label='F1 Score')
    plt.bar(r2, precision, yerr=precision_std,
            color='cornflowerblue', width=bar_width,
            edgecolor='white', label='Precision')
    plt.bar(r3, recall, yerr=recall_std,
            color='forestgreen', width=bar_width,
            edgecolor='white', label='Recall')
    plt.bar(r4, accuracy, yerr=accuracy_std,
            color='purple', width=bar_width,
            edgecolor='white', label='Accuracy')

    plt.ylim((0.5, 1.0))
    plt.xlabel('Num Neighbors', fontweight='bold')
    plt.title('Mean Cross Validation Scores')
    plt.xticks([r + bar_width for r in range(len(n))], n)

    plt.legend(title='Scoring Method', loc='lower left', bbox_to_anchor=(1, 0.5), ncol=1, frameon=True)

    plt.show()
