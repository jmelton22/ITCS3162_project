#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

hourly_data = pd.read_csv('Bike-Sharing-Dataset/hour.csv',
                          header=0, index_col=0,
                          parse_dates=[1])
# print(hourly_data.head())
# hourly_data.info()

features = hourly_data.drop(['dteday', 'casual', 'registered', 'cnt'], axis=1)
label_set = ['casual', 'registered', 'cnt']
for lab in label_set:
    labels = hourly_data[lab]

    corrs = features.corrwith(labels, axis=0,
                              method='pearson')
    corrs.sort_values(ascending=False, inplace=True)
    # print(corrs)

    fig, ax = plt.subplots(2, 2, figsize=(16, 8))
    ind = 0
    for row in [0, 1]:
        for col in [0, 1]:
            df_col = corrs.index[ind]
            ax[row, col].scatter(features[df_col], labels, s=3)

            ax[row, col].set_xlabel(df_col, fontweight='bold')
            ax[row, col].set_ylabel(lab)

            ax[row, col].text(0.05, 0.95, 'R = {:.4f}'.format(corrs[ind]),
                              transform=ax[row, col].transAxes)
            ind += 1

    plt.show()
