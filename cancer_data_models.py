#!/usr/bin/env python3

import pandas as pd

cancer_data = pd.read_csv('breastcancer_data.csv',
                          header=0,
                          index_col=0)
cancer_data.drop(cancer_data.filter(regex="Unnamed"),
                 axis=1, inplace=True)
print(cancer_data.head())
cancer_data.info()

labels = cancer_data['diagnosis']
labels.replace('M', 1, inplace=True)
labels.replace('B', 0, inplace=True)

features = cancer_data.drop('diagnosis', axis=1)

