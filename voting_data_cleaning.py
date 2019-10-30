#!/usr/bin/env python3

import pandas as pd
import numpy as np

voting_data = pd.read_csv('voting_data.csv',
                          header=0,
                          index_col=False)
print(voting_data.head())
voting_data.info()
print()

voting_data.replace('?', np.NaN, inplace=True)
voting_data.replace('y', 1, inplace=True)
voting_data.replace('n', 0, inplace=True)

print(voting_data.head())
voting_data.info()

voting_data.to_csv('voting_data_clean.csv',
                   header=True,
                   index=False)
