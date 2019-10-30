#!/usr/bin/env python3

import pandas as pd

voting_data = pd.read_csv('voting_data.csv',
                          header=0,
                          index_col=False)
print(voting_data.head())
voting_data.info()
