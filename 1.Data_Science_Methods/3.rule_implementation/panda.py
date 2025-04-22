# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# df = pd.DataFrame(np.array([[1, 3, 5],[2, 4, 6]]).T, index=['a','b','c'], columns=["price","quantity"])

# mean = df.mean()
# std = df.std()

# norm_z = (df - mean)/std

X = np.array([[5.1, 3.5, 1.4, 0.2],
              [4.3, 3, 1, 0.1],
              [5, np.nan, 1, 0.4],
              [5.1, 3.4, 2, 0.2],
              [7, 3.2, 1, 0.2],
              [6.9, 3.1, 3, 1.5],
              [6.7, 3.1, 1, np.nan],
              [6, 2.9, 2, 1.5],
              [6.1, 3, 2, 1.4],
              [6.5, 3, 3, 2.2],
              [7.7, 3.8, 3, 2.2],
              [7.4, 2.8, 1, 1.9],
              [6.8, 3.2, 1, 2.3]])
columns = ['height', 'width', 'intensity', 'weight']
labels = np.array([0]*5 + [1]*4 + [2]*4)

# creating dataframe
df = pd.DataFrame(X, columns=columns)

# adding new column
df["label"] = labels
 # or
X_new = np.hstack((X, labels.reshape(-1, 1)))
df_new = pd.DataFrame(X_new, columns=columns + ["label"])

df["area"] = df["height"] * df["width"]

df.fillna(method='ffill', inplace=True)

#compute ave area of intensity > 1
mask = df['intensity'] > 1
area1 = df.loc[mask, 'area']
ave_area = np.mean(area1)

# pro
num = (df['label'] == 2) & (df['height'] < 7).sum()
denom = (df['height'] < 7).sum()
prob = num/denom

