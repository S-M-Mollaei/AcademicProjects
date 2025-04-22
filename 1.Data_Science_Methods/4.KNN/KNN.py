# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math


  
        



df = pd.read_csv("iris.csv", header = None)
rand_size = len(df)//5

values = df.values[:,:-1].astype(float)
names = df.values[:,-1]

mask = np.array([True]*120 + [False]*rand_size)
np.random.shuffle(mask)

X_train = values[mask]
X_test = values[~mask]

y_train = names[mask]
y_test = names[~mask]

