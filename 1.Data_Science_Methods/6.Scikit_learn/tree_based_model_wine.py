# -*- coding: utf-8 -*-

from IPython.display import Image
import pydot
from sklearn.metrics import accuracy_score

import numpy as np
from collections import Counter, defaultdict
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, ParameterGrid, KFold
from sklearn.metrics import precision_recall_fscore_support, classification_report


dataset = load_wine()
X = dataset['data']
y = dataset['target']
feature_names = dataset['feature_names']

print(X[np.isnan(X)])
print(Counter(y))

clf = DecisionTreeClassifier()
clf.fit(X, y)

dot_code = export_graphviz(clf, feature_names=feature_names)
graph = pydot.graph_from_dot_data(dot_code)
Image(graph[0].create_png())

y_pred = clf.predict(X)
acc = accuracy_score(y, y_pred)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y)
clf1 = DecisionTreeClassifier()
clf1.fit(X_train, y_train)
y_pred1 = clf1.predict(X_test)

acc1 = accuracy_score(y_pred1, y_test)
precision, recall, fscore, support = precision_recall_fscore_support(y_pred1, y_test)
print(classification_report(y_test, y_pred1))

param = {'max_depth':[None, 2, 4, 8], 'splitter':['best','random']}
result = defaultdict(lambda: 0)
for config in ParameterGrid(param):
    dtc = DecisionTreeClassifier(**config)
    dtc.fit(X_train, y_train)
    y_p = dtc.predict(X_test)
    ac = accuracy_score(y_test, y_p)
    result[(config['max_depth'], config['splitter'])] += ac


kf = KFold(5)
a = kf.split(X_train)
for train, validation in kf.split(X_train):
    xt = X_train[train]
    yt = y_train[train]
    xv = X_train[validation]
    yv = y_train[validation]











