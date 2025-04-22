# -*- coding: utf-8 -*-

import csv 
import pandas as pd
import pyfpgrowth
from mlxtend.frequent_patterns import fpgrowth, association_rules, apriori
import timeit


invoices = []
flag = False
with open('online_retail.csv') as f:
    for invoice in csv.reader(f):
       if flag == False:
           flag = True
           continue
       if 'C' not in invoice[0]:
           invoices.append(invoice)


invoiceID = invoices[0][0]
items = []
itemName = []
invoiceItem = {}
invoiceItemList = []

for each in invoices:
    if each[0] == invoiceID:
        items.append(each[2])
        if each[2] not in itemName:
            itemName.append(each[2])
    else:
        invoiceItem[invoiceID] = items
        invoiceItemList.append(items)
        items = []
        invoiceID = each[0]
        items.append(each[2])

matrix = []
for i in  range(len(invoiceItemList)):
    temp = []
    for j in range(len(itemName)):
        if itemName[j] in invoiceItemList[i]:
            temp.append(1)
        else:
            temp.append(0)
    matrix.append(temp)

df = pd.DataFrame(data = matrix, columns = itemName)
# t1 = timeit.timeit(lambda: apriori(df, min_support=0.5))
# t2 = timeit.timeit(lambda: fpgrowth(df, min_support=0.5))
frequentPattern = fpgrowth(df, min_support=0.5)
rules = association_rules(frequentPattern, metric="confidence", min_threshold=0.85)
print(len(frequentPattern))
print(frequentPattern.to_string)

