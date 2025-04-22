# -*- coding: utf-8 -*-

import csv
import urllib.request

def drawing(r):
    col = 0
    for x in r:
        if  int(x)>=0 and int(x)<64:
            print("-", end="")
        elif int(x)>=64 and int(x)<128:
            print(".", end="")
        elif int(x)>=128 and int(x)<192:
            print("*", end="")
        elif int(x)>=192 and int(x)<256:
            print("#", end="")
        col += 1
        if col == 28:
            print('\n')
            col = 0
    
        
        
count = 0
target = input('enter the k: ')

with open('digit.csv') as f:
    for r in csv.reader(f):
        count += 1
        if count == int(target):
            drawing(r)


'''gettinh html'''
file1 = urllib.request.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
file2 = urllib.request.urlopen("http://api.citybik.es/v2/networks/to-bike")
'''converting to bytes'''
my_HTML = file1.read()
'''converting to string'''
utf = my_HTML.decode('utf8')

for r in utf.split('\n'):
    l = r.split(',')
    print(l[0])