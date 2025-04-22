# -*- coding: utf-8 -*-

import csv
import math

septal_length = []
septal_width = []
petal_length = []
petal_width = []

septal_length_dis = 0
septal_width_dis = 0
petal_length_dis = 0
petal_width_dis = 0

plant = set()

with open("Iris.csv") as f:
    '''body = f.read().split("\n")'''
    for r in csv.reader(f):
        try:
            '''getting summation of any elemnet'''
            
            septal_length.append(float(r[0])) 
            septal_width.append(float(r[1]))
            petal_length.append(float(r[2]))
            petal_width.append(float(r[3]))
            
            plant.add(r[4])
            
        except:
            print('no attribute')

'''computing mean value'''
septal_length_ave = sum(septal_length)/len(septal_length)
septal_width_ave = sum(septal_width)/len(septal_width)
petal_length_ave = sum(petal_length)/len(petal_length)
petal_width_ave = sum(petal_width)/len(petal_width)

'''computing squared distance'''
for i in range(0,len(septal_length)):
    septal_length_dis += (septal_length[i] - septal_length_ave)**2
    septal_width_dis += (septal_width[i] - septal_width_ave)**2
    petal_length_dis += (petal_length[i] - petal_length_ave)**2
    petal_width_dis += (petal_width[i] - petal_width_ave)**2

'''computing deviation'''    
septal_length_dev = math.sqrt(septal_length_dis)
septal_width_dev = math.sqrt(septal_width_dis)
petal_length_dev = math.sqrt(petal_length_dis)
petal_width_dev = math.sqrt(petal_width_dis)





