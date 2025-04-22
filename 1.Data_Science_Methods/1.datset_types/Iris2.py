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

count = 0

with open("Iris.csv") as f:
    '''body = f.read().split("\n")'''
    for r in csv.reader(f):
        try:
            '''getting summation of any elemnet'''
            
            if r[4] == 'Iris-setosa':
                septal_length.append(float(r[0])) 
                septal_width.append(float(r[1]))
                petal_length.append(float(r[2]))
                petal_width.append(float(r[3]))
                count +=1
                flag_setosa = True
                continue
                
            
            elif r[4] == 'Iris-versicolor':
                if flag_setosa:
                    flag_setosa = False
                    septal_length_ave = sum(septal_length)/count
                    septal_width_ave = sum(septal_width)/count
                    petal_length_ave = sum(petal_length)/count
                    petal_width_ave = sum(petal_width)/count
                    
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
                    
                    print(f'for setosa the septal_length_ave is {septal_length_ave} and the septal_length_dev is {septal_length_dev}')
                    print(f'for setosa the septal_width_ave is {septal_width_ave} and the septal_width_dev is {septal_width_dev}')
                    print(f'for setosa the petal_length_ave is {petal_length_ave} and the petal_length_ave is {petal_length_ave}')
                    print(f'for setosa the petal_width_ave is {petal_width_ave} and the petal_width_dev is {petal_width_dev}')
                    setosa = len(septal_length)
                    count = 0
                septal_length.append(float(r[0])) 
                septal_width.append(float(r[1]))
                petal_length.append(float(r[2]))
                petal_width.append(float(r[3]))
                count +=1
                flag_versicolor = True
                continue
                
                
            elif r[4] == 'Iris-virginica':
                
                 if flag_versicolor:
                    flag_versicolor = False
                    septal_length_ave = sum(septal_length)/count
                    septal_width_ave = sum(septal_width)/count
                    petal_length_ave = sum(petal_length)/count
                    petal_width_ave = sum(petal_width)/count
                    
                    for i in range(setosa,len(septal_length)):
                        septal_length_dis += (septal_length[i] - septal_length_ave)**2
                        septal_width_dis += (septal_width[i] - septal_width_ave)**2
                        petal_length_dis += (petal_length[i] - petal_length_ave)**2
                        petal_width_dis += (petal_width[i] - petal_width_ave)**2
                    
                    '''computing deviation'''    
                    septal_length_dev = math.sqrt(septal_length_dis)
                    septal_width_dev = math.sqrt(septal_width_dis)
                    petal_length_dev = math.sqrt(petal_length_dis)
                    petal_width_dev = math.sqrt(petal_width_dis)
                    
                    print(f'for versicolor the septal_length_ave is {septal_length_ave} and the septal_length_dev is {septal_length_dev}')
                    print(f'for versicolor the septal_width_ave is {septal_width_ave} and the septal_width_dev is {septal_width_dev}')
                    print(f'for versicolor the petal_length_ave is {petal_length_ave} and the petal_length_ave is {petal_length_ave}')
                    print(f'for versicolor the petal_width_ave is {petal_width_ave} and the petal_width_dev is {petal_width_dev}')
                    
                    versicolor = len(septal_length)
                    count = 0
                
                 septal_length.append(float(r[0])) 
                 septal_width.append(float(r[1]))
                 petal_length.append(float(r[2]))
                 petal_width.append(float(r[3]))
                 count +=1
                 flag_virginica =True
                 
            kkkk = len(septal_length)
        
            if flag_virginica and len(septal_length) == 150:
                
                flag_virginica = False
                septal_length_ave = sum(septal_length)/count
                septal_width_ave = sum(septal_width)/count
                petal_length_ave = sum(petal_length)/count
                petal_width_ave = sum(petal_width)/count
            
                for i in range(versicolor,len(septal_length)):
                    septal_length_dis += (septal_length[i] - septal_length_ave)**2
                    septal_width_dis += (septal_width[i] - septal_width_ave)**2
                    petal_length_dis += (petal_length[i] - petal_length_ave)**2
                    petal_width_dis += (petal_width[i] - petal_width_ave)**2
            
                    
                septal_length_dev = math.sqrt(septal_length_dis)
                septal_width_dev = math.sqrt(septal_width_dis)
                petal_length_dev = math.sqrt(petal_length_dis)
                petal_width_dev = math.sqrt(petal_width_dis)
                
                print(f'for virginica the septal_length_ave is {septal_length_ave} and the septal_length_dev is {septal_length_dev}')
                print(f'for virginica the septal_width_ave is {septal_width_ave} and the septal_width_dev is {septal_width_dev}')
                print(f'for virginica the petal_length_ave is {petal_length_ave} and the petal_length_ave is {petal_length_ave}')
                print(f'for virginica the petal_width_ave is {petal_width_ave} and the petal_width_dev is {petal_width_dev}')
                
                count = 0 
            
              
            
                    
                    
                    
        except:
            print('no attribute')


