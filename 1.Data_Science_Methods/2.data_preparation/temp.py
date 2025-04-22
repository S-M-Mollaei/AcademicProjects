# -*- coding: utf-8 -*-

from random import gauss
import matplotlib.pyplot as plt
import csv
import math

'''func for ranking temperature'''
def rank_temp(city, n, l):
    target = []
    for y in l:
        if y[3] == city:
            target.append(float(y[1]))
    
    target.sort()
    print('Coldest:------------------')
    for x in range(n):
        print(target[x])
    
    print('**********************************************')
    
    print('Warmest:+++++++++++++++++')
    for z in range(n):
        print(target[len(target)-1-z])
    
    ploting(target, city)
    normalized_target = normalizing(target)
    ploting(normalized_target, city)    

''' plot by hist'''
def ploting(temperature, city):
    mean, sigma = mean_dev(temperature)
    l = [gauss(mean, sigma) for _ in temperature]
    plt.hist(l)
    plt.title(city + ' with mean and sigma: ' + str(mean) + ' ' + str(sigma) )
    plt.show()

''' mean and variance'''
def mean_dev(vector):
    mean = sum(vector)/len(vector)
    summation = 0
    for x in vector:
        summation += (x - mean)**2
    sigma = math.sqrt(summation/len(vector))    
    
    return mean, sigma



''' normalizinf '''
def normalizing(temp_ave):
    nt = []
    for t in temp_ave:
        nt.append((t - min(temp_ave))/(max(temp_ave) - min(temp_ave)))
    return nt    
    

''' -----------------main----------------------'''
flag = True
average_temp = []


with open("temp.csv") as f:
    for r in csv.reader(f):
        
        if flag:
            print(r)
            flag = False
            continue
        
        average_temp.append(r)  

i = 0 
while i < len(average_temp):
    
    if average_temp[i][1] == '':
        
        temp = i + 1
        while temp < len(average_temp):
            if average_temp[temp][1] == '':
                temp += 1
            else:
                break
        
        if i == 0 and temp != len(average_temp):
            
            if temp - i == 1:
                average_temp[i][1] = str(float(average_temp[temp][1])/2)
                i = temp + 1
                continue
            else:
                average_temp[i][1] = str(float(average_temp[temp][1])/2)
                j = i + 1
                while j < temp:
                    average_temp[j][1] = str((float(average_temp[j-1][1]) + float(average_temp[temp][1]))/2)
                    j += 1 
                i = temp + 1
                continue
        

        if i != 0 and temp != len(average_temp):
            if temp - i == 1:
                average_temp[i][1] = str((float(average_temp[temp][1]) + float(average_temp[temp-2][1]))/2)
                i = temp + 1
                continue
            else:
                average_temp[i][1] = str(float(average_temp[temp][1])/2)
                j = i + 1
                while j < temp:
                    average_temp[j][1] = str((float(average_temp[j-1][1]) + float(average_temp[temp][1]))/2)
                    j += 1 
                i = temp + 1
                continue
        
        if temp == len(average_temp):
            if i == len(average_temp) -1:
                average_temp[i][1] = str(float(average_temp[i-1][1])/2)
                i = temp + 1
                continue
            else:
                average_temp[temp-1][1] = str(float(average_temp[i-1][1])/2)
                j = temp -2
                while j > i - 1:
                    average_temp[j][1] = str((float(average_temp[j+1][1]) + float(average_temp[i-1][1]))/2)
                    j -= 1
                i = temp + 1
                continue
    i += 1  

rank_temp('Bangkok', 3, average_temp)
rank_temp('Rome', 3, average_temp)









































# i = 0
# while i < len(average_temp):
    
#     if i == 0 and average_temp[i] == '' and average_temp[i+1] != '':
#         average_temp[i] = str(float(average_temp[i+1])/2)
#         i += 2
#         continue
    
#     if i == 0 and average_temp[i] == '' and average_temp[i+1] == '':
#         average_temp[i] = str(float(average_temp[i+2])/2)
#         average_temp[i+1] = str((float(average_temp[i]) + float(average_temp[i+2]))/2)
#         i += 3
#         continue
    
#     if i != 0 and average_temp[i] == '' and average_temp[i+1] != '':
#         average_temp[i] = str((float(average_temp[i-1]) + float(average_temp[i+1]))/2)
#         i += 2
#         continue
    
#     if i != 0 and average_temp[i] == '' and average_temp[i+1] == '':
#         average_temp[i] = str(float(average_temp[i+2])/2)
#         average_temp[i+1] = str((float(average_temp[i]) + float(average_temp[i+2]))/2)
#         i += 3
#         continue
    
#     if i == len(average_temp)-2 and average_temp[i] == '' and average_temp[i+1] == '':
#         average_temp[i+1] = str(float(average_temp[i-1])/2)
#         average_temp[i] = str((float(average_temp[i-1]) + float(average_temp[i+1]))/2)
#         i += 2
#         continue
    
#     if i == len(average_temp)-1 and average_temp[i] == '':
#         average_temp[i] = str(float(average_temp[i-1])/2)
#         continue
#     i += 1
    
    
    
    
    
    
    
    
    