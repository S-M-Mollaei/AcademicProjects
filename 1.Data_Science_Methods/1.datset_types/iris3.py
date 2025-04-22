# -*- coding: utf-8 -*-

import csv
import math

plant = set()
count = 1
#i = 0
flag = True

sl = []
sw = []
pl = []
pw = []

def average(sl, sw, pl, pw):
    ave_list = []
    ave_list.append(sum(sl)/len(sl))
    ave_list.append(sum(sw)/len(sl))
    ave_list.append(sum(pl)/len(sl))
    ave_list.append(sum(pw)/len(sl))
    
    return ave_list

def deviation(sl, sw, pl, pw, ave_list):
    dev_list = []
    
    y = 0
    for x in sl:
        y += (x - ave_list[0])**2
    dev_list.append(math.sqrt(y/len(sl)))
    y = 0
    
    for x1 in sw:
        y += (x1 - ave_list[1])**2
    dev_list.append(math.sqrt(y/len(sl)))
    y = 0
    
    for x2 in pl:
        y += (x2 - ave_list[2])**2
    dev_list.append(math.sqrt(y/len(sl)))
    y = 0
    
    for x3 in pw:
        y += (x3 - ave_list[3])**2
    dev_list.append(math.sqrt(y/len(sl)))
    
    return dev_list

with open("Iris.csv") as f:
    for r in csv.reader(f):
       try:
           plant.add(r[4])
           if count == len(plant):
               #sl[i], sw[i], pl[i], pw[i] = [float(r[0]), float(r[1]), float(r[2]), float(r[3])]
               sl.append(float(r[0]))
               sw.append(float(r[1]))
               pl.append(float(r[2]))
               pw.append(float(r[3]))
               
               if flag:
                   target = r[4]
                   flag = False
               #i += 1
           else:
               #ave_list = average(sl, sw, pl, pw, i-1)
               ave_list = average(sl, sw, pl, pw)
               dev_list = deviation(sl, sw, pl, pw, ave_list)
               
               print(f'{target}: sl_ave is {ave_list[0]}, sl_dev is {dev_list[0]}')
               print(f'{target}: sw_ave is {ave_list[1]}, sw_dev is {dev_list[1]}')
               print(f'{target}: pl_ave is {ave_list[2]}, pl_dev is {dev_list[2]}')
               print(f'{target}: pw_ave is {ave_list[3]}, pw_dev is {dev_list[3]}')
               print('\n*************************************************************\n')
               
               count += 1
               flag = True
               #i = 0
               sl = []
               sw = []
               pl = []
               pw = []
               
               #sl[i], sw[i], pl[i], pw[i] = [r[0], r[1], r[2], r[3]]
               sl.append(float(r[0]))
               sw.append(float(r[1]))
               pl.append(float(r[2]))
               pw.append(float(r[3]))
               if flag:
                   target = r[4]
                   flag = False
               #i += 1
                       
       except:
            pass
    ave_list = average(sl, sw, pl, pw)
    dev_list = deviation(sl, sw, pl, pw, ave_list)
              
    print(f'{target}: sl_ave is {ave_list[0]}, sl_dev is {dev_list[0]}')
    print(f'{target}: sw_ave is {ave_list[1]}, sw_dev is {dev_list[1]}')
    print(f'{target}: pl_ave is {ave_list[2]}, pl_dev is {dev_list[2]}')
    print(f'{target}: pw_ave is {ave_list[3]}, pw_dev is {dev_list[3]}') 
           
               