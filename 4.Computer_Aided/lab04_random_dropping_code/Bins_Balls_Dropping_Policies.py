
import random
import matplotlib.pyplot as plt 
import math
import pandas as pd
import dataframe_image as dfi 

random.seed(42)

def Average(lst):
    return sum(lst) / len(lst)


# Random Droping

def random_droping(balls, bins):
    
    for _ in range(balls):
        i = random.randint(0, len(bins) - 1)
        bins[i] += 1
    
    minimum, maximum, average = (min(bins), max(bins), Average(bins))
    
    return minimum, maximum, average


# Random Load Balancing

def comparison(bins_lst, index):
    
    func_index = []
    func_index.extend(index)
    target = func_index[0]
    func_index.pop(0)
    
    for i in func_index:
        if bins_lst[i] < bins_lst[target]:
            target = i
    return target


def random_load_balancing(balls, bins, d):

    index = [0] * d
    target_index = -1
    
    for _ in range(balls):
        
        for i in range(len(index)):
            index[i] = random.randint(0, len(bins) - 1)
            
        target_index =  comparison(bins, index)
        
        bins[target_index] += 1
    
    minimum, maximum, average = (min(bins), max(bins), Average(bins))
    
    return minimum, maximum, average


# Simulation

result = []
graphs = {}
numbers = [100, 500, int(1e3), int(1e4), int(1e5), int(1e6)]
max_drop = []
max_l2 = []
max_l4 = []

for num in numbers:
    # list of input parameters

    balls = num
    bins_dropping = list(0 for _ in range(num))
    bins_load_2 = list(0 for _ in range(num))
    bins_load_4 = list(0 for _ in range(num))
    d_list = [2, 4]
    
    temp = {num: {'dropping': [0, 0, 0], 'load_2': [0, 0, 0], 'load_4': [0, 0, 0]}}
    
    temp[num]['dropping'] = random_droping(balls, bins_dropping)
    temp[num]['load_2'] = random_load_balancing(balls, bins_load_2, d_list[0])
    temp[num]['load_4'] = random_load_balancing(balls, bins_load_4, d_list[1])
    
    result.append(temp)
    graphs[num] = [temp[num]['dropping'][1], temp[num]['load_2'][1], temp[num]['load_4'][1]]
    
    max_drop.append(temp[num]['dropping'][1])
    max_l2.append(temp[num]['load_2'][1])
    max_l4.append(temp[num]['load_4'][1])


print(result,'\n\n')

graphs, max_drop, max_l2, max_l4


# Plotting

print('***Plotting the results when num of bins and balls are equal***')
plt.plot(numbers, max_drop, marker='s', label='Random Dropping')
plt.plot(numbers, max_l2, marker='o', label='Load Balancing 2')
plt.plot(numbers, max_l4, marker='^', label='Load Balancing 4')
plt.xscale('log')
plt.xlabel('n')
plt.ylabel('Max Occupancy')
plt.legend()
plt.show()

# Comparison between theory and empirical


theory_max_upper_drop = []
theory_max_load2 = []
theory_max_load4 = []

for n in numbers:
    theory_max_upper_drop.append(3*math.log(n)/math.log(math.log(n)))
    theory_max_load2.append(math.log(math.log(n))/math.log(2))
    theory_max_load4.append(math.log(math.log(n))/math.log(4))


theory_max_upper_drop, theory_max_load2, theory_max_load4


result_comparison_drop = {'n': numbers, 'max_drop': max_drop, 'theory_max_upper_drop': theory_max_upper_drop}
result_comparison_drop_df = pd.DataFrame.from_dict(result_comparison_drop)

result_comparison_load2 = {'n': numbers, 'max_l2': max_l2, 'theory_max_load2': theory_max_load2}
result_comparison_load2_df = pd.DataFrame.from_dict(result_comparison_load2)

result_comparison_load4 = {'n': numbers, 'max_l4': max_l4, 'theory_max_load4': theory_max_load4}
result_comparison_load4_df = pd.DataFrame.from_dict(result_comparison_load4)


display(result_comparison_drop_df)
#dfi.export(result_comparison_drop_df, 'result_comparison_drop_df.png')

display(result_comparison_load2_df)
#dfi.export(result_comparison_load2_df, 'result_comparison_load2_df.png')

display(result_comparison_load4_df)
#dfi.export(result_comparison_load4_df, 'result_comparison_load4_df.png')


# Extension
# One possible extension is that we can double the number of balls and see whether the maximum accupancy is doubled or not

print('***Extension: Plotting the results when num of balls are twice the num of bins***')
result = []
graphs = {}
numbers = [100, 500, int(1e3), int(1e4), int(1e5), int(1e6)]
max_drop = []
max_l2 = []
max_l4 = []

for num in numbers:
    # list of input parameters

    balls = 2 * num
    bins_dropping = list(0 for _ in range(num))
    bins_load_2 = list(0 for _ in range(num))
    bins_load_4 = list(0 for _ in range(num))
    d_list = [2, 4]
    
    temp = {num: {'dropping': [0, 0, 0], 'load_2': [0, 0, 0], 'load_4': [0, 0, 0]}}
    
    temp[num]['dropping'] = random_droping(balls, bins_dropping)
    temp[num]['load_2'] = random_load_balancing(balls, bins_load_2, d_list[0])
    temp[num]['load_4'] = random_load_balancing(balls, bins_load_4, d_list[1])
    
    result.append(temp)
    graphs[num] = [temp[num]['dropping'][1], temp[num]['load_2'][1], temp[num]['load_4'][1]]
    
    max_drop.append(temp[num]['dropping'][1])
    max_l2.append(temp[num]['load_2'][1])
    max_l4.append(temp[num]['load_4'][1])

plt.plot(numbers, max_drop, marker='s', label='Random Dropping')
plt.plot(numbers, max_l2, marker='o', label='Load Balancing 2')
plt.plot(numbers, max_l4, marker='^', label='Load Balancing 4')
plt.xscale('log')
plt.xlabel('n')
plt.ylabel('Max Occupancy')
plt.legend()
plt.show()

result_comparison_drop_double = {'n': numbers, 'max_drop': max_drop, 'theory_max_upper_drop': theory_max_upper_drop}
result_comparison_drop_double_df = pd.DataFrame.from_dict(result_comparison_drop_double)

result_comparison_load2_double = {'n': numbers, 'max_l2': max_l2, 'theory_max_load2': theory_max_load2}
result_comparison_load2_double_df = pd.DataFrame.from_dict(result_comparison_load2_double)

result_comparison_load4_double = {'n': numbers, 'max_l4': max_l4, 'theory_max_load4': theory_max_load4}
result_comparison_load4_double_df = pd.DataFrame.from_dict(result_comparison_load4_double)


display(result_comparison_drop_double_df)
#dfi.export(result_comparison_drop_double_df, 'result_comparison_drop_double_df.png')

display(result_comparison_load2_double_df)
#dfi.export(result_comparison_load2_double_df, 'result_comparison_load2_double_df.png')


display(result_comparison_load4_double_df)
#dfi.export(result_comparison_load4_double_df, 'result_comparison_load4_double_df.png')

