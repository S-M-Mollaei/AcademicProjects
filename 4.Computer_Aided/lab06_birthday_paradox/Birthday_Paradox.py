# %% [markdown]
# # This code is about designing a simulator to evaluate the events of birthday conflicts, assuming:
# 
# 1. uniform distribution of the birthday during the year
# 2. realistic distribution of the birthday, taking from real statistics

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Real_Bth_Distribution import birth_cdf_inverse
from collections import defaultdict
from statistics import mean
from scipy.stats import t
import math

import random

# %%
# function to find how many students must enter the class to have a conflit of birthday with uniform distribution for generating random birthdays
# we put the number of days in a year 366 to include leap years

def uniform_birthday():
    
    # defining the paramaters
    year_number = 366
    list_days = np.zeros(year_number)
    student_entered = 0
    flag = 1
    
    # creating instances of uniform dist. untill a conflit appears
    while flag:

          # using uniform distribution function 
        rv_day = np.random.randint(0, year_number)
        
        # put each student in class in case there is not any student with the same birthday
        if list_days[rv_day] == 0:
            list_days[rv_day] = 1
            student_entered += 1
        else:
            flag = 0
                        
    return student_entered

# %%
# function to find how many students must enter the class to have a conflit of birthday with real distribution for generating random birthdays
# we put the number of days in a year 366 to include leap years

def real_birthday():

    # defining the paramaters
    year_number = 366
    list_days = np.zeros(year_number)
    student_entered = 0
    flag = 1
    
    # creating instances of real dist. untill a conflit appears
    while flag:
        
        # using real distribution function 
        rv_day = birth_cdf_inverse()
        
        # put each student in class in case there is not any student with the same birthday
        if list_days[rv_day] == 0:
            list_days[rv_day] = 1
            student_entered += 1
        else:
            flag = 0
                        
    return student_entered

# %%
simulation_seed = 32
np.random.seed(simulation_seed)

# %% [markdown]
# # Evaluating the average number of people to observe a conflict and comparing with the theoretical result

# %%
# noted that number of experiment (inctances) for each seed must be more 32 in order to meet central limit theorem
# so run the simulation for 1000 times which are independent from each other

simulation_iteration = 1000

number_student_conflict_uniform = []
number_student_conflict_real = []

# simulation
for _ in range(simulation_iteration):
    # put the number of entered students which producing a conflict in a list to get average number of entered students for having conflict
    number_student_conflict_uniform.append(uniform_birthday())
    number_student_conflict_real.append(real_birthday())

# computing estimated average number of entered students for having a conflict 
average_student_conflict_uniform = sum(number_student_conflict_uniform)/len(number_student_conflict_uniform)
average_student_conflict_real = sum(number_student_conflict_real)/len(number_student_conflict_real)


# %%
print('In theory for p=0.5 when n goes to infinity: for n = 365, m ≈ 22.3 and E[m] = 23.9 and our estimated E[m]s are: ')
print('***The results of the simulation***')
print(f'Average number for uniform case: {average_student_conflict_uniform}')
print(f'Average number for real case: {average_student_conflict_real}')

# %% [markdown]
# In order to compute the confidence interval for mean, we need to calculate average and variance of computed numbers and then determine the interval.
# Since both average and std are estimated, we need to use t-distribution with n-1 degree (1000-1) of freedom with alpha 0.5 for the interval.

# %%
# We now need the value of t. The function that calculates the inverse cumulative distribution is ppf.
# We need to apply the absolute value because the cumulative distribution works with the left tail, 
# so the result would be negative.averege_estimated_uniform = np.mean(number_student_conflict_uniform)
degree_of_freedom = simulation_iteration - 1
confidence = 0.95
t_distribution = np.abs(t.ppf((1-confidence)/2,degree_of_freedom))

# applying the formula for uniform case
averege_estimated_uniform = mean(number_student_conflict_uniform)
std_estimated_uniform = np.std(number_student_conflict_uniform)
confidence_interval_uniform = (averege_estimated_uniform - std_estimated_uniform * t_distribution / np.sqrt(simulation_iteration),
                       averege_estimated_uniform + std_estimated_uniform * t_distribution / np.sqrt(simulation_iteration)) 

# applying the formula for real case
averege_estimated_real = mean(number_student_conflict_real)
std_estimated_real = np.std(number_student_conflict_real)
confidence_interval_real = (averege_estimated_real - std_estimated_real * t_distribution / np.sqrt(simulation_iteration),
                       averege_estimated_real + std_estimated_real * t_distribution / np.sqrt(simulation_iteration)) 

print(f'The confidence interval for uniform case is: {confidence_interval_uniform}')
print(f'The confidence interval for real case is: {confidence_interval_real}')


# %% [markdown]
# # Evaluating the probability of birthday conflict in function of m and comparing with the theoretical result.

# %%
# list of students to put in all class to evaluate in how many classes a conflict exists with the same number of students >= 2 in each class
# noted that number of experiment (inctances) for each seed must be more 32 in order to meet central limit theorem
students = [i for i in range(2, 100)]
class_number = 20
dict_prob_conflict_uniform = {}
dict_prob_conflict_real = {}

# simulation 

for m in students:
    # create classes to put studets into
    list_class_uniform = [0] * class_number
    list_class_real = [0] * class_number
    
    # providing each class a constant number(m) of students and for the conflict after filling the classes
    for c in range(len(list_class_uniform)):

        # generating m birthdays for the current class
        birthdays_uniform = [(np.random.randint(0, 366)) for _ in range(m)]
        birthdays_real = [real_birthday() for _ in range(m)]
        
        # checking if there is a conflict in current class and make its flag 1
        if len(birthdays_uniform) != len(set(birthdays_uniform)):
            list_class_uniform[c] = 1
            
        if len(birthdays_real) != len(set(birthdays_real)):
            list_class_real[c] = 1
    
    # computing the probability of birthday conflict in function of number of student
    dict_prob_conflict_uniform[m] = sum(list_class_uniform)/len(list_class_uniform)
    dict_prob_conflict_real[m] = sum(list_class_real)/len(list_class_real)

# %% [markdown]
# In order to compute confidence interval for mean, we have two loops with size 100 and 20 which means the number of samples is 2000. 
# we need to calculate average and variance of computed numbers and then determine the interval.
# Since both average and std are estimated, we need to use t-distribution with n-1 degree (2000-1) of freedom with alpha 0.5 for the interval.

# %%
# We now need the value of t. The function that calculates the inverse cumulative distribution is ppf.
# We need to apply the absolute value because the cumulative distribution works with the left tail, 
# so the result would be negative.averege_estimated_uniform = np.mean(number_student_conflict_uniform)
simulation_iteration2 = len(students) * class_number
degree_of_freedom2 = simulation_iteration2 - 1
confidence2 = 0.95
t_distribution2 = np.abs(t.ppf((1-confidence2)/2,degree_of_freedom2))

# applying the formula for uniform case
averege_estimated_prob_conflict_uniform = mean(list(dict_prob_conflict_uniform.values()))
std_estimated_prob_conflict_uniform = np.std(list(dict_prob_conflict_uniform.values()))
confidence_interval_prob_conflict_uniform = (averege_estimated_prob_conflict_uniform - std_estimated_prob_conflict_uniform * t_distribution2 / np.sqrt(simulation_iteration2),
                       averege_estimated_prob_conflict_uniform + std_estimated_prob_conflict_uniform * t_distribution2 / np.sqrt(simulation_iteration2)) 

# applying the formula for real case
averege_estimated_prob_conflict_real = mean(list(dict_prob_conflict_real.values()))
std_estimated_prob_conflict_real = np.std(list(dict_prob_conflict_real.values()))
confidence_interval_prob_conflict_real = (averege_estimated_prob_conflict_real - std_estimated_prob_conflict_real * t_distribution2 / np.sqrt(simulation_iteration2),
                       averege_estimated_prob_conflict_real + std_estimated_prob_conflict_real * t_distribution2 / np.sqrt(simulation_iteration2))


print(f'Uniform case: mean is {averege_estimated_prob_conflict_uniform}, std is {std_estimated_prob_conflict_uniform} and confidece interval is{confidence_interval_prob_conflict_uniform}')
print(f'Real case: mean is {averege_estimated_prob_conflict_real}, std is {std_estimated_prob_conflict_real} and confidece interval is{confidence_interval_prob_conflict_real}')


# %%
# computing the theorical results for each m inj [2, 100] based on the formula 1 − exp(-m^2/(2*n))
dict_prob_conflict_theory = {}
n = 365
for i in dict_prob_conflict_uniform.keys():
    formula = 1 - np.exp(-i**2/(2*n))
    dict_prob_conflict_theory[i] = formula

# %%

plt.plot(dict_prob_conflict_theory.keys(), dict_prob_conflict_theory.values(), lw=2, label='Theorical')
plt.plot(dict_prob_conflict_theory.keys(), dict_prob_conflict_uniform.values(), lw=2, label='Uniform Distribution')
plt.plot(dict_prob_conflict_theory.keys(), dict_prob_conflict_real.values(), label='Real Distribution')
plt.xlabel('Number of student')
plt.ylabel('Prob(birthday collision)')
plt.legend()
plt.show()

# %% [markdown]
# # Extension
# One possible extension could be to use generalized version of this problem which is choosing arbitrary n instead 365(numbers days in a year) and compare the with theorical results. 

# %%
# function to find how many students must enter the class to have a conflit of birthday with uniform distribution for generating random birthdays
# we put the number of days in a year 366 to include leap years

def uniform_birthday_extension():
    
    # defining the paramaters
    year_number = 1000
    list_days = np.zeros(year_number + 1)
    student_entered = 0
    flag = 1
    
    # creating instances of uniform dist. untill a conflit appears
    while flag:

        # using uniform distribution function 
        rv_day = np.random.randint(0, year_number)
        # rv_day = random.randint(0, year_number)
        
        # put each student in class in case there is not any student with the same birthday
        if list_days[rv_day] == 0:
            list_days[rv_day] = 1
            student_entered += 1
        else:
            flag = 0
                        
    return student_entered

# %%
simulation_seed_extension = 32
np.random.seed(simulation_seed_extension)

# %%
# noted that number of experiment (inctances) for each seed must be more 32 in order to meet central limit theorem
# so run the simulation for 1000 times which are independent from each other

simulation_iteration_extension = 1000

number_student_conflict_uniform_extension = []

# simulation
for _ in range(simulation_iteration_extension):
    # put the number of entered students which producing a conflict in a list to get average number of entered students for having conflict
    number_student_conflict_uniform_extension.append(uniform_birthday_extension())

# computing estimated average number of entered students for having a conflict 
average_student_conflict_uniform_extension = sum(number_student_conflict_uniform_extension)/len(number_student_conflict_uniform_extension)

# %%
formula_extension_m = 1.17 * math.sqrt(1000)

print(f'In theory for p=0.5 when n goes to infinity: for n = 1000, m ≈ {formula_extension_m} is: ')
print('***The results of the simulation***')
print(f'Average number for uniform case: {average_student_conflict_uniform_extension}')


