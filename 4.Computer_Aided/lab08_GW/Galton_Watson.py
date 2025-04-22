# %% [markdown]
# 14 - Galton-Watson ProcessesWorkshop
# Through simulation evaluate the extinction probability within  generation  i  (q_i) and the asymptotic extinction  probability  (q) for  a Galton-Watson process in which the number of children of an individual Y is  distributed as a Poisson(lambda) R.V.  with lambda=0.6, 0.8, 0.9 0.95, 0.99, 1.01, 1.05, 1.1, 1.3.
# 
# Compare the results you  obtain with theoretical predictions, (by  finding numerically, when needed,  the solution of q= phi_Y(q)) 
# 
# In particular, you are requested to specify the stopping condition you have implemented in order to empirically "detect"   non-extinction condition.  Please try to provide a theoretical justification to such condition.
# 
# For the case  \lambda=0.8, obtain the empirical distribution (histogram) on the number of nodes in the tree.

# %%
import numpy as np
import collections
from collections import defaultdict
import matplotlib.pyplot as plt

np.random.seed(32)

# %%
# computing the percentage of each generation from cumulative distribution  
def percent_computation(target_dict, simulation_run_time):
    
    survival = {}
    extinction = {}
    
    for k, v in target_dict.items():
        temp_s = v / simulation_run_time
        survival[k] = temp_s
        
        temp_e = 1 - temp_s
        extinction[k] = temp_e
    
    return survival, extinction
    

# %%
# ******simulation parameters******
# first generation
x0 = 1
# condition to terminate creating more generation, since this number is enough to simulate the actual behavior of simulation based on different lambda
generation_limit = 30

simulation_run_time = 500
lambda_list = [0.6, 0.8, 0.9, 0.95, 0.99, 1.01, 1.05, 1.1, 1.3]
lambda_gen_dict = {}

nodes_number_list = []

for lambda_param in lambda_list:
    
    index_dict = {}
    
    # producing dictionary to compute cumulative repetitaion of each generation
    for i in range(generation_limit + 1):
        index_dict[i] = 0
    
    
    for _ in range(simulation_run_time):
        
        generation_dict = defaultdict(list)

        generation = 0
        generation_dict[generation].append(x0)
        
        # terminating creating generation when no children is provided or reaching certain number of generation
        while(sum(generation_dict[generation]) != 0) and generation <= generation_limit:
            
            temp_index = generation + 1
            
            for _ in range(sum(generation_dict[generation])):
                generation_dict[temp_index].append(np.random.poisson(lambda_param))
                
            generation += 1
            # print(generation_dict)
        
        # computing node numbers for lambda = 0.8
        if lambda_param == 0.8:
            nodes = 0
            for val in generation_dict.values():
                nodes += sum(val)
            nodes_number_list.append(nodes)
        
        # calculating cumulative repetition
        for i in range(generation):
            index_dict[i] += 1 
    
    # print(index_dict)
    survival_dict, extinction_dict = percent_computation(index_dict, simulation_run_time)
    
    lambda_gen_dict[lambda_param] = [survival_dict, extinction_dict]


# %%
for k, v in lambda_gen_dict.items():
    plot_s = v[0]
    # plot_e = v[1]
    plt.plot(plot_s.keys(), plot_s.values(), label=f'lambda = {k}')
    plt.xlabel('Generation')
    plt.ylabel('Survival Probability')
    plt.legend()

# %%
for k, v in lambda_gen_dict.items():
    # plot_s = v[0]
    plot_e = v[1]
    plt.plot(plot_e.keys(), plot_e.values(), label=f'lambda = {k}')
    plt.xlabel('Generation')
    plt.ylabel('q_i')
    plt.legend()

# %%
plt.hist(nodes_number_list)
plt.xlabel('Nodes')
plt.ylabel('Counts')


