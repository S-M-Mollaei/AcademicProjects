# %% [markdown]
# # Description
# 
# 11 - Priority queueWorkshop
# 
# Write the simulator for a queue with k servers, and waiting line of size N, which implements a strict priority service discipline (with preemption).
# 
# Customers are partitioned into two classes: High Priority (HP) customers and Low Priority (LP)
# 
# Both arrival processes are Poisson with rate  lambda_{HP}  and  lambda_{LP} respectively. 
# 
# LP customers are served only when no  HP customers are in the waiting line.
# 
# The service of a LP customer is  potentially interrupted  upon the arrival of   a HP customer if no  servers are idle.  Furthermore, upon the arrival of a HP customer, to accomodate the arriving customers,  a LP customer is potentially  dropped if there is not room in the waiting line.
# 
# Plot of the average delay  for HP customers, LP customers, end the aggregate average delay 
# 
# (i.e. the average delay on all customers) when lambda_{HP}=\lambda_{LP}=0.2, 0.4, 0.8, 1.4, 2.0, 2.4, 
# 
# 2.8, k=2, N=1000
# 
# a) E[S]_{HP}= E[S]_{LP}=1.0 
# 
# b) E[S]_{HP}=1/2 and  E[S]_{LP}=3/2  
# 
# Consider the three "usual" distributions for the service time:
# 
# EXP: exponentially distributed with mean= E[S]_{*P}
# 
# DET: deterministic =E[S]_{*P}
# 
# HYP:  distributed according to a hyper-exponential distribution with mean=E[S]_{*P} standard 
# 
# deviation=10*E[S]_{*P}

# %%
import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(66)
np.random.seed(66)

# %% [markdown]
# # Definition of classes and functions

# %%
# Class of server to serve each client
class Server:
    def __init__(self, server_name, server_status=0):
        self.server_name = server_name
        self.server_status = server_status

class PriorityQueue:
    def __init__(self):
        self.elements = []
        self.arrival_time_list = []
        # self.departure_time = 0

    def empty(self):
        return len(self.elements) == 0

    # Putting each client in the list to be served and set an arrival time for that by piosson distribution
    def put(self, item, priority, arrival_time):
        self.elements.append([priority, item])
        inter_arrival = np.random.poisson(lambda_poisson)
        self.arrival_time_list.append(arrival_time + inter_arrival)
        


    def get(self,departure_time, service_time):
        # Get the highest priority element to serve it first
        highest_priority_element = max(self.elements, key=lambda x: x[0])
        highest_priority_index = self.elements.index(highest_priority_element)
        
        departure_time = max(departure_time, self.arrival_time_list[highest_priority_index]) + service_time
        arr_time = self.arrival_time_list[highest_priority_index]
        
        # Remove the highest priority element
        self.elements.remove(highest_priority_element)
        del self.arrival_time_list[highest_priority_index]
        
        # Return the highest priority element
        return highest_priority_element, arr_time, departure_time

def hyper_expo() -> float:

    """
    This function generate an hyperexponential time according the definition of the hyperexponential itself.
    Distribution's parameters found by solving the corresponding linear system for target mean and std.
    More on that in the report.
    """
    p = .5
    l1 = 1/6
    l2 = 1/8
    u = random.random()
    if u <= p:
        expec = l1
    else:
        expec = l2
    service = np.random.exponential(1/expec)
    
    return service

# %% [markdown]
# # Running the simulation
# The number of servers is 2 and the length of queue is 1000.

# %%
# Initiate params of simulation
service_time_distribution = ['expo', 'hyperexpo', 'deterministic']
lambda_list = [0.2, 0.4, 0.8, 1.4, 2.0, 2.4, 2.8]
dict_plot = {}
dict_plot_lambda = {}


for dist in service_time_distribution:
    LP = []
    HP = []
    total = []
    
    for lambda_poisson in lambda_list:
        
        
        # Generating desired number of servers
        S1 = Server(server_name='S1')
        S2 = Server(server_name='S2')
        server_number = 2

        # Initiate priority class
        pq = PriorityQueue()

        dep_list=[0,0]
        arrive_time = 0
        counter = 0
        leaving_time_s1 = 0
        leaving_time_s2 = 0
        queue_length = 0
        queue_length_max = 100

        delay_LP = []
        delay_HP = []
        delay_total = []

        ave_delay_LP = []
        ave_delay_HP = []
        ave_delay_total = []
        
        # Start simulation
        while arrive_time < 1000 and queue_length < queue_length_max:
            
            # Generating service time with specific distribution
            if dist == 'expo':
                service_time = np.random.exponential(1)   
            elif dist == 'hyperexpo':
                service_time = hyper_expo()
            else:
                service_time = 1
                
            for _ in range(server_number):
                arrive_time += 1
                counter += 1
                priority = random.randint(0, 1)
                pq.put('client' + str(counter), priority, arrive_time)
                
                queue_length +=1

                # check if the servers can be in the serving condition
                if leaving_time_s1 > arrive_time:
                    S1.server_status = 1
                else:
                    S1.server_status = 0

                if leaving_time_s2 > arrive_time:
                    S2.server_status = 1
                else:
                    S2.server_status = 0

                # serving clients
                if S1.server_status == 0 and not pq.empty():
                    
                    a, get_in_time_s1, leaving_time_s1 = pq.get(dep_list[0], service_time)
                    dep_list[0] = leaving_time_s1
                    S1.server_status = 1
                    
                    if a[0] == 1:
                        p = "HP"
                        delay_HP.append(leaving_time_s1-get_in_time_s1) 
                        ave_delay_HP.append(sum(delay_HP)/len(delay_HP))
                        
                        delay_total.append(leaving_time_s1-get_in_time_s1)
                        ave_delay_total.append(sum(delay_total)/len(delay_total))
                    else:
                        p = "LP"
                        delay_LP.append(leaving_time_s1-get_in_time_s1) 
                        ave_delay_LP.append(sum(delay_LP)/len(delay_LP)) 
                        
                        delay_total.append(leaving_time_s1-get_in_time_s1)
                        ave_delay_total.append(sum(delay_total)/len(delay_total))
                    
                    print(f'{a[1]} as {p} arrived at {get_in_time_s1} and departed at {leaving_time_s1} from server {S1.server_name} with service_time {service_time}')
                    
                    queue_length -=1
                
                if S2.server_status == 0 and not pq.empty():
                    a, get_in_time_s2, leaving_time_s2 = pq.get(dep_list[1], service_time)
                    dep_list[1] = leaving_time_s2
                    S2.server_status = 1
                    
                    if a[0] == 1:
                        p = "HP"
                        delay_HP.append(leaving_time_s2-get_in_time_s2) 
                        ave_delay_HP.append(sum(delay_HP)/len(delay_HP))
                        
                        delay_total.append(leaving_time_s2-get_in_time_s2)
                        ave_delay_total.append(sum(delay_total)/len(delay_total))
                    else:
                        p = "LP"
                        delay_LP.append(leaving_time_s2-get_in_time_s2) 
                        ave_delay_LP.append(sum(delay_LP)/len(delay_LP)) 
                        
                        delay_total.append(leaving_time_s2-get_in_time_s2)
                        ave_delay_total.append(sum(delay_total)/len(delay_total))
                    
                    print(f'{a[1]} as {p} arrived at {get_in_time_s2} and departed at {leaving_time_s2} from server {S2.server_name} with service_time {service_time}')
                    
                    queue_length -=1
                    
                dict_plot[dist, lambda_poisson] = [ave_delay_LP, ave_delay_HP, ave_delay_total]
        LP.append(sum(ave_delay_LP)/len(ave_delay_LP))
        HP.append(sum(ave_delay_HP)/len(ave_delay_HP))
        total.append(sum(ave_delay_total)/len(ave_delay_total))
    dict_plot_lambda[dist] = [LP, HP, total]

# %%
list_lambda_plot = [0.2, 0.4, 0.8, 1.4, 2.0, 2.4, 2.8]
client_plot = ['LP', 'HP', 'Total']

for k in dict_plot_lambda.keys():
    plt.figure()
    c = 0
    for d in dict_plot_lambda[k]:
        plt.plot(list_lambda_plot, d, label=client_plot[c])
        c += 1
        plt.xlabel('Lambda')
        plt.ylabel('Average Delay')
        plt.title(k)
        plt.legend()
        plt.savefig(f'Distribution {k}.png')

# %%
name_plot_list = ['Ave_Delay_LP', 'Ave_Delay_HP', 'Ave_Delay_total']

for k in dict_plot.keys():
    index = 0
    for d in dict_plot[k]:
        plt.figure()
        plt.plot(d)
        plt.xlabel('Time')
        plt.ylabel(f'{name_plot_list[index]}')
        index += 1
        plt.title(k)
        plt.savefig(f'{k}.png')
    break
    


