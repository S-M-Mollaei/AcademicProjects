#!/usr/bin/env python
# coding: utf-8

# In[1]:


from queue import PriorityQueue
import random


# # Defining functions and classes

# In[2]:


def arrival(time, FES, queue):
    
    global users
    global customer
    
    # introducing random client arrival
    inter_arrival = random.expovariate(1.0/average_arrival_interval)
    FES.put((time + inter_arrival, 'arrival'))
    
    # managing the event 
    users += 1
    x = 'client' + str(customer)
    customer += 1
    
    # recording client id and put it in the list
    client = Client(x, time)
    queue.append(client)

    print(f'{client.name} arrived at {client.arrival_time}')
    
    # start the service in case the server is idle
    if users == 1:
        # scheduling random departure time to the clients
        service_time = random.expovariate(1.0/average_service_time)
        FES.put((time + service_time, 'departure'))
    


def departure(time, FES, queue):
    
    global users
    
    # manipulating the list of clients to get FIFO orientation
    queue.reverse()
    client = queue.pop()
    queue.reverse()
    users -= 1
    
    print(f'{client.name} departured at {time}')
    
    # checking the number of clients in line
    if users > 0:
        # scheduling random departure time to the clients
        service_time = random.expovariate(1.0/average_service_time)
        FES.put((time + service_time, 'departure'))
        

class Client:
    def __init__(self, name, arrival_time):
        self.name = name
        self.arrival_time = arrival_time
        


# # Implementing the simulation

# In[3]:


# initialization of variables
time = 0
users = 0
customer = 1
queue = []
average_arrival_interval = 3
average_service_time = 6
FES = PriorityQueue()
# the first arrival at time 0
FES.put((0,'arrival'))

# the main loop to give the service to the clients until specific time
while time < 20:
    (time, event_type) = FES.get()
    if event_type == 'arrival':
        arrival(time, FES, queue)
    elif event_type == 'departure':
        departure(time, FES, queue)


# In[ ]:




