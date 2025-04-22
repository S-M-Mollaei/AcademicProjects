#!/usr/bin/env python
# coding: utf-8

# ***This code is related to online first-person game in which the winner is the one who kills all the opponents.***

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import random

random.seed(42)


# ***Every new position is assigned in way that they are in the battle area. We assume that new positions are created when all the players reach their destination.***

# In[3]:


def newPosition(dic, playerList, data):
    
    for i in playerList:
        
        xm = random.choice(data[:,0])
        ym = random.choice(data[:,1])
        
        dic[i].firstPosition = dic[i].secondPosition
        dic[i].secondPosition = (xm, ym)
        dic[i].move = 1


# ***This function simulate the movement of fighters. Each player goes towards new position just by selecting three adjacent points (in case speed is one cell per movement) in order to avoid players to go around the path and guarantee the convergence.***

# In[4]:


def move(dic, speed, playerList):
    global xc, yc
    reachNum = 0
    
    for i in playerList:
        
        pFirst = dic[i].firstPosition
        pSecond = dic[i].secondPosition
        pCurrent = pFirst
        
        if pCurrent == pSecond:
            dic[i].move = 0
            reachNum += 1
#             print(i,' is stopped')
            continue
        
        # In case the speed (step) is greater than the distance of the initial and final position\
        #(either along x axis or y axis),speed is ignored and the distance 
        #is selected as new speed along the certain axis.
        
        if pSecond[0] - pFirst[0] > 0: 
            if speed > pSecond[0] - pFirst[0]:
                x = pSecond[0] - pFirst[0]
            else:
                x = speed
        else:
            if speed > abs(pSecond[0] - pFirst[0]):
                x = pSecond[0] - pFirst[0]
            else:
                x = -speed
        
        if pSecond[1] - pFirst[1] > 0:
            if speed > pSecond[1] - pFirst[1]:
                y = pSecond[1] - pFirst[1]
            else:
                y = speed
        else:
            if speed > abs(pSecond[1] - pFirst[1]):
                y = pSecond[1] - pFirst[1]
            else:
                y = -speed
        
        # moving one step closer to the new position
        flag = 1
        while flag == 1 and dic[i].alive == 1:
            (xc, yc) = pCurrent
            
            (xc, yc) = random.choice([(xc + x, yc), (xc, yc + y), (xc + x, yc + y)])
            
            if (xc <= max(pFirst[0], pSecond[0])) and (xc >= min(pFirst[0], pSecond[0]))                and (yc <= max(pFirst[1], pSecond[1])) and (yc >= min(pFirst[1], pSecond[1])): 
                    
                    pCurrent = (xc, yc)
                    flag = 0

        dic[i].firstPosition = (xc, yc)
    return reachNum


# ***This function is defined to decide which player can live in case of encountering in specific position during movements.***

# In[5]:


# The winner is chosen based on random binary selection.
#If more than two players reach one position, just one of them can live.
def fight(dic, duration, playerList):
    killedNum = 0
    for i in playerList:

        if dic[i].alive:
            
            for j in playerList:
                if i != j:
                    
                    if dic[i].alive:

                        if dic[i].firstPosition == dic[j].firstPosition:
#                             print('fight')
                            dic[i].alive = random.choice([0, 1])
                            dic[j].alive = abs(1 - dic[i].alive)

                            if dic[i].alive == 0:
                                dic[i].timeToLive = duration
                                playerList.remove(i)
                                dic[i].move = 0
                                dic[j].kills += 1
                            else:
                                dic[j].timeToLive = duration
                                playerList.remove(j)
                                dic[j].move = 0
                                dic[i].kills += 1
                            killedNum += 1
    return killedNum


# In[6]:


class playerProfile:
    def __init__(self, name, firstPosition, secondPosition, currentPosition, kills, timeToLive):
        self.name = name
        self.firstPosition = firstPosition
        self.secondPosition = secondPosition
        self.kills = kills
        self.timeToLive = timeToLive
        self.alive = 1
        self.move = 0


# In[7]:


def simulator(coordinate, playerSpeed, number):
    
    #defning the batlle area, speed and player number as inputs
    mid_x, mid_y, max_x, max_y, min_x, min_y,step_size,num_points =  coordinate
    x_range = np.concatenate((np.arange(min_x, mid_x, step_size), np.arange(mid_x, max_x, step_size)))
    y_range = np.concatenate((np.arange(min_y, mid_y, step_size), np.arange(mid_y, max_y, step_size)))
    data = np.array([[x, y] for x in x_range for y in y_range])  # cartesian prod
    
    gameSpeed = playerSpeed
    playerNum = number

    playerDict = {}
    
    # producing one class for each player
    for i in range(playerNum):
        xp = random.choice(data[:,0])
        yp = random.choice(data[:,1])
        if (xp, yp) not in playerDict.values():
         playerDict[f'player_{i+1}'] = playerProfile(f'player_{i+1}', (xp, yp), (xp, yp), (0, 0), 0, 0)
        else:
            i -= 1
    
    # list of player's names
    playerList = list(playerDict.keys())
    
    # simulating the game for every specific inputs
    gameTime = 0
    while len(playerList) > 1:

        newPosition(playerDict, playerList, data)

        flag =1 
        while flag:

            reachPlayer = move(playerDict, gameSpeed, playerList)
            gameTime += 1
            killed = fight(playerDict, gameTime, playerList)

            if (reachPlayer + killed) == len(playerList):
                flag = 0

    print(f'The winner is {playerDict[playerList[0]].name} with timeToLive: {gameTime}') 
    
    return [gameTime, playerDict[playerList[0]].kills]


# ***Three lists of specific numbers to each input***

# In[ ]:


def doubleNum(u):
    return 2 * u

firstGrid = [50, 50, 110, 110, 0, 0, 1, 110]
grid = [list(map(lambda u: 2 * (i+1) * u, firstGrid)) for i in range(5)]

pace = [1, 2, 3]

fighters = [100, 200, 400, 600, 800, 1000]


# ***Start simulation and defining a dictonary for each input as key and the list of game time, killed players and average killed as value***

# In[9]:


coordinateDict = {}
for coordinate in grid:
    parameters = simulator(coordinate, pace[0], fighters[0])
    parameters.append((fighters[0] - 1)/fighters[0])
    coordinateDict[coordinate[7]] = parameters


# In[10]:


speedDict = {}
for speed in pace:
    parameters = simulator(grid[0], speed, fighters[0])
    parameters.append((fighters[0] - 1)/fighters[0])
    speedDict[speed] = parameters


# In[11]:


initialDict = {}
for num in fighters:
    parameters = simulator(grid[0], pace[0], num)
    parameters.append((num - 1)/num)
    initialDict[num] = parameters


# ***Start plotting***

# In[12]:


import pandas as pd


# In[13]:


coordinatePd = pd.DataFrame.from_dict(coordinateDict, orient='index', columns=['gameTime', 'killed', 'ave'])
coordinatePd


# In[17]:


speedPd = pd.DataFrame.from_dict(speedDict, orient='index', columns=['gameTime', 'killed', 'ave'])
speedPd


# In[14]:


initialPd = pd.DataFrame.from_dict(initialDict, orient='index', columns=['gameTime', 'killed', 'ave'])
initialPd


# In[18]:


import matplotlib.pyplot as plt


# In[25]:


plt.figure(figsize=(30,10))

plt.subplot(131)
plt.plot(coordinatePd.index, coordinatePd['gameTime'])
plt.ylabel('Time to Win')
plt.xlabel('Battle Area')
plt.subplot(132)
plt.plot(speedPd.index, speedPd['gameTime'])
plt.ylabel('Time to Win')
plt.xlabel('Game Speed')
plt.subplot(133)
plt.plot(initialPd.index, initialPd['gameTime'])
plt.ylabel('Time to Win')
plt.xlabel('Number of Player')
plt.show()


# In[26]:


plt.figure(figsize=(30,10))

plt.subplot(131)
plt.plot(coordinatePd.index, coordinatePd['killed'])
plt.ylabel('Killed')
plt.xlabel('Battle Area')
plt.subplot(132)
plt.plot(speedPd.index, speedPd['killed'])
plt.ylabel('Killed')
plt.xlabel('Game Speed')
plt.subplot(133)
plt.plot(initialPd.index, initialPd['killed'])
plt.ylabel('Killed')
plt.xlabel('Number of Player')
plt.show()


# In[27]:


plt.figure(figsize=(30,10))

plt.subplot(131)
plt.plot(coordinatePd.index, coordinatePd['ave'])
plt.ylabel('Average Killed')
plt.xlabel('Battle Area')
plt.subplot(132)
plt.plot(speedPd.index, speedPd['ave'])
plt.ylabel('Average Killed')
plt.xlabel('Game Speed')
plt.subplot(133)
plt.plot(initialPd.index, initialPd['ave'])
plt.ylabel('Average Killed')
plt.xlabel('Number of Player')
plt.show()


# In[ ]:




