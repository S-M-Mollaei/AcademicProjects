import random

random.seed(42)

def conv(n, p):
    
    x = 0
    u = []
    for i in range(int(n)):
        u.append(random.uniform(0,1))
    
    for i in range(len(u)):
        if u[i] < p:
            x += 1
    return x

