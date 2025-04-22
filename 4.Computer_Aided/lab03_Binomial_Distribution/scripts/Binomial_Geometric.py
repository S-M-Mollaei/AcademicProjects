import random
import math

random.seed(42)

def geo(n, p):
    m = 1
    q = 0
    flag = 1
    
    while flag:
        u = random.uniform(0, 1)
        g = math.ceil(math.log(u)/math.log(1-p))
        q = q + g
        if q > n:
            flag = 0
            return (m - 1)
        else:
            m += 1



