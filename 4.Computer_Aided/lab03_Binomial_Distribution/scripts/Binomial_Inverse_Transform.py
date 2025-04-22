
import random
import math
import scipy.special
from scipy.stats import binom

random.seed(42)

def inv_trans(p, n):
    
    A = []
    
    for i in range(int(n)):
        A.append(scipy.stats.binom.cdf(i, n, p))
    
    u = random.uniform(0, 1)
    
    for x in range(int(n-1)):
        if u >= A[x] and u <= A[x+1]:
            return x


