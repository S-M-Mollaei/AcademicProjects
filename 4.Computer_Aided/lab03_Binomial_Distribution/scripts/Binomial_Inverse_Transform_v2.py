import random

random.seed(42)

# to calculate x!
def factorial(x):
    f = 1
    for i in range(int(x)):
        f = f * (i + 1)
    return f


# To compute binomial coefficients 
# $$\binom{a}{b}=\frac{a!}{{b!}{(a-b)!}}$$

def bincoeff(a, b):
    comb = factorial(a)/(factorial(b) * factorial(a - b))
    return comb


# To compute bpmf 
# $$bpmf(n, r, p)={\binom{n}{r}}{{p}^{r}}{{(1 - p)}^{n - r}}$$

def bpmf(n, r, p):
    return bincoeff(n, r) * p ** r * (1 - p) ** (n - r)


# To compute bpmf 
# $$bcdf(n, k, p)=\sum{\binom{n}{k}}{{p}^{k}}{{(1 - p)}^{n - k}}$$

def floor(x):
    if x >= 0:
        f = int(x)
    else:
        f = int(x) - 1
    return f

def bcdf(n, k , p):
    pVal = 0
    k_floor = floor(k)
    for i in range(k_floor + 1):
        pVal = pVal + bpmf(n, i, p)
    return pVal

def inv_trans(n, p):
    
    A = []
    
    for i in range(int(n)):
        A.append(bcdf(n, i, p))
    
    u = random.uniform(0, 1)
    
    for x in range(int(n-1)):
        if u >= A[x] and u <= A[x+1]:
            return x


