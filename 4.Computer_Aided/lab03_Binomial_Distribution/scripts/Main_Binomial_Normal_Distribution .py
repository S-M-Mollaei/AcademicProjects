
from Binomial_Inverse_Transform import inv_trans
from Binomial_Convolution import conv
from Binomial_Geometric import geo
import random
import timeit
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

random.seed(42)


# # Bin(n,p)  random variables generation

# initializition of input parameters

n = [10, 100, 1e6]
p = [0.5, .01, 1e-5]
#n = [10]
#p = [0.5]
k = 100

# implementation of convolution method to generate k instances of r.v.
conv_dic = {}

for i in range(len(n)):
    
    start = timeit.default_timer()
    
    for j in range(k):
        x_conv = conv(n[i], p[i])
        
    stop = timeit.default_timer()
    conv_dic[str((n[i], p[i]))] = (stop - start)


conv_dic.keys(), conv_dic.values()


# implementation of inverse-transform method to generate k instances of r.v.
inv_dic = {}

for i in range(len(n)):
    
    start = timeit.default_timer()
    
    for j in range(k):
        x_inv_trans = inv_trans(n[i], p[i])
        
    stop = timeit.default_timer()
    inv_dic[str((n[i], p[i]))] = (stop - start)

inv_dic.keys(), inv_dic.values()


# implementation of geometric method to generate k instances of r.v.
geo_dic = {}


for i in range(len(n)):
    
    start = timeit.default_timer()

    for j in range(k):
        x_geo = geo(n[i], p[i])
        
    stop = timeit.default_timer()
    geo_dic[str((n[i], p[i]))] = (stop - start)


geo_dic.keys(), geo_dic.values()


# plotting

plt.scatter(list(conv_dic.keys()), list(conv_dic.values()), c='red', marker='*')
plt.xlabel('(n, p)')
plt.ylabel('Time')
plt.title('Convolution')
plt.show()


plt.scatter(list(inv_dic.keys()), list(inv_dic.values()), c='black', marker='s')
plt.xlabel('(n, p)')
plt.ylabel('Time')
plt.title('Inverse_Transformation')
plt.show()



plt.scatter(list(geo_dic.keys()), list(geo_dic.values()), c='blue', marker='^')
plt.xlabel('(n, p)')
plt.ylabel('Time')
plt.title('Geometric')
plt.show()


# #  Normal random variables generation



# defining function to generate normal r.v.

def normal(snd, x_lin):
    
    c = max(snd.pdf(x_lin))

    for i in range(len(x_lin)):
        
        x = random.uniform(a, b)
        y = random.uniform(0, c)

        if y <= snd.pdf(x):
            return x


# parameters initialization
num = 1000
rv_list = []

a = -5
b = 5

# computing mean and standard deviation based on the parameters
mean = (a + b)/2
std = 2

# generating numbers over support [a,b] and normal distribution
x_lin = np.linspace(a, b, num)
snd = norm(mean, std)
f = snd.pdf(x_lin)

# generating a list of normal r.v.
for i in range(num):
    rv_list.append(normal(snd, x_lin))


# Plotting pdf and cdf of two distributions

# the main normal distribution

m, v = snd.stats(moments='mv')
print(f'mean is {m} and variance is {v}')
plt.plot(x_lin, f, label='PDF')
plt.plot(x_lin, snd.cdf(x_lin), label='PDF')
plt.legend()
plt.title('Main Normal Distribution')
plt.xlabel('x')
plt.show()



# histogram of generated r.v. distribution
rv_mean = sum(rv_list)/len(rv_list)
rv_var = np.var(rv_list)
print(f'r.v. mean is {rv_mean} and r.v. variance is {rv_var}')

plt.hist(rv_list)
plt.ylabel('Number of Occurance')
plt.xlabel('x')
plt.title('Histogram')


count, bins_count = np.histogram(rv_list)
  
# finding the PDF of the histogram using count values
pdf = count / sum(count)
  
# using numpy np.cumsum to calculate the CDF
# We can also find using the PDF values by looping and adding
cdf = np.cumsum(pdf)
  
# plotting PDF and CDF
plt.plot(bins_count[1:], pdf, color="red", label="PDF")
plt.plot(bins_count[1:], cdf, label="CDF")
plt.legend()
plt.title('Generated Normal Distribution')
plt.xlabel('x')



