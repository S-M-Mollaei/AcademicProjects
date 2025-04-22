# %% [markdown]
# # This notebook creates real birthday distribution from https://github.com/fivethirtyeight/data/tree/master/births
# 

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random 

# %%
# creating dataframe based on the .txt file

column_name = ['year', 'month', 'date_of_month', 'day_of_week,births', 'births']
df = pd.read_csv('US_births_1994-2003_CDC_NCHS.txt', header=None, sep = ',', skiprows=1, index_col=None)
df.columns = column_name
df.head()

# %%
# group the 'month', 'date_of_month' columns to produce average number of people born on specific day

temp = df.copy()
month_day_group = temp.groupby(['month', 'date_of_month'], as_index=False).mean()
month_day_group.head() #, len(month_day_group)

# %%
# computing the distribution of the birthday data

sum_births = month_day_group['births'].sum()
month_day_group["prob"] = month_day_group['births'] / sum_births
month_day_group["cdf"] = month_day_group["prob"].cumsum()
month_day_group

# %%
plt.plot(range(len(month_day_group)), month_day_group['prob'])
plt.xlabel('Days')
plt.ylabel('Probability')
plt.title('Distribution of birthdays along a year')
plt.show()

# %%
# defining the function in order to create a random day [1, 366] from the real distribution

def birth_cdf_inverse():
    # np.random.seed(seed)
    u = np.random.uniform(0, 1)
    for day, cdf in enumerate(month_day_group['cdf']):
        if u <= cdf:
            return (day)

# %%
generated_birth_cdf_inverse = []
for _ in range(1000):
    generated_birth_cdf_inverse.append(birth_cdf_inverse())

# %%
plt.hist(generated_birth_cdf_inverse), len(generated_birth_cdf_inverse)
plt.xlabel('Days')
plt.ylabel('Day Generation Number')
plt.title('Generating random birthdays of real distribution')

# %%



