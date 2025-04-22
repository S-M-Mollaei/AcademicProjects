# %% [markdown]
# # Libraries

# %%
import numpy as np
import re
import pympler.asizeof
import hashlib
import sys
import math
from collections import defaultdict
import matplotlib.pyplot as plt
# pip install bitarray
from bitarray import bitarray

np.random.seed(32)

# %% [markdown]
# # Functions

# %%
def sentence_maker(word_list, stride):
    # create sentence with specific length of stride from list of words
    sentence_set = []

    for i in range(len(word_list) - stride):
        temp = ''
        for j in range(stride):
          temp += word_list[i+j]
          temp += ' '
        sentence_set.append(temp.strip())

    return sentence_set


def get_sentence_from_words(poem_words, given_size_sentence):
  # getting sentences set from words
  sentences_list = sentence_maker(poem_words, given_size_sentence)
  sentences_set = set(sentence_maker(poem_words, given_size_sentence))
  total_sentences_number = len(sentences_set)
  total_sentences_size = pympler.asizeof.asizeof(sentences_set)

  return sentences_list, sentences_set, total_sentences_number, total_sentences_size


def fingerprint_maker(sentences, b):
  # compute the hash of a given string using md5 on a range [0,n-1]
  fp_set = set()
  n = 2 ** b

  for line in sentences:
    word_hash = hashlib.md5(line.encode('utf-8'))  # md5 hash
    # md5 hash in integer format
    word_hash_int = int(word_hash.hexdigest(), 16)
    h = word_hash_int % n  # map into [0,n-1)
    fp_set.add(h)

  return fp_set


def finding_min_bits(sentence_words_set, total_sentences_number, bits_nonsufficient):
    # start from least bit to get sufficient bit
    for b in range(bits_nonsufficient, bits_nonsufficient + 20, 1):

      fingerprint_set = fingerprint_maker(sentence_words_set, b)
      total_number_fingerprint_set = len(fingerprint_set)

      if total_number_fingerprint_set == total_sentences_number:
        return b


def bit_string_maker(sentences, b):
  # compute the hash of a given string using md5 on a range [0,n-1]
  n = 2 ** b

  bsa = bitarray(n)
  bsa.setall(0)

  for line in sentences:
    word_hash = hashlib.md5(line.encode('utf-8'))  # md5 hash
    # md5 hash in integer format
    word_hash_int = int(word_hash.hexdigest(), 16)
    h = word_hash_int % n  # map into [0,n-1)
    bsa[h] = 1


  false_positive_prob = bsa.count(1) / n
  
  return bsa, false_positive_prob



def bloom_filter_maker(sentences, b, k):
  
  def hash_generating(target):
    word_hash = hashlib.md5(target.encode('utf-8'))  # md5 hash
    # md5 hash in integer format
    word_hash_int = int(word_hash.hexdigest(), 16)
    h = word_hash_int % n  # map into [0,n-1)
    return h
      
  n = 2 ** b

  bfa = bitarray(n)
  bfa.setall(0)
  distinct_estimation = []

  for line in sentences:
    for i in range(k):
      bfa[hash_generating(line + str(i))] = 1

    
    # number of distinct element computation
    N = bfa.count(1)
    formula = -(n/k) * math.log(1 - (N/n))
    distinct_estimation.append(formula)
  # counting false positive prob.
  
  false_positive_prob = (bfa.count(1) / n) ** k
  
  return bfa, false_positive_prob, distinct_estimation



# %% [markdown]
# # Main Program

# %%
# *************************reading the file*************************

poem_pure = open('commedia.txt', 'r', encoding='UTF-8').read()
file_size = pympler.asizeof.asizeof(poem_pure) / 1e3
print(f'.txt file size is: {file_size} KB')

#******** start cleaning#********
# removing punctuation
poem_clean_punc = re.sub(r'[^\w\s]', '', poem_pure)
# spliting each line  
poem_verses = poem_clean_punc.split('\n')
# removing titles
del poem_verses[:8]

poem_words = []

for p in poem_verses:
    # removing inner titles
    if p.startswith('Inferno') or p.startswith('Purgatorio') or p.startswith('Purgatorio') or not p.strip():
        poem_verses.remove(p)
    else:
        p = p.strip()
        split_line = p.split(' ')
        for word in split_line:
            poem_words.append(word)

poem_words_distinct = set(poem_words)

# number and size of the total words of the poem
total_words_number = len(poem_words)
total_words_size = pympler.asizeof.asizeof(poem_words) / 1e3

total_words_distinct_number = len(poem_words_distinct)
total_words_distinct_size = pympler.asizeof.asizeof(poem_words_distinct) / 1e3

print(f'Number of all words: {total_words_number} with total size: {total_words_size} KB')
print(f'Number of all distinct words: {total_words_distinct_number} with total size: {total_words_distinct_size} KB')

# %%
# *************************fingerprint*************************

word_number = 6

print(f'***************Results for {word_number}_words sentences***************')
sentences_list, sentences_set, total_sentences_number, total_sentences_size = get_sentence_from_words(poem_words, word_number)

print(f'First two sentences of list: {sentences_list[0]}, {sentences_list[1]}')
print(f'First two sentences of set: {list(sentences_set)[1]}, {list(sentences_set)[2]}')
print(f'Number of distinct sentences with {word_number} words is: {total_sentences_number} with total size {total_sentences_size  / 1e3} KB and average size of each sentence {total_sentences_size/total_sentences_number} Byte')

print('########################################################################')
# computing required bits (b) to get specific false positive probability based on the number of input elements (m = total_sentences_number)

# computing bits for fingerprint
bits_nonsufficient = math.ceil(math.log2(total_sentences_number))

# to get minimun bit when total_number_fingerprint_set = total_sentences_number meaning no collison exists during storing
Bexp = finding_min_bits(sentences_set, total_sentences_number, bits_nonsufficient)
# computing epsilon by b = log2(m/ep)
epsilon_Bexp = total_sentences_number / (2 ** Bexp)
print(f'Minimum bit to have no collision is Bexp: {Bexp} with PFP: {epsilon_Bexp}')

# computing theorical bit number when p = 0.5 by m = 1.17 * sqrt(n) where n is bit number
Bteo = math.ceil(math.log2((total_sentences_number / 1.17) ** 2))
print(f'Theorical bit number when p = 0.5 by m = 1.17 * sqrt(n) where n is bit number is Bteo: {Bteo}')

print('########################################################################')
# bits_sufficient = math.ceil(math.log2(total_sentences_number/epsilon))

print(f'Required bits for P(false positive) = 0.63 is: {bits_nonsufficient}')
print(f'Required bits for P(false positive) = {epsilon_Bexp} is: {Bexp}')


# applying fingerprint function on sentences for sufficient bits
fingerprint_set = fingerprint_maker(sentences_set, Bexp)
total_number_fingerprint_set = len(fingerprint_set)
tatal_size_fingerprint_set = pympler.asizeof.asizeof(fingerprint_set) / 1e3
total_size_fingerprint_set_keys = sys.getsizeof(fingerprint_set) / 1e3

print(f'Total number of distinct fingerprint integers for sufficient bits case {Bexp} is: {total_number_fingerprint_set} with total size {tatal_size_fingerprint_set} KB')

# applying fingerprint function on sentences for sufficient bits
fingerprint_set_non = fingerprint_maker(sentences_set, bits_nonsufficient)
total_number_fingerprint_set_non = len(fingerprint_set_non)
tatal_size_fingerprint_set_non = pympler.asizeof.asizeof(fingerprint_set_non) / 1e3
total_size_fingerprint_set_keys_non = sys.getsizeof(fingerprint_set_non) / 1e3

print(f'Total number of distinct fingerprint integers for nonsufficient bits case {bits_nonsufficient} is: {total_number_fingerprint_set_non} with total size {tatal_size_fingerprint_set_non} KB')
        
print('########################################################################')

# %%
#  *************************bit string array*************************

b_list = [19, 20, 21, 22, 23]
bs_experimental_dict = {}
bs_theorical_dict = {}

# total size for bit string array is 2^b (bits) and  therorical epsilon = m/2^b so we compute memory of bit string array in bits corresponding 2^b
for b in b_list:
    
    bit_string_array, false_positive_prob_bsa = bit_string_maker(sentences_set, b)

    # by simulation
    memory_experimental = pympler.asizeof.asizeof(bit_string_array) # converting to bits 
    bs_experimental_dict[memory_experimental / 1e3] = false_positive_prob_bsa
    # by theory
    memory_therotical = 2 ** b
    epsilon_therorical = total_sentences_number / memory_therotical
    bs_theorical_dict[memory_therotical / (8 * 1e3)] = epsilon_therorical

bs_experimental_dict, bs_theorical_dict

# %%
plt.figure()

plt.plot(bs_experimental_dict.keys(), bs_experimental_dict.values(), '-*', label='Simulation Case')
plt.plot(bs_theorical_dict.keys(), bs_theorical_dict.values(), '-*', label='Theory Case')

plt.xlabel('Storage (KB)')
plt.ylabel('Probability of False Positive (PFP)')
plt.title('Bit String Array')
plt.legend()
plt.grid()
plt.show()

# %%
# *************************performance implementation for analytical case*************************

b_list = [19, 20, 21, 22, 23]
k_list = [i for i in range(1, 100)]

dict_b_k_epsilon = defaultdict(list)

for b in b_list:
    for k in k_list:
        # computing pfp by (1 - e^(mk/n))^k
        n = 2 ** b
        pr_fasle_positive = (1 - math.exp(-(k * total_sentences_number)/n)) ** k
        
        dict_b_k_epsilon[b].append(pr_fasle_positive)

for key in dict_b_k_epsilon.keys():
    plt.plot(k_list, dict_b_k_epsilon[key], label=f'b={key}')
    plt.xlabel('k')
    plt.ylabel('Pr(false positive)')
    plt.legend()
    plt.grid()

# %%
# *************************finding optimum k for bloom filter implementation*************************
m = total_sentences_number
k_opt_bloom_filter = {}
epsilon_Kopt_bloom_filter = {}

for b in b_list:
    
    n = 2 ** b
    # using formula to get float k
    kopt_temp = (n/m) * math.log(2)
    ep_list = dict_b_k_epsilon[b]
    
    # find kopt by using analytical plot to evaluate which k generates less pfp
    i = 0
    while kopt_temp > k_list[i]:
        i += 1
    
    if ep_list[i] < ep_list[i-1]:
        kopt = k_list[i]
        i = 0
    else:
        kopt = k_list[i-1]
        i = 0
        
    k_opt_bloom_filter[n / (8 * 1e3)] = kopt
    # compute epsilon by kopt
    epsilon_opt_temp = (0.5) ** (kopt)
    epsilon_Kopt_bloom_filter[n / (8 * 1e3)] = epsilon_opt_temp
    
k_opt_bloom_filter, epsilon_Kopt_bloom_filter

# %%
plt.figure()
plt.plot(k_opt_bloom_filter.keys(), k_opt_bloom_filter.values(), '-*')
plt.xlabel('Bloom Filter Storage (KB)')
plt.ylabel('Kopt')
plt.title('Optimal Number of Hash Functions vs Storage')
plt.grid()
plt.show()

plt.figure()
plt.plot(epsilon_Kopt_bloom_filter.keys(), epsilon_Kopt_bloom_filter.values(), '-*')
plt.xlabel('Bloom Filter Storage (KB)')
plt.ylabel('Probability of False Positive ')
plt.title('Theorical Case with Kopt')
plt.grid()
plt.show()

# %%
# *************************bloom filter implementation with optimum Ks*************************

b_list = [19, 20, 21, 22, 23]
k_list = [i for i in k_opt_bloom_filter.values()]
b_k_list = dict(zip(b_list, k_list))

bf_experimental_dict = {}
bf_theorical_dict = {}
bf_distinct_estimation = {}

for b, k in b_k_list.items():
    # total size for bit string array is 2^b (bits) and  therorical epsilon = m/2^b so we compute memory of bit string array in bits corresponding 2^b
    bloom_filter_array, fasle_positive_prob_bfa, distinct_estimation = bloom_filter_maker(sentences_set, b, k)
    # by simulation
    memory_experimental = pympler.asizeof.asizeof(bloom_filter_array) * 8
    bf_experimental_dict[memory_experimental / (8 * 1e3)] = fasle_positive_prob_bfa
    # by theory
    memory_therotical = 2 ** b
    epsilon_therorical = (0.6185) ** (memory_therotical/total_sentences_number)
    # epsilon_therorical = 1 / (2 ** (memory_therotical / (1.44 * total_sentences_number)))
    bf_theorical_dict[memory_therotical / (8 * 1e3)] = epsilon_therorical
    # distinct element estimation
    bf_distinct_estimation[(b,k)] = distinct_estimation
    
bf_experimental_dict, bf_theorical_dict


# %%
plt.figure()
plt.plot(bf_experimental_dict.keys(), bf_experimental_dict.values(), '-*', label='Simulation Case')
plt.plot(bf_theorical_dict.keys(), bf_theorical_dict.values(), '-*', label='Theory Case')

plt.xlabel('Storage (KB)')
plt.ylabel('False Positive Probability')
plt.title('Bloom Filter')
plt.legend()
plt.grid()
plt.show()

# %%
est_range = np.array([i+1 for i in range(len(sentences_set))])

plt.figure()
temp = np.subtract(np.array(bf_distinct_estimation[(23,61)]), est_range)
plt.plot(est_range, temp, label=f'(b, k) = {bk}')
plt.xlabel('Sentence Number')
plt.ylabel('Difference')
plt.title('Difference between Distinct Estimation Value and Sentence Number')
plt.legend()
plt.grid()

plt.figure()
plt.plot(est_range, bf_distinct_estimation[(23,61)], label=f'(b, k) = {bk}')
plt.xlabel('Sentence Number')
plt.ylabel('Distinct Estimation Value')
plt.title('Distinct Element estimation')
plt.legend()
plt.grid()

# %%



