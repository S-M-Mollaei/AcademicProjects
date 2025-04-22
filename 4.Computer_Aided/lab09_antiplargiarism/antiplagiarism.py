# %% [markdown]
# # Description
# 15 - Antiplagiarism poem system - hash tables and fingerprintingWorkshop
# 
# Devise an antiplagiarism software specialized on poems. For simplicity, assume that the software is able to detect sentences taken from "La Divina Commedia" by Dante Alighieri. The text of the poem can be downloaded from here. 
# 
# The text of the poem can be downloaded from here.
# 
#     What is the input of the software?
#     What is the output of the software?
#     Write a software in python to read the text
#     Compute the total number of words and verses.
#     Compute the total number of distinct words.
#     Define a simple way procedure to define sentences.
#     Which data structure would you use for the antiplagiarism software?
#     Which algorithm would you use to detect plagiarism?
# 
# The software must detect sentences of a given size S in terms of words. 
# 
# Compare a solution in which an hash table stores the whole sentences and the ones storing just the fingerprints. 
# 
#     What are the inputs and the outputs of the software?
#     How many sentences are stored for S=4 and S=8?
#     What is the experimental amount of stored data in bytes, independently from the adopted data structure?
#     Implement a solution storing the sentence strings in a python set. What is the actual amount of memory occupancy?
#     Implement a solution storing the fingerprints in a python set. 
#     Show the formula and the graph with the fingerprint size in function of the probability of false positive for this specific scenario, in two conditions: S=4, S=8.
#     Show a graph with the  actual amount of memory occupancy in function of the probability of false positive for S=4 and S=8.
#     Under which conditions the fingerprinting allows to reduce the actual amount of memory? Is it independent from S? Why?
# Hints:
# 
# To compute the actual amount of memory, use pympler.asizeof .
# 
# The following code could be used to compute the fingerprint of a string on a range [0,n-1]
# 
# import hash-lib
# 
# compute the hash of a given string using md5 on a range [0,n-1]
# 
# word = 'Politecnico' # string to hash
# 
# word_hash = hashlib.md5(word.encode('utf-8')) # md5 hash
# 
# word_hash_int = int(word_hash.hexdigest(), 16) # md5 hash in integer format
# 
# h = word_hash_int % n # map into [0,n-1]

# %%
import numpy as np
import re
import pympler.asizeof
import hashlib
import sys
import math
from collections import defaultdict
import matplotlib.pyplot as plt

np.random.seed(32)

# %%
# function to create sentence with length stride from list of words
def sentence_maker(word_list, stride):
    
    sentence_set = []
    
    for i in range(len(word_list) - stride):
        temp = ''
        for j in range(stride):
          temp += word_list[i+j]
          temp += ' '
        sentence_set.append(temp.strip())
         
    
    return sentence_set


# producing fingerprint of each element
def fingerprint_maker(sentences, b):
  # compute the hash of a given string using md5 on a range [0,n-1]
  temp = defaultdict(list)
  fp_set = set()
  n = 2 ** b
  
  for line in sentences:
    word_hash = hashlib.md5(line.encode('utf-8')) # md5 hash
    word_hash_int = int(word_hash.hexdigest(), 16) # md5 hash in integer format
    h = word_hash_int % n # map into [0,n-1)
    fp_set.add(h)
    temp[h].append(line)
    
  return fp_set

def get_bit_memory_plot(m, word_number):
    ep = np.linspace(0, 1, 1000000) # epsilon numbers between [0,1]
    ep_inv = np.power(ep, -1) # inverse of epsilon
    ep_inv_m = ep_inv * m # multiplying by m
    bits_array = np.log2(ep_inv_m) # computing bits array
    memory_array = bits_array * m / 8000 # memory in KB
    
    plt.figure()
    plt.plot(ep, bits_array)
    plt.xlabel('Epsilon')
    plt.ylabel('Bit Number')
    plt.title(f'Bits vs Pr(false positive)')
    
    plt.figure()
    plt.plot(ep, memory_array)
    plt.xlabel('Epsilon')
    plt.ylabel('Memory (KB)')
    plt.title(f'Memory Storage vs Pr(false positive)')
    plt.show()

def get_sentence_from_words(poem_words, given_size_sentence, epsilon_param):
    # getting sentences set from words
    sentence_words_list = sentence_maker(poem_words, given_size_sentence)
    sentence_words_set = set(sentence_maker(poem_words, given_size_sentence))
    total_sentences_number = len(sentence_words_set)
    total_sentences_size = pympler.asizeof.asizeof(sentence_words_set) / 1e6

    print(f'First two sentences of list: {sentence_words_list[0]}, {sentence_words_list[1]}')
    print(f'First two sentences of set: {list(sentence_words_set)[1]}, {list(sentence_words_set)[2]}')
    print(f'Number of distinct sentences with {word_number} words is: {total_sentences_number} with total size {total_sentences_size} MB')
    
    # computing required bits (b) to get specific false positive probability based on the number of input elements (m = total_sentences_number)
    epsilon = epsilon_param

    # computing bits for fingerprint
    bits_nonsufficient = math.ceil(math.log2(total_sentences_number))
    bits_sufficient = math.ceil(math.log2(total_sentences_number/epsilon))

    print(f'Required bits for P(false positive) = 0.63 is: {bits_nonsufficient}')
    print(f'Required bits for P(false positive) = 1e-4 is: {bits_sufficient}')
    
    
    # applying fingerprint function on sentences for sufficient bits
    fingerprint_set = fingerprint_maker(sentence_words_set, bits_sufficient)
    total_number_fingerprint_set = len(fingerprint_set)
    tatal_size_fingerprint_set = pympler.asizeof.asizeof(fingerprint_set) / 1e6
    total_size_fingerprint_set_keys = sys.getsizeof(fingerprint_set) / 1e6

    print(f'Total number of distinct fingerprint integers for sufficient bits case {bits_sufficient} is: {total_number_fingerprint_set} with total size {tatal_size_fingerprint_set} MB')
    
    # applying fingerprint function on sentences for sufficient bits
    fingerprint_set_non = fingerprint_maker(sentence_words_set, bits_nonsufficient)
    total_number_fingerprint_set_non = len(fingerprint_set_non)
    tatal_size_fingerprint_set_non = pympler.asizeof.asizeof(fingerprint_set_non) / 1e6
    total_size_fingerprint_set_keys_non = sys.getsizeof(fingerprint_set_non) / 1e6

    print(f'Total number of distinct fingerprint integers for nonsufficient bits case {bits_nonsufficient} is: {total_number_fingerprint_set_non} with total size {tatal_size_fingerprint_set_non} MB')
    
    
    return sentence_words_set, total_sentences_number, total_sentences_size


  
  

# %%
# reading the file
poem_pure = open('commedia.txt', 'r', encoding='UTF-8').read()
file_size = pympler.asizeof.asizeof(poem_pure) / 1e6
print(f'.txt file size is: {file_size} MB')

## start cleaning
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
total_words_size = pympler.asizeof.asizeof(poem_words) / 1e6

total_words_distinct_number = len(poem_words_distinct)
total_words_distinct_size = pympler.asizeof.asizeof(poem_words_distinct) / 1e6

print(f'Number of all words: {total_words_number} with total size: {total_words_size} MB')
print(f'Number of all distinct words: {total_words_distinct_number} with total size: {total_words_distinct_size} MB')

# %%
# definitionm of parameters: probability of false positive (epsilon) and the length of each sentence (word_number)
epsilon = 1e-4
word_number_list = [4, 8]

for word_number in word_number_list:
    print(f'***************Results for {word_number}_words sentences***************')
    sentence_words_set, total_sentences_number, total_sentences_size = get_sentence_from_words(poem_words, word_number, epsilon)
    get_bit_memory_plot(total_sentences_number, word_number)
    print('########################################################################')

# %%



