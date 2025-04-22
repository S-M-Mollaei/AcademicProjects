# -*- coding: utf-8 -*-

import csv
import string
import math
#from collections import defaultdict


def tokenize(docs):
    """Compute the tokens for each document.
    Input: a list of strings. Each item is a document to tokenize.
    Output: a list of lists. Each item is a list containing the tokens of the
    relative document.
    """
    tokens = []
    for doc in docs:
        for punct in string.punctuation:
            doc[0] = doc[0].replace(punct, " ")
        split_doc = [ token.lower() for token in doc[0].split(" ") if token ]
        tokens.append(split_doc)
    return tokens


def term_frequncy_TF(semtiment_list):
    sentiment_list_dic = []
    for words in sentiment_list:
        dic = {}
        for word in words:
            if word not in dic:
                dic[word] = 1
            else:
                dic[word] += 1
        sentiment_list_dic.append(dic)
    return sentiment_list_dic

def Df(sentiment_list_dic):
    document_frequency = {}
    for words in sentiment_list_dic:
        for word in words:
            if word not in document_frequency:
                document_frequency[word] = 1
            else:
               document_frequency[word] += 1
    return document_frequency

def IDF(document_frequency, n):
    inverse_document_frequency = {}
    for key, value in document_frequency.items():
        idf = math.log(n/value)
        inverse_document_frequency[key] = idf
    return inverse_document_frequency

def TF_IDF(tf, idf):
    tf_idf = []
    for words in tf:
        dic_temp = {}
        for word, value in words.items():
            weight = value * idf[word]
            dic_temp[word] = weight
        tf_idf.append(dic_temp)
    return tf_idf            
    

with open('imdb.csv') as f:
    whole_docs = []
    flag = True # to aviod getting the first row
    for d in csv.reader(f):
        if flag:
            flag = False
            continue
        whole_docs.append(d)
    
    sentiment_list = tokenize(whole_docs)
    '''computing TF'''
    sentiment_list_dic = term_frequncy_TF(sentiment_list)
    '''computing DF'''
    document_frequency = Df(sentiment_list_dic)
    '''computing IDF'''
    n = len(sentiment_list)
    inverse_document_frequency = IDF(document_frequency, n)
    inverse_document_frequency_sorted = dict(sorted(inverse_document_frequency.items(), key=lambda item: item[1]))
    '''computing TF-IDF using TF*IDF'''
    tf_idf = TF_IDF(sentiment_list_dic, inverse_document_frequency)
    
    
    
    
    
    
    