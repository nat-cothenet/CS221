#import tensorflow as tf
import collections
import string
import re
import nltk
import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import gzip

##########################
# TODO:
# Update read-in to Books data subset (full dataset won't upload)
# Review and comment code
# Test and debug TF-IDF, especially looking for ways to speed it up
##########################


###########################
# Reading in dataset
###########################

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

data=[]
path= 'reviews_Health_and_Personal_Care_5.json.gz'

for line in parse(path):
    data.append(line)

dict_ratings = collections.defaultdict(int)

X=[] #reviews
Y=[] #ratings
asin_dict = collections.defaultdict(list)
asin_dict_pos = collections.defaultdict(list)
asin_dict_neg = collections.defaultdict(list)

for i in data:
    X.append(i['reviewText'])
    Y.append(i['overall'])
    dict_ratings[i['overall']] += 1
    asin_dict[i['asin']].append(i['reviewText'])
    if i['overall'] >= 4:
        asin_dict_pos[i['asin']].append(i['reviewText'])
    else:
        asin_dict_neg[i['asin']].append(i['reviewText'])

num_words = len((" ".join(X)).split())

print "dictionary of # of reviews per rating"
print dict_ratings
print "# of reviews/# of ASIN"
print float(len(X))/float(len(asin_dict.keys()))
print "median of reviews/ASIN"
print np.median([len(i) for i in asin_dict.values()])
print "# of words/# of reviews"
print float(num_words)/float(len(X))
