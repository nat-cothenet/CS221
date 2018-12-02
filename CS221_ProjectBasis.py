import collections
import string
import re
import nltk
import gzip
import tensorflow as tf
import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize 
from nltk.stem import WordNetLemmatizer 

###########################
# Reading in dataset
###########################
    
# path = '/Users/srishti/Google Drive/000_7th Quarter/CS221/Project/untouched_data/reviews_Automotive_5.json'
# data = []
# with open(path) as f:
#     for line in f:
#         data.append(json.loads(line))


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

number_of_data_points_to_read = 100000 #Select number of data points to read
counter= 1
data=[]
path= 'reviews_Books_5.json.gz'

for line in parse(path):
    if counter <number_of_data_points_to_read:
        data.append(line)
        counter += 1
    else:
        break

num_titles_to_read = 100000
meta_data = []
meta_path= 'meta_Books.json.gz'
counter= 1

for line in parse(meta_path):
    if counter < num_titles_to_read:
        meta_data.append(line)
        counter += 1
    else:
        break
asin_title_map = collections.defaultdict(list)

for i in meta_data:
    if 'title' in i.keys():
        asin_title_map[i['asin']] = i['title']
    else:
        asin_title_map[i['asin']] = None

X=[] #reviews
Y=[] #ratings
asin_dict = defaultdict(list)

# Assigns X as a list of reviews, Y as a list of star ratings, and asin_dict as a dictionary with keys asin_pos and asin_neg
# where asin_pos contains all reviews associated with that asin with a star-rating of 4 or 5, and asin_neg has all reviews
# for that asin with a star rating of 1, 2, or 3
for i in data:
    X.append(i['reviewText'])
    Y.append(i['overall'])
    if i['overall'] >= 4:
	asin_dict[i['asin']+"_pos"].append(i['reviewText'])
    else:
	asin_dict[i['asin']+"_neg"].append(i['reviewText'])

# Merges the list of reviews for each asin_pos and asin_neg into a single string
for j in asin_dict.keys():
	asin_dict[j] = " ".join(asin_dict[j])
	asin_dict[j] = re.sub(ur"\p{P}+", " ", asin_dict[j])
	asin_dict[j] = re.sub(ur'[^\w\s]',' ',asin_dict[j])

print X[1]
print Y[1]


###########################
# Creates a numerical representation of words in each review
###########################

embeddings_index = dict()
f = open('/Users/srishti/Google Drive/000_7th Quarter/CS221/Project/untouched_data/glove.6B/glove.6B.50d.txt')
for line in f:# first element of each line is the word and remaining elements are numerical represenation of each line 
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs # creates a dict of words and numerical representation of that word
f.close()

embeddings_index['dog']


dataEmbeddings=[]

for i in X:
    list_one_sentence= []
    for j in i.split():
        print j
        if j in embeddings_index.keys():
            list_one_sentence.append(embeddings_index[j]) 

    dataEmbeddings.append(list_one_sentence)
    
###########################
# Split into training and test datasets
###########################

#seed it if we want a particular sample

#X train, Y_train is training dataset; X_test and Y_test is test dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=None, shuffle= True) #split into train and test set


###########################
# Helper functions
###########################

# Generate frequency dictionary of all words in all reviews (needed for TF-IDF)
def generateCorpus(X):
    all_words = defaultdict(int)
	
	for k in range(0,len(X)):
		wordList = re.sub("[^\w]", " ",  X[k]).split() # Clean and split data
		for words in wordList:
			all_words[words.lower()] += 1

	return all_words

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t, 'n') for t in word_tokenize(doc)]

# Returns a sorted list of tuples containing (word, weight)
def TF_IDF(asin, asin_dict, n):

    vectorizer = TfidfVectorizer(stop_words="english", analyzer='word', lowercase = True, tokenizer = LemmaTokenizer(), ngram_range=(1, 1), max_df=0.90)

    vectorizer.fit_transform(list(asin_dict.values()))
    tf = vectorizer.transform([asin_dict[asin]])
    word_map=vectorizer.get_feature_names()


    keywords = []
    for col in tf.nonzero()[1]:
        #my_str = str(word_map[col]) + ' - ' + str(tf[0, col])
        keywords.append((word_map[col],tf[0, col]))

    sorted_keywords = sorted(keywords, key=lambda t: t[1] * -1)

    return sorted_keywords[:n]
    

###########################
# Run TF-IDF
###########################

test_list = [2, 5, 6, 7, 9, 11, 12, 13]

for i in test_list:
    my_asin = asin_dict.keys()[i-1]
    print my_asin
    print asin_title_map[my_asin]
    #print asin_dict[my_asin]

    keywords = TF_IDF(my_asin, asin_dict, 5)
    print keywords
