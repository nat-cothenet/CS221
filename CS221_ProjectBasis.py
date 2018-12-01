##Srishti edited this on 11/30 to allow the code to read book data. Logic: Instead of reading the entire file just read the number of data points we need 


import tensorflow as tf
import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize 
from nltk.stem import WordNetLemmatizer 

##########################
# TODO:
# Update read-in to Books data subset (full dataset won't upload)
# Review and comment code
# Test and debug TF-IDF, especially looking for ways to speed it up
##########################


###########################
# Reading in dataset
###########################
    
# path = '/Users/srishti/Google Drive/000_7th Quarter/CS221/Project/untouched_data/reviews_Automotive_5.json'
# data = []
# with open(path) as f:
#     for line in f:
#         data.append(json.loads(line))


number_of_data_points_to_read= 100000 #Select number of data points to read
counter= 1
data=[]
path= '/Users/srishti/Desktop/reviews_Books_5.json'
with open(path) as f:
    for line in f:
        if counter <number_of_data_points_to_read:
            data.append(json.loads(line))
            counter+=1
        else: 
            break

# d[1]
# for j in data[1]:
#     print j
# print data[1]['reviewText']

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

# Returns a sorted list of tuples containing (word, weight)
def TF_IDF(asin, asin_dict, n):
	vectorizer = TfidfVectorizer(max_df = 0.90,
				     min_df = int(1),
				     tokenizer = lambda str: {wl = WordNetLemmatizer()
							      return [wl.lemmatize(t) for t in word_tokenize(str)]
							     },
				     strip_accents = ‘ascii’,
				     use_idf = True,
				     smooth_idf = True)
	dict_counts = vectorizer.fit_transform(asin_dict)
	TF_IDF = vectorizer.transform(asin_dict[asin]).toarray()
	
	word_map=vectorizer.get_feature_names()
	vec_pairs = [(word_map[j], TF_IDF[j]) for j in range(len(TF_IDF))]
	
	vec_pairs.sort(key = lambda word: word[1]), reverse = True)
	
	return vec_pairs
    
	
"""
# Returns a sorted list of tuples containing (word, weight)
def TF_IDF(asin, corpus, asin_dict):
    all_words_asin = defaultdict(int)
    total_word_num = 0
    my_X = asin_dict[asin]
    
	for k in range(len(my_X)):
		wordList = re.sub("[^\w]", " ",  my_X[k]).split() # Clean and split data
		for words in wordList:
			all_words_asin[words.lower()] += 1
            total_word_num += 1
    
    TF_IDF = list()
    num_docs = len(asin_dict)
    for k in all_words_asin.keys():
        TF = all_words_asin[k]/total_word_num
        num_docs_containing_k = 0
        for j in asin_dict.keys():
            if k in " ".join(asin_dict[j]):
                num_docs_containing_k += 1
        IDF = log(num_docs/num_docs_containing_k)
        TF_IDF.append((k, TF/IDF))
    
    TF_IDF.sort(key = lambda word: word[1]), reverse = True)
    return TF_IDF
"""
