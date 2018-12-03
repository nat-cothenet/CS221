import csv
import json
import io
import re
import array
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import random
from pprint import pprint
import collections
from scipy.stats import norm
import gzip
import math
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def ci_lower_bound(pos, n, confidence):
    if n == 0:
        return 0
    z = norm.ppf(1-(1-confidence)/2)
    phat = 1.0*pos/n
    return (phat + z*z/(2*n) - z * math.sqrt((phat*(1-phat)+z*z/(4*n))/n))/(1+z*z/n)

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

def main():
    reviews = []
    number_of_data_points_to_read = 10000 #Select number of data points to read
    counter= 1
    path= 'reviews_Health_and_Personal_Care_5.json.gz'

    for line in parse(path):
        if counter <number_of_data_points_to_read:
            reviews.append(line)
            counter += 1
        else:
            break

	helpful_dict = collections.defaultdict(list)
    review_dict = collections.defaultdict(str)

    print "done with file"
    id = 1
    for i in reviews:
        if i['helpful'][1] != 0:
                helpful_dict[id] = i['helpful']
                review_dict[id] = i['reviewText']
                id += 1
    print id
    print "read in done"
	##### Split into testing and training sets #####
    random.seed(42)

    train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(review_dict.values(), [ci_lower_bound(i[0], i[1], 0.95) for i in helpful_dict.values()], test_size=0.2, random_state=None, shuffle= True) #split into train and test set

    print "on to the corpus!"

	##### Create corpus, convert reviews to feature vectors #####

    if False: # Change to False once corpus generated
		corpus = generate_corpus(train_set_x)
		with io.open('hpc_corpus.csv', 'wb') as myfile:
			wr = csv.writer(myfile)
			for i in corpus:
				wr.writerow([i])
    else:
		with io.open('hpc_corpus.csv', 'rb') as csvfile:
			data = csv.reader(csvfile)
			corpus = list(data)
			corpus = [corpus[i][0] for i in range(len(corpus))]

    print "Corpus generated"
    corpus = {c: 0 for c in corpus}

	### Create the dictionary of each example
    features = []
    for xi in range(0,len(train_set_x)):
		#### return_dict_of_sentence is a function that takes a sentence, and the corpus dictionary as input and returns an updated dictionary with count of present words updated
		features.append(return_dict_of_sentence(train_set_x[xi], corpus))

    print len(features)
    print len(train_set_y)


    clf = MLPRegressor()
    clf.fit(X = features, y = train_set_y)
    score = clf.score(X = features, y = train_set_y)

    # Calculate the proportion correctly classified for training set
    print "Training error"
    meanSquaredError=mean_squared_error(train_set_y, clf.predict(features))
    print("MSE:", meanSquaredError)
    rootMeanSquaredError = math.sqrt(meanSquaredError)
    print("RMSE:", rootMeanSquaredError)

	###################
	# 03_Running the test data set through feature vector calculated above
	###################

	##### Extract review and rating #####

	### Create the dictionary of each example
    test_features = []
    for xi in range(0,len(test_set_x)):
		#### return_dict_of_sentence is a function that takes a sentence, and the corpus dictionary as input and returns an updated dictionary with count of present words updated
		test_features.append(return_dict_of_sentence(test_set_x[xi], corpus))

    #predictions = logit.predict(test_features)
    predictions = clf.predict(test_features)


	# Calculate the proportion correctly classified for testing set
    print "Testing error"
    meanSquaredError=mean_squared_error(test_set_y, predictions)
    print("MSE:", meanSquaredError)
    rootMeanSquaredError = math.sqrt(meanSquaredError)
    print("RMSE:", rootMeanSquaredError)


#####################
# 04_Functions used in the code above
#####################


# Returns 1 if the review is positive, -1 if the review is negative, and None for other values
def get_y(str):
	if "2" in str or "1" in str:
		return -1
	elif "4" in str or "5" in str:
		return 1
	else:
		return None
"""

def get_y(str):
	if "1" in str:
		return 1
	elif "2" in str:
		return 2
	elif "3" in str:
		return 3
	elif "4" in str:
		return 4
	elif "5" in str:
		return 5
	else:
		return None
"""

# Returns a dictionary whose keys are the full corpus of words in the training data set, and whose assigned values are all 0
def generate_corpus(train_data_list):
	all_words = set()
	feature_vec = {}

	for k in range(0,len(train_data_list)):
		wordList = re.sub("[^\w]", " ",  train_data_list[k]).split() # Clean and split data
		for words in wordList:
			all_words.add(words.lower())

	return list(all_words)
	#feature_vec = {key: 0 for key in all_words}

	#for j in all_words:
	#	feature_vec[j] = 0

	#return feature_vec

"""
# Returns a dictionary whose keys are the full corpus of words in the training data set, and whose assigned values are an arbitrary index
def generate_index(train_data_list):
	all_words = set()
	feature_vec = {}

	for k in range(0,len(train_data_list)):
		wordList = re.sub("[^\w]", " ",  train_data_list[k]).split() # Clean and split data
		for words in wordList:
			all_words.add(words.lower())

	index = 1
	for j in all_words:
		feature_vec[j] = index
		index += 1

	return feature_vec
"""

# For a string passed in, returns an ordered list of frequencies associated with each word in the corpus
def return_dict_of_sentence(string_passed, corpus):
	#feature_vec = collections.defaultdict(int)
	feature_vec = corpus
	wordList = re.sub("[^\w]", " ",  string_passed).split() # Clean and split data
	for words in wordList:
		if words in corpus.keys():
			feature_vec[words.lower()] += 1

	#return feature_vec
	#return np.array([feature_vec.keys(), feature_vec.values()])
	return feature_vec.values()

"""
# For a string passed in, returns a dictionary whose keys are the indices of the words in the sentence as they are assigned in the vocab_dict, with the value indicating the frequency of that word
def return_sentence_list(string_passed, vocab_dict):
	wordList = re.sub("[^\w]", " ",  string_passed).split() # Clean and split data
	sentence_vec = array.array('i', [0,]*len(vocab_dict))
	for words in wordList:
		 sentence_vec[vocab_dict[words.lower()]] += 1

	return sentence_vec
"""



if __name__== "__main__":
	main()
