import csv
import json
import io 
import re



#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import shuffle
from pprint import pprint
# -*- coding: utf-8 -*-
def main():
#01_Reading the Automotive.json file and dividing it to automotive_test.csv and automotive_train.csv
##########################	
	# with open('reviews_Automotive_5.json') as f:
	#     data = json.load(f)

	# reviews = []
	# for line in open('reviews_Automotive_5.json', 'r'):
	#     reviews.append(json.loads(line))

	# print ("number of reviews= " + str(len(reviews)))

	# shuffle(reviews) #To randomly select 2000 reviews
	# # test_data_set= random.sample(reviews,len(reviews)*0.1)
	# test_data_set= reviews[:2000]
	# train_data_set= reviews[2000:]

	# print test_data_set[1]
	# print test_data_set[0]
	# print train_data_set[0]

	# #writing data sets 
	# with io.open('automotive_test.csv', 'wb') as myfile:
	# 	wr = csv.writer(myfile)
	# 	for test_data in test_data_set: 
	# 		wr.writerow([test_data])
	# with io.open('automotive_train.csv', 'wb') as myfile:
	# 	wr = csv.writer(myfile)
	# 	for train_data in train_data_set: 
	# 		wr.writerow([train_data])

##########################

#02_ After finiishing the creation of test and train data sets, we comment out the above part so that we don't over write the datasets
####################
	# #reading data sets
	# with io.open('automotive_test.csv', 'rb') as csvfile:
	# 	data = csv.reader(csvfile)
	# 	test_data_obtained= list(data)

	# print ("length of test data set=" + str(len(test_data_obtained)))
	# # print (test_data_obtained[0])


	# with io.open('automotive_train.csv', 'rb') as csvfile:
	# 	data = csv.reader(csvfile)
	# 	train_data_obtained= list(data)

	# print ("length of training data set=" + str(len(train_data_obtained)))
	# # print (train_data_obtained[0])
	test_data_obtained =[]
	#getting just reviews
	with io.open('automotive_train.csv', 'rb') as csvfile:
		data = csv.reader(csvfile)
		train_data_obtained= list(data)

	# print train_data_obtained[0]
	# first_input= str(train_data_obtained[0])
	# print first_input.find("reviewText")
	# print first_input.find("overall")
	just_reviews=[]
	count=0	
	#### train_set_x contains just the review 
	train_set_x=[]
	#### train_set_y contains the integer value of the rating
	train_set_y=[]
	count_zeros= 0
	count_ones =0
	for inputs1 in train_data_obtained: # train_data_obtained is the list of sentences which contains all info like username, reivews, ratings, etc

		# if count < 5 :
		inputs= str(inputs1) 
		# print (inputs[inputs.find("reviewText")+15: inputs.find("overall")-5] )
		# print ("\n")
		# just_reviews.append(inputs[inputs.find("reviewText")+16 : inputs.find("overall")-5])

		########only_reivew is the variable that contains just the review part. Initially inputs contain other information like Username, date, etc. 
		only_review= inputs[inputs.find("reviewText")+16 : inputs.find("overall")-5]

		#######rating contains just the rating of the reivew. 
		rating= inputs[inputs.find("overall")+10 : inputs.find("summary")-5	]
		# print(int(rating))
		

		##### as there might be some garbage in the rating extracted using substring, getIntValue(int) is a function that cleans rating data. 
		int_val_rating= getIntValue(rating)
		if int_val_rating!= -1 : #  getIntValue() returns -1 for examples where rating extraction was unsuccessful
			# train_set.append([(only_review _x,int_val_rating)])
			train_set_x.append(only_review)
			train_set_y.append(float(int_val_rating))
			# train_set[1].append([int_val_rating])
			if int_val_rating ==0 :
				count_zeros= count_zeros+1
			else:
				count_ones= count_ones +1
			# int_value_of_rating= getIntValue(rating)
				# count= count+1
	print ("count_ones= " + str(count_ones) + " count_zeros= " + str(count_zeros))

	##automotive_train_x and automotive_train_y contains X and Y values of the training dataset. Where X is a the review values
	with io.open('automotive_train_x.csv', 'wb') as myfile:
		wr = csv.writer(myfile)
		for i in train_set_x: 
			wr.writerow([i])

	with io.open('automotive_train_y.csv', 'wb') as myfile:
		wr = csv.writer(myfile)
		for i in train_set_y: 
			wr.writerow([i])
	
## feature vecture is a dictonary that contains all the words in the dataset as keys and values= 0. All the words are converted to lower case
	feature_vec= {}

### generate corpus creates feature_vec dictionary 
	feature_vec= geenerate_corpus(train_set_x)
	empty_feature_vector= feature_vec.copy()



	# train_set_x=train_set_x[1:100]
	### we create the dictipnary of each example  
	for y in range(0,len(train_set_x)):
		print y
		#### return_dict_of_sentence is a function that takes a sentence, and the corpus dictionary as input and returns an updated dictionary with count of present words updated 
		dict_of_sent= return_dict_of_sentence(train_set_x[y], empty_feature_vector)


		###### feature_vec is updated using feature vector of sentence. (Srishti's commnent- there might be some mistake here) 
		for values in feature_vec.keys():
			feature_vec[values]= dict_of_sent[values]* train_set_y[y] +feature_vec[values]

	print feature_vec



####03_ running the test data set through feature vector calculated above
###################


	# with io.open('samples_picked_by_Natalie.csv', 'rb') as csvfile:
	# 	data = csv.reader(csvfile)
	# 	list_samples_by_Natalie= list(data)

	
	list_samples_by_Natalie=["I love it - have it rubber corded to the stand via a drilled hole- it just plain works!", 
							"Fit perfectly, like factory equipment! Looks like it came on teh Jeep right off of the lot. Kids grab them all the time getting in and out. Well made and sturdy!", 
							"The bulb is kind of pricey but it is the brightest by how much? not as bright as the sun but when it comes to being seen on a bike...well what's your rear worth?",
							"I followed the instructions, cleaned the tire with a bristle brush and soap...dried it with towels. Then I applied the product. This morning, my tires have a thick, plastic feeling brown residue all over...looks like rust mixed with mud. When I used this stuff before, it never left a brown residue, just a horrible attempt at a shine.Overall this product sucks, and now I need to figure out how to remove the brown reside it left. Sticking with McGuires from now on..."]
	# print list_samples_by_Natalie
	# for sentences in list_samples_by_Natalie:
	# 	dict_of_sentence_picked= return_dict_of_sentence(sentences, empty_feature_vector)
	# 	print dict_of_sentence_picked
	# 	dot_product= 0
	# 	for i in empty_feature_vector.keys():
	# 		dot_product= dot_product+ feature_vec[i] *dict_of_sentence_picked[i]*train_set_y[i]

	for z in range(0,len(list_samples_by_Natalie)):
		dict_of_sentence_picked= return_dict_of_sentence(list_samples_by_Natalie[z], empty_feature_vector)
		# print dict_of_sentence_picked
		dot_product= 0.0
		for i in empty_feature_vector.keys():
			# if feature_vec[i] > 0 :
			# 	print 1
			dot_product= dot_product+ feature_vec[i] *dict_of_sentence_picked[i]*train_set_y[z]

		print (str(train_set_x[z])+ "dot_product= " +str(dot_product))


#### 04_ functions used in the code above
#####################

def getIntValue(str):
	flag= -2
	if str== '2.0' or str=="2." or str== "2" or str== " 2.0" or str== '1.0' or str=="1." or str== "1"  or str == " 1.0":
		flag =0
		

	if str== '4.0' or str=="4." or str== "4" or str==" 4.0" or str== '5.0' or str=="5." or str== "5" or str == " 5.0":
		flag =1
	
	# if flag !=0 and flag != 1:
		# print str

	return flag



def geenerate_corpus(train_data_list):
	all_words= set()
	feature_vec1={}
	for k in range(0,len(train_data_list)):
		wordList = re.sub("[^\w]", " ",  train_data_list[k]).split()
		for words in wordList:
			all_words.add(words.lower())
	# print all_words
	for j in all_words:
		feature_vec1[j]= 0 

	# print len(all_words)
	# print feature_vec
	return feature_vec1




def return_dict_of_sentence(string_passed, empty_feature_vec):

	wordList = re.sub("[^\w]", " ",  string_passed).split()
	for words in wordList:
		empty_feature_vec[words.lower()] +=1
	# print empty_feature_vec
	return empty_feature_vec




if __name__== "__main__":
	main()

