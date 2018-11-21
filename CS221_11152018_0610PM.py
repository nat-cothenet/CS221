
# coding: utf-8

# In[51]:


import tensorflow as tf
import pandas as pd
import json
import numpy as np


# In[32]:


# df=pd.read_csv('/Users/srishti/Google Drive/000_7th Quarter/CS221/Project/Automotive_Dataset/automotive_test.csv',header=None)
# d = df.values
    
path= '/Users/srishti/Google Drive/000_7th Quarter/CS221/Project/untouched_data/reviews_Automotive_5.json'
data = []
with open(path) as f:
    for line in f:
        data.append(json.loads(line))

# d[1]
# for j in data[1]:
#     print j
# print data[1]['reviewText']

X=[] #reivews
Y=[] #ratings

for i in data:
    X.append(i['reviewText'])
    Y.append(i['overall'])

print X[1]
print Y[1]


# In[49]:





# In[57]:


embeddings_index = dict()
f = open('/Users/srishti/Google Drive/000_7th Quarter/CS221/Project/untouched_data/glove.6B/glove.6B.50d.txt')
for line in f:# first element of each line is the word and remaining elements are numerical represenation of each line 
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs# creates a dict of words and numerical representation of that word
f.close()


# In[58]:


# embeddings_index['goat']


# In[66]:


embeddings_index['dog']


# In[81]:


dataEmbeddings=[]

for i in X:
    list_one_sentence= []
    for j in i.split():
        print j
        if j in embeddings_index.keys():
            list_one_sentence.append(embeddings_index[j]) 

    dataEmbeddings.append(list_one_sentence)
    
    
        


# In[ ]:


# random.shuffle(X)
#shuffle
from sklearn.model_selection import train_test_split
##X train, Y_train is training dataset; X_test and Y_test is test dataset
#seed it if we want a particular sample
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=None, shuffle= True)#split into train and test set

