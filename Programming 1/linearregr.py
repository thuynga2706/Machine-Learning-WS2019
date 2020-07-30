#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import argparse
# In[2]:
#Set argument:
parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument('--learningRate')
parser.add_argument('--threshold')
args = parser.parse_args()


data = pd.read_csv(args.data,header=None)
learningrate = float(args.learningRate)
threshold = float(args.threshold)

#random = pd.read_csv('random.csv',header=None)
#data = pd.read_csv(sys.argv[1],header=None)

inputs = data.iloc[:,:-1] #Training samples
inputs.insert(0, 'bias', 1) #Add 1 column(bias) with value 1
output=data.iloc[:,-1] #Labels 


# In[4]:

result =[]
w = np.zeros(inputs.shape[1]) #Initialize the weights with 0, length of vector = number of columns in training samples
SSE=10
SSEold=200000
changeinSSE=1
while changeinSSE>threshold:
    prediction = np.dot(w.T,np.array(inputs).T)
    error = np.subtract(np.array(output),prediction)
    gradient = np.dot(error,np.array(inputs))
    SSE = np.sum(np.square(error)) ##SSEnew
    changeinSSE = SSEold-SSE
    result.append(np.append(w,SSE)) #add info about the weights and SSE to result list
    w = w+learningrate*gradient.T #Update the weights
    SSEold=SSE



# In[5]:


print(pd.DataFrame(result))


# In[ ]:




