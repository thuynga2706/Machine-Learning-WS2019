#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument('--output')

args = parser.parse_args()
# In[2]:


data = pd.read_csv(args.data, sep='\t',header = None).dropna(axis=1)


# In[3]:


output =data.iloc[:,0]
output = output.replace("A",1)
output = output.replace("B",0)

inputs =data.iloc[:,1:]
inputs.insert(0, 'bias', 1)


# In[4]:


learningrate = 1


# In[5]:


w = np.zeros(inputs.shape[1])
Misclass1 =[]
for i in range(101):
    prediction = np.dot(w,np.array(inputs).T)
    prediction[prediction>0]= 1
    prediction[prediction<=0]= 0
    error = np.subtract(np.array(output),prediction)
    gradient = np.dot(error,np.array(inputs))
    w = w+learningrate*gradient
    misclassification = error[error!=0].shape[0]
    Misclass1.append(misclassification)


# In[6]:


w = np.zeros(inputs.shape[1])
Misclass2 =[]
for i in range(1,102):
    prediction = np.dot(w,np.array(inputs).T)
    prediction[prediction>0]= 1
    prediction[prediction<=0]= 0
    error = np.subtract(np.array(output),prediction)
    gradient = np.dot(error,np.array(inputs))
    w = w+learningrate/i*gradient
    misclassification = error[error!=0].shape[0]
    Misclass2.append(misclassification)


# In[7]:


outputStr = ''
for i in range(len(Misclass1)): 
    outputStr += str(Misclass1[i]) + "\t"
outputStr+="\n"

for i in range(len(Misclass2)): 
    outputStr += str(Misclass2[i]) + "\t"
    
with open(args.output, "w") as outputter:
    outputter.write(outputStr)


# In[ ]:




