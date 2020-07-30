#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import argparse
import boto3
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--output')
    
    args = parser.parse_args()
    
    
    s3 = boto3.client('s3')
    s3.download_file('nga-example4', 'Example.tsv', 'Example-s3.tsv')
    # In[15]:
    
        
    data = pd.read_csv(args.data, sep='\t',header = None).dropna(axis=1)
    
    
    # In[16]:
    
    
    output =data.iloc[:,0]
    inputs =data.iloc[:,1:]
    
    data_A = data[data[0]=="A"]
    data_B = data[data[0]=="B"]
    
    inputsA =data_A.iloc[:,1:]
    inputsB =data_B.iloc[:,1:]
    
    
    # In[17]:
    
    
    ProbA = len(inputsA)/len(data)
    ProbB = len(inputsB)/len(data)


# In[18]:


    def getmeanandvariance(data):
        mean = data.sum()/len(data)
        variance = np.sum(np.square(data - mean))/(len(data)-1)
        return mean,variance
    
    
    # In[19]:
    
    
    def getcontinuousprob(data,meanvariancedata):
        #data: array of x
        #meanvariancedata: data to get mean and variance
        #return probability of x given distribution of meanvariancedata
        mean,variance = getmeanandvariance(meanvariancedata)
        array = np.exp(-np.square(data-mean)/(2*variance))/np.sqrt(2*np.pi*variance)
        return array
    
    
    # In[20]:
    
    
    inputs["A"] = ProbA
    inputs["B"] = ProbB
    for i in range(inputsA.shape[1]):
        inputs["A"] = inputs["A"]*getcontinuousprob(inputs.iloc[:,i],inputsA.iloc[:,i])
        inputs["B"] = inputs["B"]*getcontinuousprob(inputs.iloc[:,i],inputsB.iloc[:,i])
    
    
    # In[21]:
    
    
    inputs.loc[inputs["A"]>=inputs["B"],"prediction"] = "A"
    inputs.loc[inputs["A"]<inputs["B"],"prediction"] = "B"
    
    
    # In[22]:
    
    
    meanA, variA = getmeanandvariance(inputsA)
    meanB, variB = getmeanandvariance(inputsB)
    
    
    # In[23]:
    
    
    outputStr=''
    for i in range(inputsA.shape[1]):
        outputStr += str(meanA.to_list()[i]) + "\t"
        outputStr += str(variA.to_list()[i]) + "\t"
    outputStr += str(ProbA)
    outputStr+="\n"
    
    
    # In[24]:
    
    
    for i in range(inputsA.shape[1]):
        outputStr += str(meanB.to_list()[i]) + "\t"
        outputStr += str(variB.to_list()[i]) + "\t"
    outputStr += str(ProbB)
    outputStr+="\n"
    
    
    # In[12]:
    
    
    outputStr += str((inputs.prediction == output).value_counts().loc[False])
    
    
    # In[13]:
    
    
    with open(args.output, "w") as outputter:
        outputter.write(outputStr)

if __name__== "__main__":
    main()