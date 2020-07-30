#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import argparse
import os 
parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument('--output')

args = parser.parse_args()

# In[9]:


data = pd.read_csv(args.data, sep='\t',header = None).dropna(axis=1)
data = data.iloc[:,1:]


# In[10]:


c1=[0,5]
c2=[0,4]
c3=[0,3]


# In[11]:


def kmeans(c1,c2,c3):
    distance = pd.DataFrame()
    distance['c1'] = (data - c1).pow(2).sum(axis=1)
    distance['c2'] = (data - c2).pow(2).sum(axis=1)
    distance['c3'] = (data - c3).pow(2).sum(axis=1)

    distance["min_value"] = distance.min(axis=1)
    distance["centroid"] = distance.idxmin(axis=1)

    c1= data[data.index.isin(distance[distance.centroid == "c1"].index)].mean().to_list()
    c2= data[data.index.isin(distance[distance.centroid == "c2"].index)].mean().to_list()
    c3= data[data.index.isin(distance[distance.centroid == "c3"].index)].mean().to_list()
    error = distance.min_value.sum()
    return c1,c2,c3,error


# In[12]:


Error=''
Centroid =','.join(str(e) for e in c1) + "\t" +','.join(str(e) for e in c2) + "\t" + ','.join(str(e) for e in c3) + "\n" 
olderror = 0
c1,c2,c3,error = kmeans(c1,c2,c3)
while error!=olderror :
    Error += str(error) + "\n"
    olderror=error
    if all((c1,c2,c3,error != kmeans(c1,c2,c3))[3]):
        Centroid +=','.join(str(e) for e in c1) + "\t"+ ','.join(str(e) for e in c2) + "\t" + ','.join(str(e) for e in c3) + "\n"
    c1,c2,c3,error = kmeans(c1,c2,c3) 


# In[13]:
path = args.output

with open(os.path.join(path,args.data[:-4]+'-Progr.tsv'), "w") as outputter:
    outputter.write(Error)


# In[14]:

with open(os.path.join(path,args.data[:-4] +'-Proto.tsv'), "w") as outputter:
    outputter.write(Centroid)
 


# In[ ]:




