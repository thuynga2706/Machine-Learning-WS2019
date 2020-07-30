#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import math
import xml.etree.cElementTree as ET
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument('--output')
args = parser.parse_args()
# In[2]:


def data_preprocess(file):

    # Read input data.
    data = pd.read_csv(file, header=None)
    # Add new header for each columns (att0 ...attn).
    # The header for the last column is label, which is the given prediction.
    len_columns = len(data.columns.values)
    columns_list = data.columns.tolist()
    for i in range(len_columns - 1):
        columns_list[i] = "att" + str(i)
    columns_list[len_columns - 1] = "label"
    data.columns = columns_list
    return data


# In[123]:


data= data_preprocess(args.data)
entropybase = data.label.nunique()


# In[4]:


def entropy(dataset):
    labelvalues = dataset['label'].value_counts()
    if len(dataset['label'].unique()) == 1:# if an attibute column only have one corresponding label value, it's already pure
        Entropy = 0.0
    else:
        Entropy = 0
        count = dataset.shape[0]
        for i in labelvalues:
            Pi = i/count
            Entropy += (-Pi)*math.log(Pi,entropybase)
    return Entropy


# In[18]:


def partition(dataset,attibute): 
    #input : data (table) - output(p of each value in attribute)
    partitioneachcol = dataset[attibute].value_counts()/dataset.shape[0]
    return partitioneachcol


# In[20]:


def partitiondata(dataset,attribute):
    uniquevalue = dataset[attribute].unique()
    partitiondata={}
    for i in uniquevalue:
        smalldata = dataset[dataset[attribute] == i]
        partitiondata[i] = smalldata
    return partitiondata
Partitiondata = partitiondata(data,'att5')


# In[7]:


def entropycolumn(dataset,attribute):
    Partition = partition(dataset,attribute)
    Partitiondata = partitiondata(dataset,attribute)
    Entropy = {}
    for i in range(len(Partition.index)):
        individualentropy = entropy(Partitiondata[Partition.index[i]])
        Entropy[Partition.index[i]] = individualentropy
    return Entropy


# In[8]:


def informationgain(dataset,column):
    Partition = partition(dataset,column)
    Partitiondata = partitiondata(dataset,column)
    Entropycolumn = entropycolumn(dataset,column)
    Entropysplit = 0
    for key in Entropycolumn:
        Entropysplit+=Entropycolumn[key]*Partition[key]
    Informationgain = entropy(data) - Entropysplit
    return Entropycolumn,Informationgain


# In[9]:


def attributetosplit(data):
    Entropy = entropy(data)
    columns = data.columns[:-1]
    bestIG = 0
    bestcolumn = None
    Entropycolumnbest = None
    for i in columns:
        Entropycolumn,Informationgain = informationgain(data,i)
        if Informationgain>bestIG:
            bestIG = Informationgain
            bestcolumn = i
            Entropycolumnbest = Entropycolumn
    return Entropy,bestcolumn,Entropycolumnbest


# In[120]:


i=0
def recursive(data):
    global i
    current_xml = ET.Element('node')
    Entropy,bestcolumn,Entropycolumnbest = attributetosplit(data)
    
    if Entropy != None:
        current_xml.set('entropy ', str(Entropy))
    if Entropy == 0:
        current_xml.text = data.label.iloc[0]
            
    global feature_parent
    if feature_parent != None:
        current_xml.set('feature', str(feature_parent))   
    if bestcolumn != None and i!= 0:
        current_xml.set('value', str(i))
    
    try:      
            
        if Entropy >0:       
            data1 = partitiondata(data,bestcolumn)
            for i in data1.keys():     
                feature_parent = bestcolumn                
                current_xml.append(recursive(data1[i]))            
    except:
        pass
    return current_xml


# In[124]:


feature_parent = None
xml = recursive(data)


# In[125]:


tree_string = ET.tostring(xml).decode("UTF-8")
tree_file = open(args.output, "w")
tree_file.write(tree_string)
print("Hi! Your file is ready.")

# In[ ]:




