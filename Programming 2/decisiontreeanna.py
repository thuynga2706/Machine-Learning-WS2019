#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 15:49:54 2018

"""

import sys
import pandas as pd
import numpy as np
import math
import xml.etree.cElementTree as ET
import argparse


def data_preprocess(file):
    """
    :return: data: input data as a pandans dataFrame
            columns_list: a list of added headers 
    """
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
    return data, columns_list


def get_freqs(data, attrn):
    """
    :param: data: a pandas data frame
          : attrn: a selected feature from feature_list

    :return: freqs: a list which contains the frequency of all unique
                    category values of one column
    """
    freqs = data[attrn].value_counts().values.tolist()
    return freqs


# Return data set of child nodes which is splitted by a selected feature
def get_data_children(data_parent, attrn):

    # Store unique values
    values_unique = data_parent[attrn].unique()

    # Initialize data children list
    data_children = []

    for value in values_unique:
        # filter dataframe
        data_parent_filtered = data_parent[data_parent[attrn] == value]
        # drop feature column
        data_parent_filtered = data_parent_filtered.drop(columns=[attrn])
        # add dataframe to list,
        # data_children.append(data_parent_filtered)
        data_children.append(
            {
                "feature": attrn,
                "value" : value,
                "data": data_parent_filtered,
            }
        )

    return data_children


def get_values(data, attrn):
    """
    :param: data: a pandas data frame
          : attrn: a selected feature from feature_list

    :return: values: a list which contains all unique category values of one column
    """
    values = data[attrn].unique()
    print('values: {}'.format(values))
    return values

# Main identification tree class
# There will be one instance of the class per node

class IdenTree:
    # The prediction of the node with 0 entropy
    # Update by SetLabel(self)
    label = None
    # The dataframe for the node itself
    data_parent = None
    # The entropy of the node;
    # Update by Entropy(self)
    entropy = None
    # Selected feature for splitting in the next step;
    # Update by ChooseAndSplit(self)
    feature = None
    # Feature of parent node by which the tree got splitted (and the value belongs to)
    feature_parent = None
    # Level of the node in the whole tree;
    # Update by ChooseAndSplit(self)
    level = 0
    # A list of the data frame for all children nodes in next step (if have)
    # Update by ChooseAndSplit(self)
    data_children = None
    # The list of features which haven't been selected yet
    # Update by ChooseAndSplit(self)
    features_list = None
    # The list of all used features
    # Update by ChooseAndSplit(self)
    features_used = None
    # The corresponding feature value of the node
    # Updated by SetFeatureValue(self)
    feature_value = None

    def __init__(self, data_parent, feature_value, feature_parent):
        # The initialization of the instance needs the corresponding data frame for the node
        self.data_parent = data_parent.copy()
        self.features_list = data_parent.copy().columns[:-1].values.tolist()
        self.entropy = self.Entropy(self.data_parent)
        self.feature_parent = feature_parent
        self.features_used = []
        self.feature_value = feature_value
        self.ChooseAndSplit()
        self.SetLabel()

    def Entropy(self, data):
        """
        :param: data: a pandas data frame
        :return: entropy of the given data frame
        """
        if len(data['label'].unique()) == 1:
            # if an attibute column only have one corresponding label value, it's already pure
            entropy_temp = 0.0
        else:
            entropy_temp = 0.0
            freqs = get_freqs(data, 'label')
            sum_freqs = sum(freqs)
            for freq in freqs:
                prob = freq / sum_freqs
                entropy_temp += ((-prob) * math.log(prob, entropy_base))
        return entropy_temp

    def InfoGain(self, attrn):
        """
        :param: attrn: a feature in the feature_list
        :return: infromation gain if data is splitted by feature attrn
        """
        # Get children data set if data_parent is spliited by attrn
        children_temp = get_data_children(self.data_parent, attrn)
        #        print(children_temp)
        # freq_per_child is a list count the frequency of each unique feature value in a feature colum
        freq_per_child = get_freqs(self.data_parent, attrn)
        #        print(freq_per_child)
        sum_freq = sum(freq_per_child)
        #        print(sum_freq)
        freq_per_child = np.asarray(freq_per_child)
        #        print(freq_per_child)
        #        print(type(freq_per_child))
        weight_per_child = freq_per_child / sum_freq
        #        print(weight_per_child)
        #        print(type(weight_per_child))
        entropy_children = []
        for child in children_temp:
            entropy_children.append(self.Entropy(child['data']))
        #        print(entropy_children)
        EA = sum(weight_per_child * entropy_children)
        #        print(EA)
        gain = self.entropy - EA
        return gain

    def ChooseAndSplit(self):
        # Choose the feature with the highest info_gain
        # Update attibutes(feature, features_used,features_list,data_children) of the node
        if len(self.features_list) != 0:
            best_gain = float('-Inf')

            # print('\n')
            # print('level: {}'.format(self.level))
            # print('start checking for the best feature in dataset')
            #            print(best_gain)
            for feature in self.features_list:

                # print('----- check feature {}'.format(feature))
                #                print(feature)
                gain = self.InfoGain(feature)

                # print('Gain: {}'.format(gain))
                #                print(gain)
                if gain > best_gain:
                    # print('update best gain')
                    best_gain = gain
                    self.feature = feature
                    # print('write feature: {}'.format(feature))
        #        print(best_gain)

            # print('Chosen feature: {}'.format(self.feature))

            self.features_used.append(self.feature)
            self.features_list.remove(self.feature)
            self.data_children = get_data_children(self.data_parent,
                                                   self.feature)
            self.level = self.level+1

        return

    def SetLabel(self):
        # Update the label if the entropy of the node is 0
        # print('check entropy: {}'.format(self.entropy))
        if self.entropy == 0:
            # print(self.data_parent['label'])
            self.label = self.data_parent['label'].iloc[0]
            # print('stored label: {}'.format(self.label))
            self.data_children = None
        else:
            self.label = None



def iterate_tree (current_data, feature_value, feature_parent):

    # Init tree
    tree = IdenTree(current_data, feature_value, feature_parent)

    # Store entropy, feature(e.g color), feature_value(e.g blue / green / red)
    current_xml = ET.Element('node')

    if tree.entropy != None:
        current_xml.set('entropy ', str(tree.entropy))
    if feature_parent != None:
        current_xml.set('feature', str(feature_parent))
    if tree.feature_value != None:
        current_xml.set('value', str(tree.feature_value))

    # Check if its a leaf
    if tree.label == None:
        for child_dataset in tree.data_children:
            current_xml.append(iterate_tree(child_dataset['data'], child_dataset['value'], child_dataset['feature']))
    else:
        current_xml.text = str(tree.label)

    return current_xml


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Enter filename to the data csv file")
    parser.add_argument("--output", help="Enter filename of the output xml")
    args = parser.parse_args()

    print('Read data....')
    # data: the dataFrame of the root node
    # attrn_list: A list features of the root node
    # Use later for initialization of idenTree() class
    data, attrn_list = data_preprocess(args.data)
    del attrn_list[-1]

    print('Calculate entropy....')
    # Get the log base for entropy cauculation later
    entropy_base = len(data['label'].unique())

    print('Build tree....')
    # tree_xml = ET.Element("tree")
    tree_xml = iterate_tree(data.copy(), None, None)

    tree_string = ET.tostring(tree_xml).decode("UTF-8")
    tree_file = open(args.output, "w")
    tree_file.write(tree_string)

    print('Your xml file is ready. Cheers! 🍸')








