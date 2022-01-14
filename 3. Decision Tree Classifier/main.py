'''
Assume df is a pandas dataframe object of the dataset given
'''

import numpy as np
import pandas as pd
import random


'''Calculate the entropy of the enitre dataset'''
# input:pandas_dataframe
# output:int/float
def get_entropy_of_dataset(df):
    if df is None:
    	return None
    coloumn_of_attribute = df[[df.columns[-1]]].values
    
    unique_attribute_values = np.unique(coloumn_of_attribute)
    entropy=0
    for i in unique_attribute_values:
    	count=0
    	for j in coloumn_of_attribute:
    		if(i==j):
    			count+=1
    	entropy+=count/len(coloumn_of_attribute)*np.log2(count/len(coloumn_of_attribute))
    return -1 *entropy
    
def get_entropy_of_attribute(list_of_attribute):
    if len(list_of_attribute)==0:
    	return None
    coloumn_of_attribute = list_of_attribute
    
    
    unique_attribute_values = np.unique(coloumn_of_attribute)
    entropy=0
    for i in unique_attribute_values:
    	count=0
    	for j in coloumn_of_attribute:
    		if(i==j):
    			count+=1
    	entropy+=count/len(coloumn_of_attribute)*np.log2(count/len(coloumn_of_attribute))
    return -1 *entropy

'''Return avg_info of the attribute provided as parameter'''
# input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
# output:int/float
def get_avg_info_of_attribute(df, attribute):
    if df is None:
    	return None
    attribute_values = df[attribute].values
    total=len(attribute_values)
    unique_attribute_values = np.unique(attribute_values)
    entropy_values_of_attributes=0
    last_coloumn = df[[df.columns[-1]]].values
    for i in unique_attribute_values:
    	yes_no_of_that_vari=[]
    	for j in range(len(last_coloumn)):
    		if(attribute_values[j]==i):
    			yes_no_of_that_vari.append(last_coloumn[j])
    	len_of_list=len(yes_no_of_that_vari)
    	entropy_values_of_attributes+=(len_of_list/total * get_entropy_of_attribute(yes_no_of_that_vari))
    return entropy_values_of_attributes
	
'''Return Information Gain of the attribute provided as parameter'''
# input:pandas_dataframe,str
# output:int/float
def get_information_gain(df, attribute):
    if df is None:
    	return None
    information_gain=get_entropy_of_dataset(df)-get_avg_info_of_attribute(df, attribute)
    return information_gain

def get_selected_attribute(df):
    if df is None:
    	return None
    dict_of_attri = {}
    coloumn_name=[i for i in df.columns]
    coloumn_name=coloumn_name[:-1]
    maximum_val=0;attri=coloumn_name[0]
    for i in coloumn_name:
    	ig=get_information_gain(df,i)
    	dict_of_attri[i]=ig
    	if(ig>maximum_val):
    		attri=i
    		maximum_val=ig
    return (dict_of_attri,attri)

